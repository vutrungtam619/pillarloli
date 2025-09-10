import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import config
from packages import Voxelization, nms_cuda
from utils import project_point_to_camera, Anchors, anchor_target, anchors2bboxes, limit_period

# Inverted Residual (MobileNetV2 style)
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        self.conv = nn.Sequential(
            # pointwise
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # pointwise-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# CBAM (Channel + Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
        x = x * sa
        return x

# Image backbone với CBAM + FPN 
class ImageFeature(nn.Module):
    def __init__(self, new_shape, mean, std, out_channels):
        super(ImageFeature, self).__init__()
        self.new_shape = new_shape
        self.mean = torch.as_tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.as_tensor(std, dtype=torch.float32)[:, None, None]
        
        # Stem (stride 2 → H/2, W/2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage1 (stride 2 → H/4, W/4)
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=3),
        )

        # Stage2 (stride 2 → H/8, W/8)
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 128, stride=2, expand_ratio=3),
            CBAM(128)
        )

        # Stage3 (stride 1 → H/8, W/8)
        self.stage3 = nn.Sequential(
            InvertedResidual(128, 256, stride=1, expand_ratio=3),
            CBAM(256)
        )

        # FPN lateral conv
        self.lateral3 = nn.Conv2d(256, out_channels, 1)
        self.lateral2 = nn.Conv2d(128, out_channels, 1)
        self.lateral1 = nn.Conv2d(64, out_channels, 1)
        self.lateral0 = nn.Conv2d(32, out_channels, 1)

        # Output conv
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def transform_tensor(self, batch_images):
        batch_tensor = []
        for image in batch_images:            
            image_tensor = torch.from_numpy(image).permute(2,0,1).to(dtype=torch.float32).div_(255.0)
            image_tensor = image_tensor.sub_(self.mean).div_(self.std).unsqueeze(0) 
            image_tensor = F.interpolate(image_tensor, size=self.new_shape, mode='bilinear', align_corners=False)
            
            batch_tensor.append(image_tensor.squeeze(0))
        
        batch_tensor = torch.stack(batch_tensor, dim=0) # (B, C, H, W)
        return batch_tensor

    def forward(self, x, device):
        batch_tensor = self.transform_tensor(x).to(device)
        
        c0 = self.stem(batch_tensor)    # (B, 32, H,   W)
        c1 = self.stage1(c0)            # (B, 64, H/2, W/2)
        c2 = self.stage2(c1)            # (B, 128,H/4, W/4)
        c3 = self.stage3(c2)            # (B, 256,H/8, W/8)

        # FPN top-down fusion
        p3 = self.lateral3(c3)               
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        p0 = self.lateral0(c0) + F.interpolate(p1, size=c0.shape[-2:], mode='bilinear', align_corners=False)
        
        del c0, c1, c2, c3, p3, p2, p1, batch_tensor
        # Output at stride 1 resolution
        out = self.out_conv(p0)  # (B, out_channels, H, W)      
            
        return out
    
class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super(PillarLayer, self).__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_num_points, max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        """ Generate pillar from points
        Args:
            batched_pts [list torch.tensor float32, (N, 4)]: list of batch points, each batch have shape (N, 4)
                        
        Returns:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            coors_batch [torch.tensor int64, (p1 + p2 + ... + pb, 1 + 3)]: coordinate of each pillar, 1 is the batch index
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar
        """
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar
    
class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channels, out_channels):
        super().__init__()
        self.out_channel = out_channels
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        
    def select_topk_points(self, pillars, offset_pt_center, mask, k=5):
        """
        Select top-k closest points to the mean center in each pillar.
        Args:
            pillars: ((p1+...+pb), N, 3+...) original points (xyz in first 3 channels)
            offset_pt_center: ((p1+...+pb), N, 3) offset from mean center
            mask: ((p1+...+pb), N) valid points mask
            k: int, number of points to keep
        Returns:
            topk_points_xyz: ((p1+...+pb), k, 3)
            topk_valid_mask: ((p1+...+pb), k)
        """
        num_points = pillars.size(1)
        dists = torch.sum(offset_pt_center ** 2, dim=-1)  # ((p1+...+pb), N)
        
        large_dist = 1e6
        dists = torch.where(mask, dists, large_dist)

        k_actual = min(k, num_points)
        _, indices_topk = torch.topk(dists, k=k_actual, largest=False, dim=-1)
        topk_valid_mask = torch.gather(mask, 1, indices_topk)

        num_pillars, _, xyz_dim = pillars[:, :, :3].shape
        idx_expanded_xyz = indices_topk.unsqueeze(-1).expand(-1, -1, xyz_dim)
        topk_points_xyz = torch.gather(pillars[:, :, :3], dim=1, index=idx_expanded_xyz)

        return topk_points_xyz, topk_valid_mask


    def extract_image_features(self, topk_points_xyz, topk_valid_mask, batch_idx, batch_image_map, batched_image_shape, calib, C, H, W):
        """
        Project top-k points into image plane và lấy image feature bằng grid_sample.
        Args:
            topk_points_xyz: (P_b, k, 3)
            topk_valid_mask: (P_b, k)
            batch_idx: int
            batch_image_map: (1, C, H, W) feature map của ảnh
            calib: dict với 'Tr_velo_to_cam', 'R0_rect', 'P2'
            C, H, W: channel, height, width của image feature map
            batched_image_shape: list[(H_img, W_img)] kích thước gốc ảnh
        Returns:
            features_image: (P_b, C)
        """
        device = topk_points_xyz.device
        P_b, k_used, _ = topk_points_xyz.shape

        pts_flat = topk_points_xyz.reshape(-1, 3)
        valid_flat = topk_valid_mask.reshape(-1)

        if not valid_flat.any():
            return torch.zeros((P_b, C), dtype=torch.float32, device=device)

        pts_flat_valid = pts_flat[valid_flat]
        u, v = project_point_to_camera(pts_flat_valid, calib)

        H_img, W_img = batched_image_shape[batch_idx]
        scale_u = W / W_img
        scale_v = H / H_img
        u_scaled = u * scale_u
        v_scaled = v * scale_v

        inside = (u_scaled >= 0) & (u_scaled <= (W - 1)) & (v_scaled >= 0) & (v_scaled <= (H - 1))
        if not inside.any():
            return torch.zeros((P_b, C), dtype=torch.float32, device=device)

        u_in, v_in = u_scaled[inside], v_scaled[inside]

        u_norm = (u_in / (W - 1)) * 2 - 1
        v_norm = (v_in / (H - 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, -1, 1, 2)  # (1, N_in, 1, 2)

        fmap = batch_image_map  # (1, C, H, W)
        sampled = F.grid_sample(fmap, grid, mode="bilinear", align_corners=True)  # (1, C, N_in, 1)
        sampled = sampled.squeeze(-1).squeeze(0).permute(1, 0)  # (N_in, C)

        pillar_idx_flat = torch.arange(P_b, device=device).unsqueeze(1).expand(-1, k_used).reshape(-1)
        pillar_idx_valid = pillar_idx_flat[valid_flat]
        pillar_idx_inside = pillar_idx_valid[inside]

        sum_per_pillar = torch.zeros((P_b, C), dtype=torch.float32, device=device)
        cnt_per_pillar = torch.zeros((P_b, 1), dtype=torch.float32, device=device)

        if pillar_idx_inside.numel() > 0:
            sum_per_pillar.index_add_(0, pillar_idx_inside, sampled)
            cnt_per_pillar.index_add_(0, pillar_idx_inside, torch.ones((pillar_idx_inside.size(0), 1), device=device))

        cnt_per_pillar = cnt_per_pillar.clamp_min_(1.0)
        features_image = sum_per_pillar / cnt_per_pillar
        
        del pts_flat, valid_flat, u, v, u_scaled, v_scaled, inside, u_in, v_in, u_norm, v_norm, grid, fmap, sampled, pillar_idx_flat, pillar_idx_valid, pillar_idx_inside, sum_per_pillar, cnt_per_pillar

        return features_image

    def forward(self, pillars, coors_batch, npoints_per_pillar, batched_image_map, batched_image_shape, batched_calibs, batch_size):
        """ Encode pillars into BEV feature map of lidar and image
        Args:
            pillars [torch.tensor float32, (p1 + p2 + ... + pb, N, c)]: c is number of features per point
            coors_batch [torch.tensor int64, (p1 + p2 + ... + pb, 1 + 3)]: coordinate of each pillar, 1 is the batch index
            num_points_per_pillar [torch.tensor int64, (p1 + p2 + ... + pb, )]: number of points in each pillar
            batched_image_map [torch.tensor float32, (B, out_channels, H, W)]: image feature
            batched_image_shape [list int32, (B, 2)]: list of image original shape
            batched_calibs [list of dict]: calib info
            batch_size [int32]: batch size
            
        Returns: C is number of out_channels
            batch_canvas_lidar [torch.tensor float32, (B, C, H, W)]: BEV lidar map
            batch_canvas_image [torch.tensor float32, (B, C, H, W)]: BEV image map
        """
        device = pillars.device
        P, N, _ = pillars.shape
        C, H, W = batched_image_map.shape[1:]

        # --- 1. Mean center of each pillar
        mean_center = torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]

        # --- 2. Offset to mean center
        offset_pt_center = pillars[:, :, :3] - mean_center

        # --- 3. Valid mask (exclude zero padding points)
        voxel_ids = torch.arange(N, device=device)
        mask = voxel_ids[None, :] < npoints_per_pillar[:, None]  # (P, N)

        # --- 4. Encode lidar features (same as PointPillars)
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)

        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)
        features[:, :, 0:1] = x_offset_pi_center
        features[:, :, 1:2] = y_offset_pi_center
        features *= mask[:, :, None]

        features = features.permute(0, 2, 1).contiguous()  # (P, C_in, N)
        features = F.relu(self.bn(self.conv(features)))    # (P, C_out, N)
        pooling_features = torch.max(features, dim=-1)[0] # (P, C_out)
        
        # --- 5. Select top-k points per pillar
        topk_points_xyz, topk_valid_mask = self.select_topk_points(pillars, offset_pt_center, mask, k=config['top_k'])

        # --- 6. Scatter lidar & image features into BEV maps
        batched_canvas_lidar, batched_canvas_image = [], []
        for b in range(batch_size):
            cur_mask = coors_batch[:, 0] == b
            cur_coors = coors_batch[cur_mask]
            cur_lidar_feat = pooling_features[cur_mask]

            # scatter lidar
            canvas_lidar = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas_lidar[cur_coors[:, 1], cur_coors[:, 2]] = cur_lidar_feat
            canvas_lidar = canvas_lidar.permute(2, 1, 0).contiguous()
            batched_canvas_lidar.append(canvas_lidar)

            # scatter image
            cur_topk_xyz = topk_points_xyz[cur_mask]
            cur_topk_valid = topk_valid_mask[cur_mask]

            cur_img_feat = self.extract_image_features(cur_topk_xyz, cur_topk_valid, b, batched_image_map[b:b+1], batched_image_shape, batched_calibs[b], C, H, W)

            canvas_image = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas_image[cur_coors[:, 1], cur_coors[:, 2]] = cur_img_feat
            canvas_image = canvas_image.permute(2, 1, 0).contiguous()
            batched_canvas_image.append(canvas_image)

        batched_canvas_lidar = torch.stack(batched_canvas_lidar, dim=0)
        batched_canvas_image = torch.stack(batched_canvas_image, dim=0)

        return batched_canvas_lidar, batched_canvas_image

class ImageDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernels=[3, 7, 15], pool_strides=[2, 4, 8]):
        super().__init__()
        assert len(out_channels) == len(pool_kernels) == len(pool_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            blocks = []
            blocks.append(nn.MaxPool2d(kernel_size=pool_kernels[i], stride=pool_strides[i], padding=pool_kernels[i] // 2))
            blocks.append(nn.Conv2d(in_channels, out_channels[i], kernel_size=1, bias=False))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            self.multi_blocks.append(nn.Sequential(*blocks))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, R0):
        """ Downsampling BEV image map into 3 blocks R1, R2, R3
        Args:
            R0 [torch.tensor, (B, 64, 496, 432)]: BEV image
        
        Returns: list of the following blocks
            R1 [torch.tensor float32, (B, 64, 248, 216)]
            R2 [torch.tensor float32, (B, 128, 124, 108)]
            R3 [torch.tensor float32, (B, 256, 62, 54)]
        """
        
        outs = []
        for block in self.multi_blocks:
            outs.append(block(R0))
        return outs
        
class LidarDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels, layer_nums, layer_strides):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channels, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channels = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, C0):
        """ Downsampling BEV lidar map into 3 blocks C1, C2, C3
        Args:
            C0 [torch.tensor, (B, 64, 496, 432)]: BEV lidar
        
        Returns: list of the following blocks
            C1 [torch.tensor float32, (B, 64, 248, 216)]
            C2 [torch.tensor float32, (B, 128, 124, 108)]
            C3 [torch.tensor float32, (B, 256, 62, 54)]
        """
        outs = []
        for i in range(len(self.multi_blocks)):
            C0 = self.multi_blocks[i](C0)
            outs.append(C0)
        return outs

class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.fusion_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels[i], kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels[i]),
                    nn.ReLU(inplace=True)
                )
            )
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, C_blocks, R_blocks):
        """ Fusion two blocks on each stage
        Args:
            C_blocks [list torch.tensor, [(B, 64, 248, 216), (B, 128, 124, 108), (B, 256, 62, 54)]]: lidar blocks
            R_blocks [list torch.tensor, [(B, 64, 248, 216), (B, 128, 124, 108), (B, 256, 62, 54)]]: image blocks

        Returns:
            CR_blocks [list torch.tensor, [(B, 64, 248, 216), (B, 128, 124, 108), (B, 256, 62, 54)]]: fusion blocks
        """
        CR_blocks = []
        for i, fusion in enumerate(self.fusion_blocks):
            C, R = C_blocks[i], R_blocks[i]
            CR = torch.cat([C, R], dim=1)  # concat
            CR = fusion(CR)               # conv1x1
            CR_blocks.append(CR)
        return CR_blocks
    
class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], out_channels[i], upsample_strides[i], stride=upsample_strides[i], bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        Args: 
            x [torch.t]
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out
    
class Head(nn.Module):
    def __init__(self, in_channels, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channels, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channels, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channels, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred
    
class Pointpillars(nn.Module):
    def __init__(
        self,
        nclasses=config['num_classes'], 
        voxel_size=config['voxel_size'],
        point_cloud_range=config['pc_range'],
        max_num_points=config['max_points'],
        max_voxels=config['max_voxels'],
        new_shape = config['new_shape'], 
        mean = config['mean'],
        std = config['std'], 
    ):
        super(Pointpillars, self).__init__()
        
        self.nclasses = nclasses
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_feature = ImageFeature(
            new_shape=new_shape,
            mean=mean,
            std=std,
            out_channels=64,
        )
        
        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size, 
            point_cloud_range=point_cloud_range, 
            max_num_points=max_num_points, 
            max_voxels=max_voxels
        )
        
        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size, 
            point_cloud_range=point_cloud_range, 
            in_channels=9, 
            out_channels=64
        )
        
        self.image_down = ImageDownsampling(
            in_channels=64,
            out_channels=[64, 128, 256],
            pool_kernels=[3, 7, 15],
            pool_strides=[2, 4, 8]
        )
        
        self.lidar_down = LidarDownsampling(
            in_channels=64, 
            out_channels=[64, 128, 256], 
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2]
        )
        
        self.backbone = Backbone(
            in_channels=[128, 256, 512],
            out_channels=[64, 128, 256]
        )
        
        self.neck = Neck(
            in_channels=[64, 128, 256], 
            upsample_strides=[1, 2, 4], 
            out_channels=[128, 128, 128]
        )
        
        self.head = Head(
            in_channels=384, 
            n_anchors=2 * nclasses, 
            n_classes=nclasses
        )
        
        self.anchors_generator = Anchors(
            ranges=config['anchor_ranges'], 
            sizes=config['anchor_sizes'], 
            rotations=config['anchor_rotations']
        )
        
        self.assigners = [
            {'pos_iou_thr': 0.4, 'neg_iou_thr': 0.25, 'min_iou_thr': 0.25},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        # val and test
        self.nms_pre = 500
        self.nms_thr = 0.1
        self.score_thr = 0.3
        self.max_num = 30
        
    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * torch.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            result = {
                'lidar_bboxes': np.zeros((0, 7), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int64),
                'scores': np.zeros((0,), dtype=np.float32)
            }
            return result
        
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results
    
    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None, batched_images=None, batched_image_shape=None, batched_calibs=None):
        batch_size = len(batched_pts)
        
        image_feature = self.image_feature(batched_images, self.device)
        
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        
        bev_lidar, bev_image = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar, image_feature, batched_image_shape, batched_calibs, batch_size)
        
        image_blocks = self.image_down(bev_image)
        
        lidar_blocks = self.lidar_down(bev_lidar)
        
        backbone = self.backbone(lidar_blocks, image_blocks)
        
        neck = self.neck(backbone)
        
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(neck)
        
        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors, 
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_gt_labels, 
                assigners=self.assigners,
                nclasses=self.nclasses
            )
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        
        elif mode == 'val':
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors, 
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_gt_labels, 
                assigners=self.assigners,
                nclasses=self.nclasses
            )
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict

        elif mode == 'test':
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred, 
                bbox_pred=bbox_pred, 
                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                batched_anchors=batched_anchors
            )
            
            return results
        
        else:
            raise ValueError