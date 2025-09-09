import cv2, torch
import numpy as np
from datasets import Kitti
from configs.config import config
from utils import bbox3d2corners, points_lidar2image, project_point_to_camera, points_camera2lidar, get_frustum, projection_matrix_to_CRT_kitti, vis_pc, vis_img_3d

if __name__ == '__main__':
    # load sample
    start_id = 1
    
    train, val = Kitti(config['data_root'],'train'), Kitti(config['data_root'],'val')
    dataset = train if start_id in train.sorted_ids else val
    
    sample = dataset[dataset.sorted_ids.index(start_id)]
    pts, gt_bboxes_3d, gt_labels, calib, img, img_shape = sample['pts'], sample['gt_bboxes_3d'], sample['gt_labels'], sample['calib_info'], sample['image'], sample['image_shape']
    Tr, R0, P2 = calib['Tr_velo_to_cam'], calib['R0_rect'], calib['P2']

    # bbox corners
    gt_corners = bbox3d2corners(gt_bboxes_3d)

    # frustum in lidar
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    frustum_cam = get_frustum([0,0,img_shape[1],img_shape[0]],C)
    frustum_lidar = points_camera2lidar(frustum_cam[None,...],Tr,R0)[0]

    # open3d: pc + gt bbox + frustum
    vis_pc(pts, gt_bboxes=gt_corners, gt_labels=gt_labels, pred_bboxes=[frustum_lidar], pred_labels=[-1])

    # project point cloud to image
    u, v = project_point_to_camera(torch.from_numpy(pts[:,:3]),calib)
    u, v = u.numpy(), v.numpy()
    mask = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
    num_outside = np.sum(~mask)
    print(f"Tổng số điểm point cloud: {pts.shape[0]}")
    print(f"Số điểm nằm ngoài ảnh: {num_outside}")
    img_pc = img.copy()
    [cv2.circle(img_pc, (int(x),int(y)), 1, (0,255,0), -1) for x, y in zip(u[mask],v[mask])]
    gt_img_pts = points_lidar2image(gt_corners, Tr, R0, P2)
    img_vis = vis_img_3d(img_pc, gt_image_points = gt_img_pts, gt_labels = gt_labels)
    
    cv2.imshow("pc+gt", cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()