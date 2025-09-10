import argparse
import os
import cv2
import torch
import numpy as np

from configs.config import config
from datasets import Kitti
from models import Pointpillars
from utils import bbox3d2corners, points_lidar2image, vis_img_3d, keep_bbox_from_image_range, keep_bbox_from_lidar_range, bbox3d2corners_camera, points_camera2image


def process_sample(idx, sample, model, device, CLASSES, save_dir):
    pts = sample['pts']
    gt_bboxes_3d = sample['gt_bboxes_3d']
    gt_labels = sample['gt_labels']
    calib_info = sample['calib_info']
    img = sample['image']
    img_shape = sample['image_shape']

    tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
    r0_rect = calib_info['R0_rect'].astype(np.float32)
    P2 = calib_info['P2'].astype(np.float32)

    # convert points to tensor
    pc_torch = torch.from_numpy(pts).to(device)

    with torch.no_grad():
        pred = model(
            batched_pts=[pc_torch],
            batched_images=[img],
            batched_calibs=[calib_info],
            batched_image_shape=[img_shape],
            mode='test'
        )[0]

    # ===== Prediction filtering =====
    result = {
        'lidar_bboxes': pred['lidar_bboxes'],
        'scores': pred['scores'],
        'labels': pred['labels']
    }

    # 1. lọc theo ảnh
    result = keep_bbox_from_image_range(
        result, tr_velo_to_cam, r0_rect, P2, img_shape
    )

    # 2. lọc theo range point cloud
    result = keep_bbox_from_lidar_range(
        result, np.array(config['pc_range'])
    )

    pred_scores = result['scores']
    pred_labels = result['labels']
    pred_bboxes = result['camera_bboxes']

    pred_corners = bbox3d2corners_camera(pred_bboxes)
    pred_image_points = points_camera2image(pred_corners, P2)

    # ===== Ground truth (GT) =====
    gt_corners = bbox3d2corners(gt_bboxes_3d)
    gt_img_pts = points_lidar2image(gt_corners, tr_velo_to_cam, r0_rect, P2)

    img_vis = vis_img_3d(
        img.copy(),
        gt_image_points=gt_img_pts,
        gt_labels=gt_labels,
        pred_image_points=pred_image_points,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        rt=True
    )

    cv2.imshow("3D Detection", cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # save and continue
            save_path = os.path.join(save_dir, f"{idx:06d}.png")
            cv2.imwrite(save_path, img_vis)
            print(f"Saved image to {save_path}")
            return True
        elif key == 27:  # ESC to quit
            return False


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    dataset = Kitti(config['data_root'], split='val')

    # Load model
    model = Pointpillars().to(device)
    checkpoint_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint_dict['checkpoint'])
    model.eval()
    print("Model loaded successfully.")

    for idx in dataset.sorted_ids:
        if idx < args.start_idx:
            continue
        print(f"Processing sample {idx}")
        sample = dataset[dataset.sorted_ids.index(idx)]
        if not process_sample(idx, sample, model, device, config['classes'], save_dir):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/epoch_40.pth')
    parser.add_argument('--start_idx', type=int, default=1)
    args = parser.parse_args()
    main(args)
