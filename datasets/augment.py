import numpy as np
import cv2
from configs.config import config

def random_flip_fusion(data_dict, flip_prob=0.5):
    if np.random.rand() >= flip_prob:
        return data_dict
    pts = data_dict['pts']
    gt_bboxes_3d = data_dict['gt_bboxes_3d']
    pts[:, 1] *= -1
    gt_bboxes_3d[:, 1] *= -1
    gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6]
    data_dict['pts'] = pts
    data_dict['gt_bboxes_3d'] = gt_bboxes_3d
    image = data_dict['image']
    data_dict['image'] = cv2.flip(image, 1).copy()
    calib = data_dict['calib_info']
    P2 = calib['P2']
    w = image.shape[1]
    P2[0, 2] = w - P2[0, 2]   # cx -> W - cx
    P2[0, 3] *= -1            # Tx -> -Tx
    data_dict['calib_info'] = calib
    return data_dict

def color_jitter_fusion(data_dict, brightness=0.2, contrast=0.2, saturation=0.2, prob=0.5):
    if np.random.rand() > prob:
        return data_dict
    image = data_dict['image'].astype(np.float32) / 255.0
    mean = image.mean(axis=(0, 1), keepdims=True)
    gray = image.mean(axis=2, keepdims=True)
    if brightness > 0:
        image *= 1.0 + np.random.uniform(-brightness, brightness)
    if contrast > 0:
        image = (image - mean) * (1.0 + np.random.uniform(-contrast, contrast)) + mean
    if saturation > 0:
        image = (image - gray) * (1.0 + np.random.uniform(-saturation, saturation)) + gray
    data_dict['image'] = np.clip(image * 255, 0, 255).astype(np.uint8)
    return data_dict

def random_point_dropout(data_dict, max_dropout_ratio=0.2, prob=0.5):
    if np.random.rand() > prob:
        return data_dict
    pts = data_dict['pts']
    N = pts.shape[0]
    dropout_ratio = np.random.uniform(0, max_dropout_ratio)
    keep_idx = np.random.choice(N, int(N * (1 - dropout_ratio)), replace=False)
    data_dict['pts'] = pts[keep_idx]
    return data_dict

def point_jitter(data_dict, sigma=0.01, clip=0.05, prob=0.5):
    if np.random.rand() > prob:
        return data_dict
    pts = data_dict['pts']
    jitter = np.clip(sigma * np.random.randn(*pts.shape), -clip, clip)
    pts[:, :3] += jitter[:, :3]   # chỉ x,y,z
    data_dict['pts'] = pts
    return data_dict

def point_range_filter(data_dict, point_range=config['pc_range']):
    pts = data_dict['pts']
    mask = (
        (pts[:, 0] > point_range[0]) &
        (pts[:, 1] > point_range[1]) &
        (pts[:, 2] > point_range[2]) &
        (pts[:, 0] < point_range[3]) &
        (pts[:, 1] < point_range[4]) &
        (pts[:, 2] < point_range[5])
    )
    data_dict['pts'] = pts[mask]
    return data_dict

def object_range_filter(data_dict, object_range=config['pc_range']):
    gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], data_dict['gt_labels']
    gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']
    mask = (
        (gt_bboxes_3d[:, 0] > object_range[0]) &
        (gt_bboxes_3d[:, 1] > object_range[1]) &
        (gt_bboxes_3d[:, 0] < object_range[3]) &
        (gt_bboxes_3d[:, 1] < object_range[4])
    )
    data_dict['gt_bboxes_3d'] = gt_bboxes_3d[mask]
    data_dict['gt_labels'] = gt_labels[mask]
    data_dict['gt_names'] = gt_names[mask]
    data_dict['difficulty'] = difficulty[mask]
    return data_dict

def points_shuffle(data_dict):
    pts = data_dict['pts']
    idx = np.arange(len(pts))
    np.random.shuffle(idx)
    data_dict['pts'] = pts[idx]
    return data_dict

def train_data_aug(data_dict):
    data_dict = random_flip_fusion(data_dict)
    data_dict = color_jitter_fusion(data_dict)
    data_dict = random_point_dropout(data_dict)
    data_dict = point_jitter(data_dict)
    data_dict = point_range_filter(data_dict)
    data_dict = object_range_filter(data_dict)
    data_dict = points_shuffle(data_dict)
    return data_dict

def val_data_aug(data_dict):
    data_dict = point_range_filter(data_dict)
    data_dict = object_range_filter(data_dict)
    return data_dict