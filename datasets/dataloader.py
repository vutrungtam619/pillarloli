import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler


def collate_fn(list_data):
    batched_pts_list = []
    batched_gt_bboxes_list = []
    batched_labels_list = []
    batched_names_list = []
    batched_difficulty_list = []
    batched_image_shape = []
    batched_images = []
    batched_calib_list = []
    batched_idx = []
    
    for data_dict in list_data:
        pts = data_dict['pts']
        gt_bboxes_3d = data_dict['gt_bboxes_3d']
        gt_labels = data_dict['gt_labels']
        gt_names = data_dict['gt_names']
        difficulty = data_dict['difficulty']
        image_shape = data_dict['image_shape'] 
        image = data_dict['image']
        calib_info = data_dict['calib_info']
        idx = data_dict['index']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names) 
        batched_difficulty_list.append(torch.from_numpy(difficulty))
        batched_image_shape.append(image_shape)
        batched_images.append(image)
        batched_calib_list.append(calib_info)
        batched_idx.append(idx)
    
    rt_data_dict = dict(
        batched_pts=batched_pts_list, 
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_difficulty=batched_difficulty_list,
        batched_image_shape=batched_image_shape,
        batched_images=batched_images,
        batched_calib=batched_calib_list,
        batched_idx=batched_idx
    )

    return rt_data_dict

def get_train_dataloader(dataset, batch_size, num_workers, drop_last=False, shuffle=True):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader


def get_val_dataloader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader

