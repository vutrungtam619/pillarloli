import numpy as np
import os
import cv2
from configs.config import config
from torch.utils.data import Dataset
from utils import read_pickle, read_points, bbox_camera2lidar
from datasets import train_data_aug, val_data_aug

root = os.path.dirname(os.path.dirname(__file__))

class Kitti(Dataset): 
    def __init__(self, data_root, split):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.data_infos = read_pickle(os.path.join(root, 'datasets', f'kitti_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())        
        self.classes = config['classes']

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        """ Get the information of one item
        Args:
            index [int]: index of item

        Returns: dict with the following, m is the number of objects in item
            pts [np.ndarray float32, (n, 4)]: LiDAR points in this item
            gt_bboxes_3d [np.ndarray float32, (m, 7)]: bounding box in LiDAR coordinate
            gt_labels [np.ndarray int32, (m, )]: numerical labels for each object
            gt_names [np.ndarray string, (m, )]: object class name
            difficulty [np.ndarray float32, (m, )]: 0 is easy, 1 is moderate, 2 is hard, -1 is not classify
            image_shape [tuple int32, (2)]: image shape in (height, width)
            image [np.ndarray float32]: image
            calib_info [dict]: calib information
        """
        
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info, lidar_info = data_info['image'], data_info['calib'], data_info['annos'], data_info['lidar']
        idx = data_info['index']
        pts = read_points(lidar_info['lidar_path']).astype(np.float32)    
        Tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        R0_rect = calib_info['R0_rect'].astype(np.float32)
        
        names = annos_info['name']
        locations = annos_info['location']
        dimensions = annos_info['dimension']
        rotation_y = annos_info['rotation_y']
        difficulty = annos_info['difficulty']
        
        gt_bboxes = np.concatenate([locations, dimensions, rotation_y[:, None]], axis=1)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, Tr_velo_to_cam, R0_rect)     
        gt_labels = np.array([self.classes.get(name, -1) for name in names])
        
        image_shape = image_info['image_shape']
        image_path = os.path.join(self.data_root, image_info['image_path'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels,
            'gt_names': names,
            'difficulty': difficulty,
            'image_shape': image_shape,
            'image': image,
            'calib_info': calib_info,
            'index': idx
        }
        
        if self.split in ['train', 'trainval']:
            data_dict = train_data_aug(data_dict)
        else:
            data_dict = val_data_aug(data_dict)        
            
        return data_dict

    def __len__(self):
        return len(self.data_infos)