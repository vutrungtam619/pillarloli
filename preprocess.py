import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from configs.config import config

from utils import read_calib, read_points, read_label, write_points, write_pickle, remove_outside_points, remove_outside_bboxes, get_points_num_in_bbox

root = os.path.dirname(os.path.abspath(__file__))

def judge_difficulty(annotation_dict):
    """ classify difficulty level of each object base on KITTI bench mark
    Args: Dict with the following key, m is the number of ground truth objects in sample
        name [np.ndarray string, (m, )]: name of the object category in image, include Car, Pedestrian, Cyclist, Dontcare
        truncated [np.ndarray float32, (m, )]: how much the object extend outside the image, from 0.0 -> 1.0
        occluded [np.ndarray float32, (m, )]: how much the objet is block by others, from 0 -> 3 (fully visible -> unknow)
        alpha [np.ndarray float32, (m, )]: observation agle of the object in camera coordinate (radian)
        bbox [np.ndarray float32, (m, 4)]: 2d bounding box in x_min, y_min, x_max, y_max
        dimension [np.ndarray float32, (m, 3)]: 3d dimension in height, width, length
        location [np.ndarray float32, (m, 3)]: 3d location of the object center in camera coordinate, include x, y, z (right, down, forward)
        rotation_y [np.ndarray float32, (m, )]: rotation of the object around y-axis (radian)

    Returns:
        difficultys [np.ndarray int32, (m, )]: 0 is easy, 1 is moderate, 2 is hard, -1 is not classify
    """
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
        
    return np.array(difficultys, dtype=np.int32)

def remove_dontcare(annos_info):
    keep_ids = [i for i, name in enumerate(annos_info['name']) if name not in ['DontCare', 'Misc', 'Truck', 'Tram', 'Person_sitting']]
    for k, v in annos_info.items():
        annos_info[k] = v[keep_ids]
    return annos_info

def create_data_info_pkl(data_root, data_type, label):
    """ convert data into pickle file for fast reading & training
    Args:
        data_root [string]: path to dataset kitti
        data_type [string]: type of file pickle, include train, val, trainval, test
        label [bool]: only train and val have label

    Returns: file pickle, with each id is key, following by these values
        'index': id of the sample
        'image': include image_shape, image_path
        'calib': inlcude P0, P1, P2, P3, R0_rect, tr_velo_to_cam, tr_imu_to_velo
        'point_path': path to point cloud file after redueced (only in image):
        'annos': annotation dict
    """
    print(f"Processing {data_type} data into pkl file....")
    sep = os.path.sep
    
    ids_file = os.path.join(root, 'index', f'{data_type}.txt') # Path to txt file index
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()] # List of index in string 
    
    split = 'training' if label else 'testing' 
    
    lidar_reduced_folder = os.path.join(root, 'datasets', 'lidar_reduced', split)
    os.makedirs(lidar_reduced_folder, exist_ok=True)
    
    kitti_infos_dict = {}
    for id in tqdm(ids):        
        cur_info_dict = {}
        
        # store path
        image_path = os.path.join(data_root, split, 'image_2', f'{id}.png')
        lidar_path = os.path.join(data_root, split, 'velodyne', f'{id}.bin')
        calib_path = os.path.join(data_root, split, 'calib', f'{id}.txt')

        # save index
        cur_info_dict['index'] = int(id)
                
        # save image infor
        image = cv2.imread(image_path)   
        image_shape = image.shape[:2]
        cur_info_dict['image'] = {
            'image_shape': image_shape,
            'image_path': sep.join(image_path.split(sep)[-3:]), # Example: training/image_2/000001.png
        }
        
        # save calib info
        calib_dict = read_calib(calib_path)
        cur_info_dict['calib'] = calib_dict        
        
        # save lidar info
        lidar_reduced_path = os.path.join(lidar_reduced_folder, f'{id}.bin')
        lidar_points = read_points(lidar_path)
        reduced_points = remove_outside_points(
            points=lidar_points, 
            r0_rect=calib_dict['R0_rect'], 
            tr_velo_to_cam=calib_dict['Tr_velo_to_cam'], 
            P2=calib_dict['P2'], 
            image_shape=image_shape
        )
        write_points(lidar_reduced_path, reduced_points)
        cur_info_dict['lidar'] = {
            'lidar_total': reduced_points.shape[0],
            'lidar_path': lidar_reduced_path,
        }
        
        if label:
            label_path = os.path.join(data_root, split, 'label_2', f'{id}.txt')
            annotation_dict = read_label(label_path)
            
            annotation_dict = remove_dontcare(annotation_dict)
            annotation_dict = remove_outside_bboxes(
                annotation_dict,
                r0_rect=calib_dict['R0_rect'],
                tr_velo_to_cam=calib_dict['Tr_velo_to_cam'],
                P2=calib_dict['P2'],
                image_shape=image_shape
            )

            annotation_dict['difficulty'] = judge_difficulty(annotation_dict) 
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                points=reduced_points,
                r0_rect=calib_dict['R0_rect'],
                tr_velo_to_cam=calib_dict['Tr_velo_to_cam'],
                dimensions=annotation_dict['dimension']  ,
                location=annotation_dict['location']    ,
                rotation_y=annotation_dict['rotation_y']  ,
                name=annotation_dict['name']  
            ) 
            
            cur_info_dict['annos'] = annotation_dict
        
        kitti_infos_dict[int(id)] = cur_info_dict

    save_pkl_path = os.path.join(root, 'datasets', f'kitti_infos_{data_type}.pkl')
    write_pickle(save_pkl_path, kitti_infos_dict)  
    
    return None
    
def main(args):
    data_root = args.data_root
    
    kitti_train_infos_dict = create_data_info_pkl(data_root, data_type='train', label=True)
    kitti_val_infos_dict = create_data_info_pkl(data_root, data_type='val', label=True)

    print("......Processing finished!!!")  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default=config['data_root'], help='your data root for kitti')
    args = parser.parse_args()

    main(args)
