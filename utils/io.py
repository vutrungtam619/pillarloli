import numpy as np
import pickle
import os

def read_points(file_path, dim=4, suffix = '.bin'):
    assert os.path.splitext(file_path)[1] == suffix
    
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)

def write_points(file_path, data):
    with open(file_path, 'w') as f:
        data.tofile(f)
    return None
        
def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(file_path, results):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    return None

def read_label(file_path, suffix = '.txt'):
    """ Read label file, each line is one object in image
    Args:
        file_path [string]: path to label file (txt)
    
    Returns: Dict with following key, m is the number of objects in 1 sample
        name [np.ndarray string, (m, )]: name of the object category in image, include Car, Pedestrian, Cyclist, DontCare
        truncated [np.ndarray float32, (m, )]: how much the object extend outside the image, from 0.0 -> 1.0
        occluded [np.ndarray float32, (m, )]: how much the objet is block by others, from 0 -> 3 (fully visible -> unknow)
        alpha [np.ndarray float32, (m, )]: observation agle of the object in camera coordinate (radian)
        bbox [np.ndarray float32, (m, 4)]: 2d bounding box in x_min, y_min, x_max, y_max
        dimension [np.ndarray float32, (m, 3)]: 3d dimension in legnth, height, width
        location [np.ndarray float32, (m, 3)]: 3d location of the object center in camera coordinate, include x, y, z (right, down, forward)
        rotation_y [np.ndarray float32, (m, )]: rotation of the object around y-axis (radian)
    """
    assert os.path.splitext(file_path)[1] == suffix
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    
    annotation = {}
    annotation['name'] = np.array([line[0] for line in lines])
    annotation['truncated'] = np.array([line[1] for line in lines], dtype=np.float32)
    annotation['occluded'] = np.array([line[2] for line in lines], dtype=np.int32)
    annotation['alpha'] = np.array([line[3] for line in lines], dtype=np.float32)
    annotation['bbox'] = np.array([line[4:8] for line in lines], dtype=np.float32)
    annotation['dimension'] = np.array([line[8:11] for line in lines], dtype=np.float32)[:, [2, 0, 1]] # hwl -> camera coordinates (lhw)
    annotation['location'] = np.array([line[11:14] for line in lines], dtype=np.float32)
    annotation['rotation_y'] = np.array([line[14] for line in lines], dtype=np.float32)
    
    return annotation

def write_label(file_path, result):    
    name = result['name']
    truncated = result['truncated']
    occluded = result['occluded']
    alpha = result['alpha']
    bbox = result['bbox']
    dimensions = result['dimension']
    location = result['location']
    rotation_y = result['rotation_y']
    score = result['score']
    
    with open(file_path, 'w') as f:
        for i in range(len(name)):
            bbox_str = ' '.join(map(str, bbox[i]))
            hwl = ' '.join(map(str, dimensions[i]))
            xyz = ' '.join(map(str, location[i]))
            line = f'{name[i]} {truncated[i]} {occluded[i]} {alpha[i]} {bbox_str} {hwl} {xyz} {rotation_y[i]} {score[i]}\n'
            f.writelines(line)
            
    return None

def read_calib(file_path, extend_matrix=True):
    """ Read calib file
    Args:
        file_path [string]: path to calib file (txt)
        extend_matrix [bool]: choose to convert to homogeneous matrix
    
    Returns: Dict with following keys
        In case extend_matrix = False:
        P0, P1, P2, P3 [np.ndarray float32, (3, 4)]: projection matrix, convert 3D camera coordinate to 2D pixel coordinate
        R0_rect [np.ndarray float32, (3, 3)]: rectification matrix of camera, allign all cameras into a common coordinate
        Tr_velo_to_cam [np.ndarray float32, (3, 4)]: transformation matrix from LiDAR to camera
        Tr_imu_to_velo [np.ndarray float32, (3, 4)]: convert IMU coordinate to lidar coordinate
        
        In case extend_matrix = True: 
        P0, P1, P2, P3 [np.ndarray float32, (4, 4)]: projection matrix, convert 3D camera coordinate to 2D pixel coordinate
        R0_rect [np.ndarray float32, (4, 4)]: rectification matrix of camera, allign all cameras into a common coordinate
        Tr_velo_to_cam [np.ndarray float32, (4, 4)]: transformation matrix from LiDAR to camera
        Tr_imu_to_velo [np.ndarray float32, (4, 4)]: convert IMU coordinate to lidar coordinate        
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    P0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    R0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=np.float32).reshape(3, 3)
    Tr_velo_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    Tr_imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict=dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo
    )
    
    return calib_dict 