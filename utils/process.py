import copy
import numba
import numpy as np
import random
import torch
from packages import boxes_overlap_bev, boxes_iou_bev

def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)


def bbox_lidar2camera(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([y_size, z_size, x_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = r0_rect @ tr_velo_to_cam
    xyz = extended_xyz @ rt_mat.T
    bboxes_camera = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return bboxes_camera


def points_camera2image(points, P2):
    '''
    points: shape=(N, 8, 3) 
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    '''
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0) # (n, 8, 4)
    image_points = extended_points @ P2.T # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]
    return image_points


def points_lidar2image(points, tr_velo_to_cam, r0_rect, P2):
    '''
    points: shape=(N, 8, 3) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    '''
    # points = points[:, :, [1, 2, 0]]
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0) # (N, 8, 4)
    rt_mat = r0_rect @ tr_velo_to_cam
    camera_points = extended_points @ rt_mat.T # (N, 8, 4)
    # camera_points = camera_points[:, :, [1, 2, 0, 3]]
    image_points = camera_points @ P2.T # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]

    return image_points


def points_camera2lidar(points, tr_velo_to_cam, r0_rect):
    """Convert 3D points from camera coordinates to LiDAR coordinates.
    Args:
        points [np.ndarray float32, (N, 8, 3)]: input 3D points in camera coordinates, usually the 8 corners of N bounding boxes.
        tr_velo_to_cam [np.ndarray float32, (4, 4)]: transformation matrix from LiDAR to camera.
        r0_rect [np.ndarray float32, (4, 4)]: rectification matrix aligning camera coordinates into a common reference frame.
        
    Returns:
        xyz [np.ndarray float32, (N, 8, 3)]: transformed 3D points in LiDAR coordinates.
    """
    extended_xyz = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    return xyz[..., :3]


def bbox3d2bevcorners(bboxes):
    '''
    bboxes: shape=(n, 7)

                ^ x (-0.5 * pi)
                |
                |                (bird's eye view)
       (-pi)  o |
        y <-------------- (0)
                 \ / (ag)
                  \ 
                   \ 

    return: shape=(n, 4, 2)
    '''
    centers, dims, angles = bboxes[:, :2], bboxes[:, 3:5], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bev_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    bev_corners = bev_corners[None, ...] * dims[:, None, :] # (1, 4, 2) * (n, 1, 2) -> (n, 4, 2)

    # 2. rotate
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (N, 2, 2)
    bev_corners = bev_corners @ rot_mat # (n, 4, 2)

    # 3. translate to centers
    bev_corners += centers[:, None, :] 
    return bev_corners.astype(np.float32)


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def bbox3d2corners_camera(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
        z (front)            6 ------ 5
        /                  / |     / |
       /                  2 -|---- 1 |   
      /                   |  |     | | 
    |o ------> x(right)   | 7 -----| 4
    |                     |/   o   |/    
    |                     3 ------ 0 
    |
    v y(down)                   
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[0.5, 0.0, -0.5], [0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, 0.0, -0.5],
                               [0.5, 0.0, 0.5], [0.5, -1.0, 0.5], [-0.5, -1.0, 0.5], [-0.5, 0.0, 0.5]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around y axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, angle
    rot_mat = np.array([[rot_cos, np.zeros_like(rot_cos), rot_sin],
                        [np.zeros_like(rot_cos), np.ones_like(rot_cos), np.zeros_like(rot_cos)],
                        [-rot_sin, np.zeros_like(rot_cos), rot_cos]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def group_rectangle_vertexs(bboxes_corners):
    '''
    bboxes_corners: shape=(n, 8, 3)
    return: shape=(n, 6, 4, 3)
    '''
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]], axis=1) # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]], axis=1) # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]], axis=1) # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]], axis=1) # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs


@numba.jit(nopython=True)
def bevcorner2alignedbbox(bev_corners):
    '''
    bev_corners: shape=(N, 4, 2)
    return: shape=(N, 4)
    '''
    # xmin, xmax = np.min(bev_corners[:, :, 0], axis=-1), np.max(bev_corners[:, :, 0], axis=-1)
    # ymin, ymax = np.min(bev_corners[:, :, 1], axis=-1), np.max(bev_corners[:, :, 1], axis=-1)

    # why we don't implement like the above ? please see
    # https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html#calculation
    n = len(bev_corners)
    alignedbbox = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        cur_bev = bev_corners[i]
        alignedbbox[i, 0] = np.min(cur_bev[:, 0])
        alignedbbox[i, 2] = np.max(cur_bev[:, 0])
        alignedbbox[i, 1] = np.min(cur_bev[:, 1])
        alignedbbox[i, 3] = np.max(cur_bev[:, 1])
    return alignedbbox


# modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/data_augment_utils.py#L31
@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.
    Args:
        boxes (np.ndarray): Corners of current boxes. # (n1, 4, 2)
        qboxes (np.ndarray): Boxes to be avoid colliding. # (n2, 4, 2)
        clockwise (bool, optional): Whether the corners are in
            clockwise order. Default: True.
    return: shape=(n1, n2)
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = bevcorner2alignedbbox(boxes)
    qboxes_standup = bevcorner2alignedbbox(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                    min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                    max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                                       D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                                       D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


def group_plane_equation(bbox_group_rectangle_vertexs):
    """Compute plane equations from grouped rectangle vertices of 3D bounding boxes.
    Args:
        bbox_group_rectangle_vertexs [np.ndarray float32, (n, 6, 4, 3)]: vertices of all bounding box faces, where n = number of boxes, each box has 6 faces, each face has 4 vertices, and each vertex has (x, y, z).
        
    Returns:
        plane_equation_params [np.ndarray float32, (n, 6, 4)]: plane equation coefficients (a, b, c, d) of each face, representing ax + by + cz + d = 0.
    """
    # 1. generate vectors for a x b
    vectors = bbox_group_rectangle_vertexs[:, :, :2] - bbox_group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1]) # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', bbox_group_rectangle_vertexs[:, :, 0], normal_vectors) # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params


@numba.jit(nopython=True)
def points_in_bboxes(points, plane_equation_params):
    """ Check whether each 3D point lies inside given 3D bounding boxes using plane equations. 
    Args:
        points [np.ndarray float32, (N, 3)]: input 3D points in LiDAR coordinates.
        plane_equation_params [np.ndarray float32, (n, 6, 4)]: plane equation coefficients of all bounding boxes, where n = number of boxes, each box has 6 faces, each face is represented by (a, b, c, d).
        
    Returns:
        masks [np.ndarray bool, (N, n)]: boolean mask where masks[i, j] is True if point i is inside bounding box j, False otherwise.
    """
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks


def remove_pts_in_bboxes(points, bboxes, rm=True):
    '''
    points: shape=(N, 3)
    bboxes: shape=(n, 7)
    return: shape=(N, n), bool
    '''
    # 1. get 6 groups of rectangle vertexs
    bboxes_corners = bbox3d2corners(bboxes) # (n, 8, 3)
    bbox_group_rectangle_vertexs = group_rectangle_vertexs(bboxes_corners) # (n, 6, 4, 3)

    # 2. calculate plane equation: ax + by + cd + d = 0
    group_plane_equation_params = group_plane_equation(bbox_group_rectangle_vertexs)

    # 3. Judge each point inside or outside the bboxes
    # if point (x0, y0, z0) lies on the direction of normal vector(a, b, c), then ax0 + by0 + cz0 + d > 0.
    masks = points_in_bboxes(points, group_plane_equation_params) # (N, n)

    if not rm:
        return masks
        
    # 4. remove point insider the bboxes
    masks = np.any(masks, axis=-1)

    return points[~masks]


# modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/utils.py#L11
def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def nearest_bev(bboxes):
    '''
    bboxes: (n, 7), (x, y, z, w, l, h, theta)
    return: (n, 4), (x1, y1, x2, y2)
    '''    
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(bboxes[:, 6].cpu(), offset=0.5, period=np.pi).to(bboxes_bev)
    bboxes_bev = torch.where(torch.abs(bboxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)
    
    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat([bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2


def iou2d(bboxes1, bboxes2, metric=0):
    '''
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    '''
    bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :]) # (n, m)
    bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :]) # (n, m)
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

    iou_area = bboxes_w * bboxes_h # (n, m)
    
    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1] # (n, )
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1] # (m, )
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-8)
    return iou


def iou2d_nearest(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    '''
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou


def iou3d(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7)
    return: (n, m)
    '''
    # 1. height overlap
    bboxes1_bottom, bboxes2_bottom = bboxes1[:, 2], bboxes2[:, 2] # (n, ), (m, )
    bboxes1_top, bboxes2_top = bboxes1[:, 2] + bboxes1[:, 5], bboxes2[:, 2] + bboxes2[:, 5] # (n, ), (m, )
    bboxes_bottom = torch.maximum(bboxes1_bottom[:, None], bboxes2_bottom[None, :]) # (n, m) 
    bboxes_top = torch.minimum(bboxes1_top[:, None], bboxes2_top[None, :])
    height_overlap =  torch.clamp(bboxes_top - bboxes_bottom, min=0)

    # 2. bev overlap
    bboxes1_x1y1 = bboxes1[:, :2] - bboxes1[:, 3:5] / 2
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 3:5] / 2
    bboxes2_x1y1 = bboxes2[:, :2] - bboxes2[:, 3:5] / 2
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 3:5] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 6:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 6:]], dim=-1)
    bev_overlap = boxes_overlap_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    # 3. overlap and volume
    overlap = height_overlap * bev_overlap
    volume1 = bboxes1[:, 3] * bboxes1[:, 4] * bboxes1[:, 5]
    volume2 = bboxes2[:, 3] * bboxes2[:, 4] * bboxes2[:, 5]
    volume = volume1[:, None] + volume2[None, :] # (n, m)

    # 4. iou
    iou = overlap / (volume - overlap + 1e-8)

    return iou
    

def iou3d_camera(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7)
    return: (n, m)
    '''
    # 1. height overlap
    bboxes1_bottom, bboxes2_bottom = bboxes1[:, 1] - bboxes1[:, 4], bboxes2[:, 1] -  bboxes2[:, 4] # (n, ), (m, )
    bboxes1_top, bboxes2_top = bboxes1[:, 1], bboxes2[:, 1] # (n, ), (m, )
    bboxes_bottom = torch.maximum(bboxes1_bottom[:, None], bboxes2_bottom[None, :]) # (n, m) 
    bboxes_top = torch.minimum(bboxes1_top[:, None], bboxes2_top[None, :])
    height_overlap =  torch.clamp(bboxes_top - bboxes_bottom, min=0)

    # 2. bev overlap
    bboxes1_x1y1 = bboxes1[:, [0, 2]] - bboxes1[:, [3, 5]] / 2
    bboxes1_x2y2 = bboxes1[:, [0, 2]] + bboxes1[:, [3, 5]] / 2
    bboxes2_x1y1 = bboxes2[:, [0, 2]] - bboxes2[:, [3, 5]] / 2
    bboxes2_x2y2 = bboxes2[:, [0, 2]] + bboxes2[:, [3, 5]] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 6:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 6:]], dim=-1)
    bev_overlap = boxes_overlap_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    # 3. overlap and volume
    overlap = height_overlap * bev_overlap
    volume1 = bboxes1[:, 3] * bboxes1[:, 4] * bboxes1[:, 5]
    volume2 = bboxes2[:, 3] * bboxes2[:, 4] * bboxes2[:, 5]
    volume = volume1[:, None] + volume2[None, :] # (n, m)

    # 4. iou
    iou = overlap / (volume - overlap + 1e-8)

    return iou


def iou_bev(bboxes1, bboxes2):
    '''
    bboxes1: (n, 5), (x, z, w, h, theta)
    bboxes2: (m, 5)
    return: (n, m)
    '''
    bboxes1_x1y1 = bboxes1[:, :2] - bboxes1[:, 2:4] / 2
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 2:4] / 2
    bboxes2_x1y1 = bboxes2[:, :2] - bboxes2[:, 2:4] / 2
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 2:4] / 2
    bboxes1_bev = torch.cat([bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 4:]], dim=-1)
    bboxes2_bev = torch.cat([bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 4:]], dim=-1)
    bev_overlap = boxes_iou_bev(bboxes1_bev, bboxes2_bev) # (n, m)

    return bev_overlap


def keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape):
    '''
    result: dict(lidar_bboxes, labels, scores)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    P2: shape=(4, 4)
    image_shape: (h, w)
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    h, w = image_shape

    lidar_bboxes = result['lidar_bboxes']
    labels = result['labels']
    scores = result['scores']
    camera_bboxes = bbox_lidar2camera(lidar_bboxes, tr_velo_to_cam, r0_rect) # (n, 7)
    bboxes_points = bbox3d2corners_camera(camera_bboxes) # (n, 8, 3)
    image_points = points_camera2image(bboxes_points, P2) # (n, 8, 2)
    image_x1y1 = np.min(image_points, axis=1) # (n, 2)
    image_x1y1 = np.maximum(image_x1y1, 0)
    image_x2y2 = np.max(image_points, axis=1) # (n, 2)
    image_x2y2 = np.minimum(image_x2y2, [w, h])
    bboxes2d = np.concatenate([image_x1y1, image_x2y2], axis=-1)

    keep_flag = (image_x1y1[:, 0] < w) & (image_x1y1[:, 1] < h) & (image_x2y2[:, 0] > 0) & (image_x2y2[:, 1] > 0)
    
    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    return result


def keep_bbox_from_lidar_range(result, pcd_limit_range):
    '''
    result: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    pcd_limit_range: []
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    lidar_bboxes, labels, scores = result['lidar_bboxes'], result['labels'], result['scores']
    if 'bboxes2d' not in result:
        result['bboxes2d'] = np.zeros_like(lidar_bboxes[:, :4])
    if 'camera_bboxes' not in result:
        result['camera_bboxes'] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result['bboxes2d'], result['camera_bboxes']
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :] # (n, 3)
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :] # (n, 3)
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)
    
    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    return result


def points_in_bboxes_v2(points, r0_rect, tr_velo_to_cam, dimensions, location, rotation_y, name):
    """ Check which LiDAR points fall inside each 3D bounding box.
    Args:
        points [np.ndarray float32, (N, 4)]: LiDAR points in homogeneous coordinates (x, y, z, reflectance).
        r0_rect [np.ndarray float32, (4, 4)]: rectification matrix aligning all cameras into a common reference frame.
        tr_velo_to_cam [np.ndarray float32, (4, 4)]: transformation matrix from LiDAR to camera coordinates.
        dimensions [np.ndarray float32, (m, 3)]: bounding box size (length, height, width) for m boxes.
        location [np.ndarray float32, (m, 3)]: box center locations in camera coordinates.
        rotation_y [np.ndarray float32, (m,)]: yaw angles (rotation around Y-axis) of bounding boxes.
        name [np.ndarray object/string, (m,)]: class names or labels of bounding boxes.
        
    Returns:
        indices [np.ndarray bool, (N, m_valid)]: mask where indices[i, j] is True if point i is inside bounding box j.
        m_total [int]: total number of bounding boxes including 'DontCare'.
        m_valid [int]: number of valid bounding boxes excluding 'DontCare'.
        bboxes_lidar [np.ndarray float32, (m_valid, 7)]: valid bounding boxes in LiDAR coordinates, each as (x, y, z, dx, dy, dz, yaw).
        name [np.ndarray object/string, (m_valid,)]: class names of valid bounding boxes.
    """
    n_total_bbox = len(dimensions)
    n_valid_bbox = len([item for item in name if item != 'DontCare'])
    location, dimensions = location[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotation_y, name = rotation_y[:n_valid_bbox], name[:n_valid_bbox]
    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1)
    bboxes_lidar = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
    bboxes_corners = bbox3d2corners(bboxes_lidar)
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, n), N is points num, n is bboxes number
    return indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name


def get_points_num_in_bbox(points, r0_rect, tr_velo_to_cam, dimensions, location, rotation_y, name):
    """ Count how many LiDAR points fall inside each 3D bounding box.
    Args:
        points [np.ndarray float32, (N, 4)]: LiDAR points in homogeneous coordinates (x, y, z, reflectance).
        r0_rect [np.ndarray float32, (4, 4)]: rectification matrix aligning all cameras into a common reference frame.
        tr_velo_to_cam [np.ndarray float32, (4, 4)]: transformation matrix from LiDAR to camera coordinates.
        dimensions [np.ndarray float32, (m, 3)]: bounding box size (length, height, width) for m boxes.
        location [np.ndarray float32, (m, 3)]: box center locations in camera coordinates.
        rotation_y [np.ndarray float32, (m, )]: yaw angles (rotation around Y-axis) of bounding boxes.
        name [np.ndarray object/string, (m, )]: class names or labels of bounding boxes.
        
    Returns:
        points_num [np.ndarray int32, (m, )]: number of LiDAR points inside each bounding box. For invalid bboxes (e.g., 'DontCare'), returns -1.
    """
    indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
        points_in_bboxes_v2(
            points=points, 
            r0_rect=r0_rect, 
            tr_velo_to_cam=tr_velo_to_cam, 
            dimensions=dimensions, 
            location=location, 
            rotation_y=rotation_y, 
            name=name)
    points_num = np.sum(indices, axis=0)
    non_valid_points_num = [-1] * (n_total_bbox - n_valid_bbox)
    points_num = np.concatenate([points_num, non_valid_points_num], axis=0)
    return np.array(points_num, dtype=np.int32)


# Modified from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L609
def remove_outside_points(points, r0_rect, tr_velo_to_cam, P2, image_shape):
    """ Remove points which are outside of image.
    Args:
        points [np.ndarray float32, (N, 4)]: total LiDAR points
        r0_rect [np.ndarray float32, (4, 4)]: rectification matrix of camera, allign all cameras into a common coordinate
        tr_velo_to_cam [np.ndarray float32, (4, 4)]: transformation matrix from LiDAR to camera
        P2 [np.ndarray float32, (4, 4)]: projection matrix, convert 3D camera coordinate to 2D pixel coordinate
        
    Returns:
        points [np.ndarray float32, (n, 4)]: reduced LiDAR points (in image)
    """
    
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = points_camera2lidar(frustum.T[None, ...], tr_velo_to_cam, r0_rect) # (1, 8, 3)
    group_rectangle_vertexs_v = group_rectangle_vertexs(frustum)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, 1)
    points = points[indices.reshape([-1])]
    return points

def remove_outside_bboxes(annotation_dict, r0_rect, tr_velo_to_cam, P2, image_shape):
    """
    Remove 3D bounding boxes that have more than 4 corners projected outside the image frustum.
    
    Args:
        dimensions [np.ndarray float32, (m, 3)]
        location [np.ndarray float32, (m, 3)]
        rotation_y [np.ndarray float32, (m, )]
        name [np.ndarray object/string, (m, )]
        r0_rect, tr_velo_to_cam, P2 [np.ndarray float32, (4, 4)]
        image_shape [tuple int, (h, w)]
        
    Returns:
        filtered_dimensions, filtered_location, filtered_rotation_y, filtered_name
    """
    names = annotation_dict['name']
    locs = annotation_dict['location']
    dims = annotation_dict['dimension']
    rots = annotation_dict['rotation_y']
    
    bboxes_cam = np.concatenate([locs, dims, rots[:, None]], axis=1)
    corners_cam = bbox3d2corners_camera(bboxes_cam)  # (m,8,3)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    img_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(img_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = points_camera2lidar(frustum.T[None, ...], tr_velo_to_cam, r0_rect)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs(frustum))
    keep = []
    for i in range(len(corners_cam)):
        corners_lidar = points_camera2lidar(corners_cam[i][None, ...], tr_velo_to_cam, r0_rect)[0]
        mask = points_in_bboxes(corners_lidar, frustum_surfaces)[..., 0]  # (8,)
        out_count = np.sum(~mask)
        keep.append(out_count <= 4)
    keep = np.array(keep, dtype=bool)
    for k, v in annotation_dict.items():
        if isinstance(v, np.ndarray) and len(v) == len(keep):
            annotation_dict[k] = v[keep]
    return annotation_dict


# Copied from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L609
def projection_matrix_to_CRT_kitti(proj):
    """ Split projection matrix of kitti.
    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.
    Args:
        proj [np.ndarray float32, (4, 4)]: projection matrix, convert 3D camera coordinate to 2D pixel coordinate
        
    Returns:
        C [np.ndarray float32, (3, 3)]: intrinsic matrix
        R [np.ndarray float32, (3, 3)]: rotation matrix
        T [np.ndarray float32, (3, 3)]: translation matrix
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    
    return C, R, T


# Copied from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L661
def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """ Get frustum corners in camera coordinates.
    Args:
        bbox_image [list int, [4]]: bounding box of image in pixel coordinate, include x_min, y_min, x_max, y_max
        C [np.ndarray float32, (3, 3)]: intrinsic matrix
        
    Returns:
        ret_xyz [np.ndarray float32, (8, 3)]: 8 corners of frustum in camera coordinate
    """
    
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array([[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array([fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array([fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    
    return ret_xyz

def project_point_to_camera(point, calib, eps=1e-6):
    """ Project mean centers of pillars to camera image coordinates.
    Args: 
        point (torch.Tensor): shape (P1 + P2 + ... + Pb, 3) where P1 + P2 + ... + Pb is the total number of pillars in the batch.
        calib (dict): calibration dictionary containing 'Tr_velo_to_cam', 'R0_rect', and 'P2' matrices. All matrix is 4x4.
        
    Returns:
        u (torch.Tensor): horizontal pixel coordinates in the camera image. Shape (P1 + P2 + ... + Pb,).
        v (torch.Tensor): vertical pixel coordinates in the camera image. Shape (P1 + P2 + ... + Pb,).     
    """              
    device = point.device
    combined_np = calib['P2'] @ calib['R0_rect'] @ calib['Tr_velo_to_cam']
    combined_transform = torch.from_numpy(combined_np).to(device=device, dtype=torch.float32)
    pts_velo_hom = torch.nn.functional.pad(point, (0, 1), value=1.0)  # (N,4)
    pts_cam = pts_velo_hom @ combined_transform.T  # (N,4)
    z = pts_cam[:, 2].clamp(min=eps)
    u = pts_cam[:, 0] / z
    v = pts_cam[:, 1] / z

    return u, v