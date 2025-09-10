import cv2
import numpy as np
import open3d as o3d
import os
from configs.config import config

COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[255, 0, 0],   # Red
              [0, 255, 0],   # Green
              [0, 0, 255],   # Blue
              [255, 255, 0]] # Yellow

LINES = [
        [0, 1],
        [1, 2], 
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 6],
        [7, 3],
        [1, 5],
        [4, 0]
    ]


def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[item, item, item] for item in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply


def ply2npy(ply):
    return np.array(ply.points)


def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def vis_core(plys):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    render_opt = vis.get_render_option()
    render_opt.line_width = 8.0

    for ply in plys:
        vis.add_geometry(ply)

    vis.run()
    vis.destroy_window()

def vis_pc(pc, gt_bboxes=None, gt_labels=None, pred_bboxes=None, pred_labels=None):
    ''' 
    pc: point cloud, can be Open3D PointCloud or np.ndarray (N, 4)
    gt_bboxes: np.ndarray, (n_gt, 8, 3)   # corners only
    gt_labels: (n_gt, )
    pred_bboxes: np.ndarray, (n_pred, 8, 3)   # corners only
    pred_labels: (n_pred, )
    '''

    # Convert numpy point cloud to Open3D PointCloud
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)
    
    # Add coordinate axis
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis_objs = [pc, mesh_frame]

    # ---- Ground Truth boxes ----
    if gt_bboxes is not None:
        for i in range(len(gt_bboxes)):
            bbox = gt_bboxes[i]
            color = COLORS[-1]  # Yellow for GT
            vis_objs.append(bbox_obj(bbox, color=color))

    # ---- Predicted boxes ----
    if pred_bboxes is not None:
        for i in range(len(pred_bboxes)):
            bbox = pred_bboxes[i]
            label = pred_labels[i] if pred_labels is not None else -1
            if label >= 0 and label < 3:
                color = COLORS[label]
            else:
                color = COLORS[-1]
            vis_objs.append(bbox_obj(bbox, color=color))

    # ---- Visualize everything ----
    vis_core(vis_objs)

def vis_img_3d(img, gt_image_points=None, gt_labels=None, pred_image_points=None, pred_labels=None, pred_scores=None, rt=True):
    class_map = {v:k for k,v in config['classes'].items()}  # id -> name

    # ---- Ground Truth ----
    if gt_image_points is not None and gt_labels is not None:
        for i in range(len(gt_image_points)):
            bbox_points = gt_image_points[i]  # (8,2)
            color = COLORS_IMG[-1]  # Yellow for GT
            for line_id in LINES:
                x1, y1 = bbox_points[line_id[0]]
                x2, y2 = bbox_points[line_id[1]]
                cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
         
            cls_id = int(gt_labels[i])
            cls_name = class_map.get(cls_id, str(cls_id))
            x_min, y_min = np.min(bbox_points[:,0]), np.min(bbox_points[:,1])  # top-left corner
            cv2.putText(img, cls_name, (int(x_min), int(y_min)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ---- Prediction ----
    if pred_image_points is not None and pred_labels is not None:
        for i in range(len(pred_image_points)):
            label = pred_labels[i]
            bbox_points = pred_image_points[i]  # (8,2)
            if label >= 0 and label < 3:
                color = COLORS_IMG[label]
            else:
                color = COLORS_IMG[-1]
            for line_id in LINES:
                x1, y1 = bbox_points[line_id[0]]
                x2, y2 = bbox_points[line_id[1]]
                cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 1)
          
            x_min, y_min = np.min(bbox_points[:,0]), np.min(bbox_points[:,1])
            cls_name = class_map.get(label, str(label))
            text = cls_name if pred_scores is None else f"{cls_name} {pred_scores[i]:.2f}"
            cv2.putText(img, text, (int(x_min), int(y_min)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img if rt else (cv2.imshow('bbox', img), cv2.waitKey(0))


