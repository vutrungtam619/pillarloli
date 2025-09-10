import os
root = os.path.dirname(os.path.dirname(__file__))

config = {
    'classes': {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2},
    'num_classes': 3,
    'data_root': 'kitti',
    'pc_range': [0, -39.68, -3, 69.12, 39.68, 1],
    'voxel_size': [0.16, 0.16, 4],
    'max_voxels': (10000, 15000),
    'max_points': 32,
    'new_shape': (384, 1280),
    'mean': [0.36783523, 0.38706144, 0.3754649],
    'std': [0.31566228, 0.31997792, 0.32575161],
    'top_k': 5,
    'anchor_ranges': [[0, -39.68, -0.6, 69.12, 39.68, -0.6], [0, -39.68, -0.6, 69.12, 39.68, -0.6], [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
    'anchor_sizes': [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
    'anchor_rotations': [0, 1.57],
    'checkpoint_dir': os.path.join(root, 'checkpoints'),
    'log_dir': os.path.join(root, 'logs'),
    'batch_size_train': 8,
    'batch_size_val': 2,
    'num_workers': 4,
    'init_lr': 0.00033,
    'epoch': 40,
    'ckpt_freq': 2,
    'log_freq': 25,
}