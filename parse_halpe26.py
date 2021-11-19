import numpy as np
import json
import matplotlib.pyplot as plt
import os

# Joints such as ear, eyes are not necassary
wanted_joints = list(range(5,18)) + [19]

def normalize_halpe26(poses, img):
    hip_idx = 13 #19 before removal of unneeded joints
    for i, det in enumerate(poses):
        nrows = det['box'][3] 
        ncols = det['box'][2]
        
        # Define the centroid of pelvis (hip) as center of image
        center_x = det['keypoints'][hip_idx, 0]
        center_y = det['keypoints'][hip_idx, 1]
        
        keypts = poses[i]['keypoints'].copy()[:,:-1]
        keypts_norm = keypts.copy()
        keypts_norm[:, 0] = (center_x - keypts[:, 0]) / ncols
        keypts_norm[:, 1] = (center_y - keypts[:, 1]) / nrows
        
        poses[i]['keypoints'][:, :-1] = keypts_norm
    return poses


if __name__ == "__main__":
    print(wanted_joints, len(wanted_joints))

