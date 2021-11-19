import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import os
# import polygon as poly


def load_keypoints(json_path):
    with open(json_path) as f:
        data = json.load(f)
        # Loop over person detections and convert keypts to numpy arr
        for i, det in enumerate(data):
            data[i]['keypoints'] = np.array(det['keypoints']).reshape((-1, 3))
            data[i]['box'] = np.array(det['box'])
        return data

def normalize_detections(poses, img, format='MPII', lhip_idx=11, rhip_idx=8):
    if format == 'COCO':
        rhip_idx = 12
    
    for i, det in enumerate(poses):
        nrows = det['box'][3] 
        ncols = det['box'][2]
        
        # Define the centroid of hips as the center of the image
        # center_x = det['box'][0] + ncols / 2
        # center_y = det['box'][1] + nrows / 2
        center_x = (det['keypoints'][lhip_idx, 0] + det['keypoints'][rhip_idx, 0]) / 2
        center_y = (det['keypoints'][lhip_idx, 1] + det['keypoints'][rhip_idx, 1]) / 2
        
        keypts = poses[i]['keypoints'].copy()[:,:-1]
        keypts_norm = keypts.copy()
        keypts_norm[:, 0] = (center_x - keypts[:, 0]) / ncols
        keypts_norm[:, 1] = (center_y - keypts[:, 1]) / nrows
        
        poses[i]['keypoints'][:, :-1] = keypts_norm
    
    return poses

def plot_poses(poses, img, marker_set, save_folder, normalized = True):
    for i, det in enumerate(poses):
        x1 = int(det['box'][0])
        x2 = int(det['box'][2] + x1)
        y1 = int(det['box'][1])
        y2 = int(det['box'][3] + y1)
        crop = img[y1:y2, x1:x2]
        if normalized:
            fig, ax = plt.subplots()
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor((0, 0, 0))
            # plt.title(f'Player {i} normalized pose')
            for j, pt in enumerate(det['keypoints']):
                plt.scatter(pt[0], pt[1], s=15, marker=marker_set[j])
                plt.xlim((-1,1))
                plt.ylim((-1,1))
        else:
            plt.title(f'Player {i} pose')
            plt.imshow(crop)
            plt.scatter(det['keypoints'][:, 0], det['keypoints'][:, 1], c='r', s=40)
        plt.show()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # cv2.imshow('plot', data)
        save_path = os.path.join(save_folder, f'player-{i}-norm-pose.png')
        cv2.imwrite(save_path, data)


def point_to_vertex(pt, r, theta):
    """(x, y, c) -> n vertices that can be drawn with cv2.polylines"""
    # 24 is for 360 / 15 (number of joints)
    # This gives a unique polygon for each joint
    x = round(pt[0] + r * np.sin(np.deg2rad(theta)))
    y = round(pt[1] + r * np.cos(np.deg2rad(theta)))
    return np.array([x, y])

    
def polygon_maker(poses, img, r):
    for i, det in enumerate(poses):
        x1 = int(det['box'][0])
        x2 = int(det['box'][2] + x1)
        y1 = int(det['box'][1])
        y2 = int(det['box'][3] + y1)
        crop = img[y1:y2, x1:x2]

        joint_img = np.zeros_like(img)


        joints = []

        # Loop through each set of keypoints
        for i, pt in enumerate(det['keypoints']):
            vertices = []
            # Each keypoint must have a different pattern, so the number of vertices == index + 1
            thetas = np.linspace(0, 360, num=i+3)
            for j in range(i+3):
                vertices.append(point_to_vertex(pt, r, thetas[j]))
            # Convert vertices to np and draw them on image
            vert = np.array(vertices)
            joint_img = cv2.polylines(joint_img, [vert], True, (255, 0, 0), thickness=1)
            cv2.imshow('shape_img', joint_img)
            cv2.waitKey(0)
        return

def cv_marker_pts(poses, img):
    joint_img = np.zeros_like(img)

    # Loop through each set of keypoints
    for i, det in enumerate(poses):
        pts = (det['keypoints'][:, :-1]).astype(np.int32)
        ln_type = cv2.LINE_AA
        for j, pt in enumerate(pts):
            if j == 7:
                ln_type = cv2.FILLED
            mark_type = j % 7
            joint_img = cv2.drawMarker(joint_img, tuple(pt), (255, 0 ,0), markerType=mark_type, markerSize=5, thickness=1,  line_type=cv2.LINE_AA)
            cv2.imshow('frame', joint_img)
            cv2.waitKey(0)
        pass

def plt_marker_pts(poses, img, marker_set):
    joint_img = np.zeros((1080, 1920))

    # Loop through each person
    for i, det in enumerate(poses):
        # Loop through each point, giving each one a new marker
        for j, pt in enumerate(det['keypoints']):
            plt.imshow(img, origin='upper')
            plt.scatter(pt[0], pt[1], marker=marker_set[j], s=13)
        plt.show()
        plt.clf()

if __name__ == '__main__':
    img_path = 'out-002.jpg'
    json_path = 'alphapose-results.json'
    example_img_saveloc = 'saved_example_poses'
    
    marker_set_MPII = ['.', 'v', '<', '1', '8', 's', 'p', '*', 'x', 'D', \
                    '|', '_', '$C$', '$R$', '$P$'] #15 different plot markers

    marker_set_COCO = ['.', 'v', '<', '1', '8', 's', 'p', '*', 'x', 'D', \
                    '|', '_', '$C$', '$R$', '$P$', '$U$', '$W$', '$Z$'] #18 different plot markers

    img = cv2.imread(img_path)
    poses = load_keypoints(json_path)
    # plt_marker_pts(poses, img, marker_set_COCO)
    # polygon_maker(poses, img, 15)
    # cv_marker_pts(poses, img)
    
    # plot_poses(poses, img, normalized=False)
    
    norm_poses = normalize_detections(poses, img)
    plot_poses(norm_poses, img, marker_set_COCO, example_img_saveloc)
    # polygon_maker(norm_poses, img, 5)
    # plt_marker_pts(poses, img, marker_set_COCO)
    