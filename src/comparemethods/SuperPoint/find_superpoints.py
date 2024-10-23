from pprint import pprint
from pathlib import Path
import numpy as np
import cv2
import yaml
import torch
import matplotlib.pyplot as plt
import sys
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

detection_thresh = 0.005
nms_radius = 5

url_def = r'https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/superpoint.py'
url_ckpt_ml = r'https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth'
path_ckpt_ml = os.path.join('\\'.join(os.path.abspath(__file__).split('\\')[:-1]),'weights', 'superpoint_v1_ml.pth')

path_img = sys.argv[1]
reference_img = sys.argv[2]
moving_img = sys.argv[3]
output_folder = sys.argv[4]

def get_superpoint_features(path_img, name_img, reference_image = None):
    os.path.join(path_img, name_img)
    
    image = cv2.imread(os.path.join(path_img, name_img)).mean(-1) / 255
    if reference_image is None:
        pass
    else:
        ratio = reference_image.shape[0] / image.shape[0]
        image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
        
    torch.hub.download_url_to_file(url_def, 'superpoint_ml.py')
    # torch.hub.download_url_to_file(url_ckpt_ml, path_ckpt_ml)

    from superpoint_ml import SuperPoint as SuperPointML
    detection_thresh_ml = detection_thresh / 10  # heuristic
    sp_ml = SuperPointML(dict(keypoint_threshold=detection_thresh_ml, nms_radius=nms_radius)).eval()
    ckpt_ml = torch.load(path_ckpt_ml, map_location='cpu')
    sp_ml.load_state_dict(ckpt_ml)

    pred_ml = sp_ml({'image': torch.from_numpy(image[None,None]).float()})
    points_ml = pred_ml['keypoints'][0]
    descriptors = pred_ml['descriptors'][0]
    scores = pred_ml['scores'][0]
    
    plt.imshow(image, cmap = 'gray')
    plt.scatter(*points_ml.T, lw=1, s=6, c='lime');
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    os.makedirs(os.path.join(output_folder, 'superpoint'), exist_ok=True)
    plt.savefig(os.path.join(output_folder, 'superpoint', name_img.replace('.png', '') + '_superpoint.png'), transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    return image, np.array(points_ml), descriptors.detach().numpy().T, scores.detach().numpy()

reference_image, superpoint_reference, superpoint_desc_ref, scores_ref = get_superpoint_features(path_img, reference_img)
_, superpoint_moving, superpoint_desc_mov, scores_mov = get_superpoint_features(path_img, moving_img, reference_image = reference_image)


with open(os.path.join(output_folder, 'superpoint_points.pickle'), 'wb') as handle:
    pickle.dump([superpoint_reference, superpoint_desc_ref, scores_ref, superpoint_moving, superpoint_desc_mov, scores_mov], handle, protocol=pickle.HIGHEST_PROTOCOL)