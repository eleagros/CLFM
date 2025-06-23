# Created by Romy Gros, 2025
# This file is new and was added to support SuperPoint keypoint extraction and saving.
#
# Licensed under the MIT License (see LICENSE_MIT for details).
import numpy as np
import cv2
import torch
import sys
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

detection_thresh = 0.005
nms_radius = 5

# url_def = r'https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/superpoint.py'
# url_ckpt_ml = r'https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth'

reference_img = sys.argv[1]
moving_img = sys.argv[2]
output_folder = sys.argv[3]
path_ckpt_ml = os.path.join(os.path.abspath(sys.argv[4]), 'weights', 'superpoint_v1_ml.pth')

def get_superpoint_features(name_img, reference_image = None):
    
    image = cv2.imread(name_img).mean(-1) / 255
    if reference_image is None:
        pass
    else:
        ratio = reference_image.shape[0] / image.shape[0]
        image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
        
    # torch.hub.download_url_to_file(url_def, 'superpoint_ml.py')
    # torch.hub.download_url_to_file(url_ckpt_ml, path_ckpt_ml)

    from superpoint_ml import SuperPoint as SuperPointML
    detection_thresh_ml = detection_thresh / 10  # heuristic
    sp_ml = SuperPointML(dict(keypoint_threshold=detection_thresh_ml, nms_radius=nms_radius)).eval()
    ckpt_ml = torch.load(path_ckpt_ml, map_location='cpu', weights_only=False)
    sp_ml.load_state_dict(ckpt_ml)

    pred_ml = sp_ml({'image': torch.from_numpy(image[None,None]).float()})
    points_ml = pred_ml['keypoints'][0]
    descriptors = pred_ml['descriptors'][0]
    scores = pred_ml['scores'][0]
    
    return image, points_ml.detach().numpy(), descriptors.detach().numpy().T, scores.detach().numpy()

reference_image, superpoint_reference, superpoint_desc_ref, scores_ref = get_superpoint_features(reference_img)
_, superpoint_moving, superpoint_desc_mov, scores_mov = get_superpoint_features(moving_img, reference_image = reference_image)

with open(os.path.join(output_folder, 'superpoint_points.pickle'), 'wb') as handle:
    pickle.dump([superpoint_reference, superpoint_desc_ref, scores_ref, superpoint_moving, superpoint_desc_mov, scores_mov], handle, protocol=pickle.HIGHEST_PROTOCOL)