#!/usr/bin/env python3
import os
import cv2
import pickle
import numpy as np
import sys
from learnedmatcher import LearnedMatcher
from extract_sift import ExtractSIFT
import matplotlib.pyplot as plt

current_path = r'C:\Users\romai\Documents\repositories\compareMatchingMethods\src\comparemethods\OANet-master'
model_path = os.path.join(current_path, 'models', 'gl3d/sift-4000/model_best.pth')

img1_name = sys.argv[1]
img2_name = sys.argv[2]

detector = ExtractSIFT(8000)
lm = LearnedMatcher(model_path, inlier_threshold=1, use_ratio=0, use_mutual=0)

kpt1, desc1 = detector.run(img1_name)
kpt2, desc2 = detector.run(img2_name)

with open(os.path.join(sys.argv[3], 'sift_points.pickle'), 'wb') as handle:
    pickle.dump([kpt1, desc1, kpt2, desc2], handle, protocol=pickle.HIGHEST_PROTOCOL)