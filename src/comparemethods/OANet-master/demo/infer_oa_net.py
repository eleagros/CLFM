import os
import sys
import pickle
import cv2
import numpy as np
from learnedmatcher import LearnedMatcher
from extract_sift import ExtractSIFT

current_path = r'C:\Users\romai\Documents\repositories\compareMatchingMethods\src\comparemethods\OANet-master'
model_path = os.path.join(current_path, 'models', 'gl3d/sift-4000/model_best.pth')

print(sys.argv)
with open(os.path.join(sys.argv[1], sys.argv[2] + '_points.pickle'), 'rb') as handle:
    if sys.argv[2] == 'superpoint':
        [kpt1, desc1, _, kpt2, desc2, _] = pickle.load(handle)
    else:
        [kpt1, desc1, kpt2, desc2] = pickle.load(handle)
    
print(kpt1.shape, desc1.shape, kpt2.shape, desc2.shape)

lm = LearnedMatcher(model_path, inlier_threshold=1, use_ratio=0, use_mutual=0)
_, corr1, corr2 = lm.infer([kpt1, kpt2], [desc1, desc2])

print(os.path.join(sys.argv[1], 'matches_' + sys.argv[2] + '_oanet.pickle'))
with open(os.path.join(sys.argv[1], 'matches_' + sys.argv[2] + '_oanet.pickle'), 'wb') as handle:
    pickle.dump([corr1, corr2], handle, protocol=pickle.HIGHEST_PROTOCOL)