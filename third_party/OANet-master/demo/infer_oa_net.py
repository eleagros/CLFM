# Created by Romy Gros, 2025
# This file is new and was added to support OANet inference using a pre-trained model and saving matched points.
#
# Licensed under the MIT License (see LICENSE_MIT for details).

import os
import sys
import pickle
from learnedmatcher import LearnedMatcher
import pathlib

model_path = os.path.join(pathlib.Path(__file__).parent.resolve().__str__().replace('demo', ''), 'models', 'gl3d/sift-4000/model_best.pth')
with open(os.path.join(sys.argv[1]), 'rb') as handle:
        [kpt1, kpt2, desc1, desc2] = pickle.load(handle)
        
lm = LearnedMatcher(model_path, inlier_threshold=1, use_ratio=0, use_mutual=0)
_, corr1, corr2 = lm.infer([kpt1, kpt2], [desc1, desc2])

with open(os.path.join(sys.argv[1]), 'wb') as handle:
    pickle.dump([corr1, corr2], handle, protocol=pickle.HIGHEST_PROTOCOL)