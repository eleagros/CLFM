#!/usr/bin/env python3
#
# Created by Romy Gros, 2025
# This file is new and was added to support SIFT keypoint extraction and saving.
#
# Licensed under the MIT License (see LICENSE_MIT for details).

import os
import pickle
import sys
from learnedmatcher import LearnedMatcher
from extract_sift import ExtractSIFT
import pathlib

model_path = os.path.join(pathlib.Path(__file__).parent.resolve().__str__().replace('demo', ''), 'models', 'gl3d/sift-4000/model_best.pth')
img1_name = sys.argv[1]
img2_name = sys.argv[2]

detector = ExtractSIFT(8000)

kpt1, desc1 = detector.run(img1_name)
kpt2, desc2 = detector.run(img2_name)

with open(os.path.join(sys.argv[3], 'sift_points.pickle'), 'wb') as handle:
    pickle.dump([kpt1, desc1, kpt2, desc2], handle, protocol=pickle.HIGHEST_PROTOCOL)