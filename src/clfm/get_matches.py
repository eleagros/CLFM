import os
from pathlib import Path
import shutil
import numpy as np
import torch
import pickle
import sys
import cv2
import random

from clfm import utils

_RESIZE_IMGS = (832, 832)

def mnn(
    ref: np.ndarray,
    mov: np.ndarray,
    desc_ref: np.ndarray,
    desc_mov: np.ndarray,
    score_ref: np.ndarray = None,
    score_mov: np.ndarray = None,
    parameters: dict = None
):
    """
    Wrapper for mutual nearest neighbor matcher.

    Args:
        ref, mov, desc_ref, desc_mov: Keypoints and descriptors for both images.
        score_ref, score_mov: Not used.
        parameters: Not used.

    Returns:
        tuple: (matched_points_ref, matched_points_mov)
    """
    # Convert inputs to torch tensors and call mnn_matcher
    return mnn_matcher(
        torch.tensor(ref),
        torch.tensor(mov),
        torch.tensor(desc_ref),
        torch.tensor(desc_mov)
    )

def mnn_matcher(
    points_0: torch.Tensor,
    points_1: torch.Tensor,
    desc_0: torch.Tensor,
    desc_1: torch.Tensor
):
    """
    Find mutual nearest neighbor matches and estimate homography.

    Args:
        points_0 (np.ndarray or torch.Tensor): Keypoints from image 0.
        points_1 (np.ndarray or torch.Tensor): Keypoints from image 1.
        desc_0 (np.ndarray or torch.Tensor): Descriptors from image 0.
        desc_1 (np.ndarray or torch.Tensor): Descriptors from image 1.

    Returns:
        tuple: (matched_points_0, matched_points_1)
    """
    # Ensure torch tensors
    if not torch.is_tensor(points_0):
        points_0 = torch.from_numpy(points_0).float()
    if not torch.is_tensor(points_1):
        points_1 = torch.from_numpy(points_1).float()
    if not torch.is_tensor(desc_0):
        desc_0 = torch.from_numpy(desc_0).float()
    if not torch.is_tensor(desc_1):
        desc_1 = torch.from_numpy(desc_1).float()

    # Find the mutual nearest neighbor matches
    matches = mutual_nearest_match(desc_0, desc_1)
    matched_points_0 = points_0[matches[:, 0]].cpu().numpy()
    matched_points_1 = points_1[matches[:, 1]].cpu().numpy()

    return matched_points_0, matched_points_1

def mutual_nearest_match(
    desc_0: torch.Tensor,
    desc_1: torch.Tensor
):
    """
    Find mutual nearest neighbor matches between two sets of descriptors.

    Args:
        desc_0 (torch.Tensor): Descriptors from image 0.
        desc_1 (torch.Tensor): Descriptors from image 1.

    Returns:
        torch.Tensor: Array of index pairs (N_matches x 2).
    """
    # Compute L2 distance matrix between descriptors
    dists = torch.cdist(desc_0, desc_1, p=2)
    idx1 = torch.argmin(dists, dim=1)
    idx0 = torch.argmin(dists, dim=0)
    
    # Find mutual matches
    matches = []
    for i, j in enumerate(idx1):
        if idx0[j] == i:
            matches.append((i, j.item()))
    return torch.tensor(matches, dtype=torch.long)

def oanet_matcher(
    ref: np.ndarray,
    mov: np.ndarray,
    desc_ref: np.ndarray,
    desc_mov: np.ndarray,
    score_ref: np.ndarray = None,
    score_mov: np.ndarray = None,
    parameters: dict = None
):
    """
    Run OANet matcher via external script.

    Args:
        ref, mov, desc_ref, desc_mov: Keypoints and descriptors for both images.
        score_ref, score_mov: Not used.
        parameters: Not used.

    Returns:
        tuple: (matched_points_ref, matched_points_mov)
    """
    # Create temporary directory and save input data
    cwd = os.getcwd()
    os.makedirs(f'{cwd}/tmp', exist_ok=True)
    with open(f'{cwd}/tmp/vals.pickle', 'wb') as handle:
        pickle.dump([ref, mov, desc_ref, desc_mov], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Create the command and run the OANet script
    path_script_rep = utils.get_OANet_path()
    path_script = 'infer_oa_net.py'
    os.system(f'cd {path_script_rep} && python {path_script} {cwd}/tmp/vals.pickle')
    
    # Collect the results
    with open(f"{cwd}/tmp/vals.pickle", 'rb') as handle:
        matches = pickle.load(handle)
        
    shutil.rmtree(f'{cwd}/tmp', ignore_errors=True)
    
    return matches[0][:,:2], matches[1][:,:2]

def superglue_matcher(
    ref: np.ndarray,
    mov: np.ndarray,
    desc_ref: np.ndarray,
    desc_mov: np.ndarray,
    score_ref: np.ndarray = None,
    score_mov: np.ndarray = None,
    parameters: dict = None
):
    """
    Run SuperGlue matcher via external script.

    Args:
        ref, mov, desc_ref, desc_mov, score_ref, score_mov: Keypoints, descriptors, and scores for both images.
        parameters (dict): Contains paths to images.

    Returns:
        tuple: (matched_points_ref, matched_points_mov)
    """
    # Create temporary directory and save input data
    cwd = os.getcwd()
    os.makedirs(f'{cwd}/tmp', exist_ok=True)
    save_superglue_input(
        ref,
        mov,
        desc_ref,
        desc_mov,
        score_ref,
        score_mov,
        cwd=cwd
    )    
    folder_output_superglue = os.path.abspath(f'{cwd}/tmp')
    shutil.copy(parameters['path_image_fixed'], os.path.join(folder_output_superglue, 'image_fixed.png'))
    shutil.copy(parameters['path_image_moving'], os.path.join(folder_output_superglue, 'image_moving.png'))
    
    # Create the command and run the SuperGlue script
    path_superglue_script = utils.get_SuperGlue_path()
    path_superglue_script = os.path.join(path_superglue_script, 'demo_superglue.py')
    cmd = f"python {path_superglue_script} --input {folder_output_superglue} --output_dir {folder_output_superglue} --no_display --resize -1"
    os.system(cmd)
    
    # Load the results
    with open(f'{folder_output_superglue}/matches_000000_000001.pickle', 'rb') as handle:
        [matched_points_ref, matched_points_mov] = pickle.load(handle)

    shutil.rmtree(f'{cwd}/tmp', ignore_errors=True)
    
    return matched_points_ref, matched_points_mov

def save_superglue_input(
    ref: np.ndarray,
    mov: np.ndarray,
    desc_ref: np.ndarray,
    desc_mov: np.ndarray,
    score_ref: np.ndarray = None,
    score_mov: np.ndarray = None,
    cwd: str = os.getcwd()
):
    """
    Save SuperGlue input dictionaries as pickles for reference and moving images.

    Args:
        ref (np.ndarray): Keypoints for the reference image.
        mov (np.ndarray): Keypoints for the moving image.
        desc_ref (np.ndarray): Descriptors for the reference image.
        desc_mov (np.ndarray): Descriptors for the moving image.
        score_ref (np.ndarray): Keypoint scores for the reference image.
        score_mov (np.ndarray): Keypoint scores for the moving image.
        cwd (str): Current working directory (for tmp folder).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dict_ref = {
        'keypoints0': [torch.tensor(ref).to(device).type(torch.float32)],
        'descriptors0': [torch.tensor(desc_ref.T).to(device).type(torch.float32)],
        'scores0': (torch.tensor(score_ref).to(device).type(torch.float32),)
    }
    with open(os.path.join(cwd, 'tmp', 'dict_ref.pickle'), 'wb') as handle:
        pickle.dump(dict_ref, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dict_mov = {
        'keypoints1': [torch.tensor(mov).to(device).type(torch.float32)],
        'descriptors1': [torch.tensor(desc_mov.T).to(device).type(torch.float32)],
        'scores1': (torch.tensor(score_mov).to(device).type(torch.float32),)
    }
    with open(os.path.join(cwd, 'tmp', 'dict_mov.pickle'), 'wb') as handle:
        pickle.dump(dict_mov, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def match_omniglue(
    parameters: dict
):  
    """
    Run OmniGlue matcher via external script.

    Args:
        parameters (dict): Contains paths to images.

    Returns:
        tuple: (matched_points_ref, matched_points_mov)
    """
    # Get the OmniGlue path and run the demo script
    path_omniglue = utils.get_omniglue_path()

    # Uncomment the line below to run on CPU (if GPU is not available)
    cmd = (
        f'cd {path_omniglue} && '
        # f'CUDA_VISIBLE_DEVICES="" python demo.py '
        f'{parameters["path_image_fixed"]} '
        f'{parameters["path_image_moving"]}'
    )

    print(f"Running command: {cmd}")
    os.system(cmd)
    
    # Load the matched points from the pickle file
    with open(f'{path_omniglue}/points_matched.pickle', 'rb') as handle:
        [matched_points_ref, matched_points_mov] = pickle.load(handle)
    
    return matched_points_ref, matched_points_mov


def match_matchanything(
    parameters: dict,
    type_mode: str = 'eloftr',
):  
    """
    Run Match Anything matcher via external script.

    Args:
        parameters (dict): Contains paths to images.

    Returns:
        tuple: (matched_points_ref, matched_points_mov)
    """
    img0 = cv2.imread(parameters['path_image_fixed'])
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.imread(parameters['path_image_moving'])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    if str(utils.get_matchanything_path()) not in sys.path:
        sys.path.append(str(utils.get_matchanything_path()))
    if str(Path(utils.get_SuperGlue_path()).parent) not in sys.path:
        sys.path.append(str(Path(utils.get_SuperGlue_path()).parent))
    from imcui.ui.utils import run_matching

    random.seed(42)
    ransac_reproj_threshold = 12

    results = run_matching(
        img0, img1,
        match_threshold=0.15,
        extract_max_keypoints=5000,
        keypoint_threshold=0.015,
        key=f"matchanything_{type_mode}",
        ransac_method="CV2_RANSAC",
        ransac_reproj_threshold=ransac_reproj_threshold,
        ransac_confidence=0.999,
        ransac_max_iter=10000,
        choice_geometry_type="Homography",
        matcher_zoo=get_matcher_zoo(),
        force_resize=False,
        image_width=img0.shape[1],
        image_height=img0.shape[0],
        use_cached_model=False,
    )

    points_reference, points_moving = results[3][0], results[3][1]

    mkpts0 = points_reference
    # mkpts0[:, 0] = mkpts0[:, 0] * (img0.shape[1] / _RESIZE_IMGS[0])
    # mkpts0[:, 1] = mkpts0[:, 1] * (img0.shape[0] / _RESIZE_IMGS[1])
    mkpts1 = points_moving
    # mkpts1[:, 0] = mkpts1[:, 0] * (img1.shape[1] / _RESIZE_IMGS[0])
    # mkpts1[:, 1] = mkpts1[:, 1] * (img1.shape[0] / _RESIZE_IMGS[1])
    return mkpts0, mkpts1

def get_matcher_zoo() -> dict:
    """
    Returns a dictionary containing configurations for different MatchAnything models.
    """
    if str(utils.get_matchanything_path()) not in sys.path:
        sys.path.append(str(utils.get_matchanything_path()))
    from imcui.ui.utils import parse_match_config

    base_conf = {
        "matchanything_eloftr": {
            "dense": True,
            "matcher": "matchanything_eloftr",
        },
        "matchanything_roma": {
            "dense": True,
            "matcher": "matchanything_roma"
        }
    }

    zoo = {}
    for key, conf in base_conf.items():
        zoo[key] = parse_match_config(conf)
    return zoo