import os
import pickle
import numpy as np
np.random.seed(42)
from clfm import utils

def get_sift_features(
    parameters: dict
):
    """
    Extract SIFT features from the fixed and moving images.

    This function calls an external script to compute SIFT keypoints and descriptors,
    then loads the results from a pickle file.

    Args:
        parameters (dict): Dictionary containing paths for the fixed and moving images,
            and the output folder for storing results. Obtained from CLFM's configuration.

    Returns:
        tuple: (sift_reference, sift_moving, sift_desc_ref, sift_desc_mov, None, None)
            - sift_reference (np.ndarray): Keypoints from the fixed image (Nx2).
            - sift_moving (np.ndarray): Keypoints from the moving image (Mx2).
            - sift_desc_ref (np.ndarray): Descriptors for the fixed image.
            - sift_desc_mov (np.ndarray): Descriptors for the moving image.
            - None: Placeholder for compatibility.
            - None: Placeholder for compatibility.
    """
    path_script_rep = utils.get_OANet_path()
    path_script = os.path.join(path_script_rep, 'get_sift_points.py')
    os.system(f"cd {path_script_rep} && python {path_script}  {parameters['path_image_fixed']} {parameters['path_image_moving']} {parameters['folder_output_tile']}")
    
    with open(os.path.join(parameters['folder_output_tile'], 'sift_points.pickle'), 'rb') as handle:
        [sift_reference, sift_desc_ref, sift_moving, sift_desc_mov] = pickle.load(handle)
    
    return [sift_reference[:,:2], sift_moving[:,:2], sift_desc_ref, sift_desc_mov, None, None]

def get_superpoint_features(
    parameters: dict
):
    """
    Extract SuperPoint features from the fixed and moving images.

    This function calls an external script to compute SuperPoint keypoints and descriptors,
    then loads the results from a pickle file.

    Args:
        parameters (dict): Dictionary containing paths for the fixed and moving images,
            and the output folder for storing results.

    Returns:
        tuple: (superpoint_reference, superpoint_moving, superpoint_desc_ref, superpoint_desc_mov, ref_score, mov_score)
            - superpoint_reference (np.ndarray): Keypoints from the fixed image.
            - superpoint_moving (np.ndarray): Keypoints from the moving image.
            - superpoint_desc_ref (np.ndarray): Descriptors for the fixed image.
            - superpoint_desc_mov (np.ndarray): Descriptors for the moving image.
            - ref_score (np.ndarray): Keypoint scores for the fixed image.
            - mov_score (np.ndarray): Keypoint scores for the moving image.
    """
    path_script_rep = utils.get_SuperPoint_path()
    path_script = os.path.join(path_script_rep, 'find_superpoints.py')
    os.system(f"python {path_script} {parameters['path_image_fixed']} {parameters['path_image_moving']} {parameters['folder_output_tile']} {path_script_rep}")

    with open(os.path.join(parameters['folder_output_tile'], 'superpoint_points.pickle'), 'rb') as handle:
        [superpoint_reference, superpoint_desc_ref, ref_score, superpoint_moving, superpoint_desc_mov, mov_score] = pickle.load(handle)
    
    return superpoint_reference, superpoint_moving, superpoint_desc_ref, superpoint_desc_mov, ref_score, mov_score


def get_silk_features(
    parameters: dict
):
    """
    Load precomputed SiLK features from a pickle file.

    Note: This function assumes that SiLK features have already been extracted and saved. The code
    for extracting SiLK features is not included.

    Args:
        parameters (dict): Dictionary containing the output folder for storing results.

    Returns:
        tuple: (silk_reference, silk_moving, silk_desc_ref, silk_desc_mov, None, None)
            - silk_reference (np.ndarray): Keypoints from the fixed image.
            - silk_moving (np.ndarray): Keypoints from the moving image.
            - silk_desc_ref (np.ndarray): Descriptors for the fixed image.
            - silk_desc_mov (np.ndarray): Descriptors for the moving image.
            - None: Placeholder for compatibility.
            - None: Placeholder for compatibility.
    """
    with open(os.path.join(parameters['folder_output_tile'], 'silk_points.pickle'), 'rb') as handle:
        [silk_reference, silk_desc_ref, silk_moving, silk_desc_mov] = pickle.load(handle)
    return silk_reference, silk_moving, silk_desc_ref, silk_desc_mov, None, None
