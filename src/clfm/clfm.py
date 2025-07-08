import os
import re
import numpy as np
import time
from tqdm import tqdm

from clfm import utils, get_features, get_matches

class CLFMProject:
    """
    CLFMProject manages configuration, parameters, and image preparation for a CLFM alignment project.

    This class sets up the directory structure, collects all relevant file paths for each image pair,
    and prepares images for downstream feature matching and registration.

    Attributes:
        repetitions (int): Number of repetitions for timing or benchmarking.
        type_alignment (str): Type of alignment ('histology' or 'polarimetry').
        path_main_data (str): Path to the main data directory.
        parameters_dict (dict): Dictionary of parameters for each image pair/tile.
        images (dict): Dictionary of loaded images for each tile (after prepare_imgs()).

    Methods:
        get_parameters():
            Collects and organizes all file paths and directories for each image pair/tile.
        prepare_imgs():
            Loads and prepares images for all tiles, depending on the alignment type.
    """
    def __init__(
        self,
        path_database: str,
        type_alignment: str,
        threshold: float = None,
        mm_per_pixel: float = None
    ):
        """
        Initialize the CLFMProject.

        Args:
            path_database (str): Path to the main data directory.
            type_alignment (str): Type of alignment ('histology' or 'polarimetry').
            threshold (float): Maximum number of pixels to be considered a correct match.
            mm_per_pixel (float): mm_per_pixel per pixel for the images.
        Raises:
            ValueError: If type_alignment is not one of 'histology', 'polarimetry', or 'custom'.
        """
        self.repetitions = 1
        if type_alignment not in ['histology', 'polarimetry', 'custom']:
            raise ValueError("type_alignment must be one of: 'histology', 'polarimetry', or 'custom'")
        self.type_alignment = type_alignment
        self.path_main_data = path_database
        if self.type_alignment == 'custom':
            if threshold is None or mm_per_pixel is None:
                raise ValueError("For 'custom' alignment, both 'threshold' and 'um_per_pixel' must be provided.")
            self.threshold = threshold
            self.mm_per_pixel = mm_per_pixel
        else:
            # Default values for histology and polarimetry
            self.threshold = 15 if self.type_alignment == 'polarimetry' else 10
            self.mm_per_pixel = 0.026 if self.type_alignment == 'histology' else 0.050
            
        self.get_parameters()

    def get_parameters(
        self
    ):
        """
        Collects and organizes all file paths and directories for each image pair/tile.

        Populates self.parameters_dict with a dictionary for each tile, containing paths for:
        - data, output, results, coordinates, plots, distances
        - fixed and moving images

        Also ensures all necessary directories exist.
        """
        self.folder_alignment = self.path_main_data
        self.folder_data = os.path.join(self.folder_alignment, 'data')
        self.folder_output = os.path.join(self.folder_alignment, 'output')
        self.folder_result = os.path.join(self.folder_alignment, 'results')
        self.folder_gt = os.path.join(self.folder_alignment, 'ground_truth')
            
        self.parameters_dict = {}

        pairs = set(
            re.search(r'pair_(\d{4})_', fname).group(1)
            for fname in os.listdir(self.folder_data)
            if re.search(r'pair_(\d{4})_', fname)
        )
        
        for tile in pairs:
            parameters_current_tile = {
                'current_tile': tile,
                'type_alignment': self.type_alignment,
                'folder_data': self.folder_data,
                'folder_output': self.folder_output,
                'folder_output_tile': os.path.join(self.folder_output, tile),
                'folder_result': self.folder_result,
                'folder_result_tile': os.path.join(self.folder_result, tile),
                'folder_gt': self.folder_gt,
                'folder_gt_tile': os.path.join(self.folder_gt, tile),
                'path_coordinates': os.path.join(self.folder_result, tile, "coordinates"),
                'path_plot': os.path.join(self.folder_result, tile, "plots"),
                'path_distances': os.path.join(self.folder_result, tile, "distances"),
                'path_image_fixed': os.path.join(self.folder_data, f"pair_{tile}_fixed.png"),
                'path_image_moving': os.path.join(self.folder_data, f"pair_{tile}_moving.png"),
            }

            # Ensure all necessary directories exist
            for path in [
                parameters_current_tile['folder_output'],
                parameters_current_tile['folder_result'],
                parameters_current_tile['folder_gt'],
                parameters_current_tile['folder_output_tile'],
                parameters_current_tile['folder_result_tile'],
                parameters_current_tile['folder_gt_tile']
                ]:
                os.makedirs(path, exist_ok=True)
                
            for path in [
                parameters_current_tile['path_coordinates'],
                parameters_current_tile['path_plot'],
                parameters_current_tile['path_distances']
                ]:
                os.makedirs(path, exist_ok=True)

            self.parameters_dict[tile] = parameters_current_tile
            
    
    def prepare_imgs(
        self
    ):
        """
        Loads and prepares images for all tiles, depending on the alignment type.

        Raises:
            NotImplementedError: If 'HE' alignment is requested (not implemented).
            ValueError: If an invalid alignment type is provided.
        """
        self.images = utils.load_imgs(self.parameters_dict)

class Matcher:
    """
    Matcher class for local feature matching comparisons.

    This class encapsulates the logic for extracting features from image pairs,
    matching them using various algorithms, and optionally post-processing the matches.

    Attributes:
        feature_points (str): The feature extraction method ('sift', 'superpoint', 'silk', 'omniglue').
        matcher_method (str): The matching method ('mnn', 'oanet', 'superglue').
        repetitions (int): Number of times to repeat matching for timing statistics.
        matcher (callable): The matching function to use.
        fun_pts (callable): The feature extraction function.
        fun (callable): The feature matching function.
        matched_pts (dict): Stores matched points for each image pair.
    """
    
    def __init__(
        self,
        feature_points: str ='sift',
        matcher_method: str ='mnn',
        repetitions: int =1
    ):
        """
        Initialize the Matcher.

        Args:
            feature_points (str): Feature extraction method ('sift', 'superpoint', 'silk', 'omniglue').
            matcher_method (str): Matching method ('mnn', 'oanet', 'superglue', 'omniglue').
            repetitions (int): Number of repetitions for timing.
        Raises:
            ValueError: If an invalid feature or matcher method is provided.
        """
        self.feature_points = feature_points
        self.matcher_method = matcher_method
        self.repetitions = repetitions

        # Set feature extraction and matching functions
        if feature_points == 'omniglue':
            self.matcher = get_matches.match_omniglue
            self.fun_pts = None
            self.fun = None
        else:
            if feature_points == 'sift':
                self.fun_pts = get_features.get_sift_features
            elif feature_points == 'superpoint':
                self.fun_pts = get_features.get_superpoint_features
            elif feature_points == 'silk':
                self.fun_pts = get_features.get_silk_features
            else:
                raise ValueError('Invalid feature points method.')

            if matcher_method == 'mnn':
                self.fun = get_matches.mnn
            elif matcher_method == 'oanet':
                self.fun = get_matches.oanet_matcher
            elif matcher_method == 'superglue':
                self.fun = get_matches.superglue_matcher
            else:
                raise ValueError('Invalid matcher method.')

            self.matcher = self.matcher_general
            
    def run(
        self,
        parameters_dict: dict,
        images: dict
    ):
        """
        Run the matching process for all image pairs.

        Args:
            parameters_dict (dict): Dictionary of parameters for each image pair.
            images (dict): Dictionary of loaded images for each key (obtained from CLFMProject).

        Side Effects:
            - Writes matched coordinates to text files.
            - Generates and saves match visualization plots.
            - Prints timing statistics.
        """
        times = []
        self.matched_pts = {}

        for key, parameters in tqdm(
            parameters_dict.items(),
            total =  len(parameters_dict),
            desc='Local feature matching'
        ):
            
            for _ in range(self.repetitions):
                start_time = time.time()
                if self.feature_points == 'omniglue':
                    self.matched_pts[key] = self.matcher(parameters)
                else:
                    self.matched_pts[key] = self.matcher(self.fun_pts, self.fun, parameters)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Write matched coordinates to text files
            utils.write_coordinates_txt(
                f"{parameters['path_coordinates']}/{self.feature_points}_{self.matcher_method}.txt",
                [self.matched_pts[key][0], self.matched_pts[key][1]]
            )

            # Generate visualization plots
            utils.plot_results(
                images[key]['reference'], images[key]['moving'],
                self.matched_pts[key][0], self.matched_pts[key][1],
                f"{parameters['path_plot']}/{self.feature_points}_{self.matcher_method}.pdf"
            )
            
        # Print timing statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        print('Processing time: {:.2f} ± {:.2f} s'.format(mean_time, std_time))

    def matcher_general(
        self,
        fun_pts: callable,
        fun: callable,
        parameters
    ):
        """
        General matcher pipeline for non-omniglue methods.

        Args:
            fun_pts (callable): Feature extraction function.
            fun (callable): Feature matching function.
            parameters (dict): Parameters for the current image pair.

        Returns:
            tuple: (matched_ref, matched_mov) arrays of matched keypoints.
        """
        ref, mov, desc_ref, desc_mov, score_ref, score_mov = fun_pts(parameters)
        matched_ref, matched_mov = fun(ref, mov, desc_ref, desc_mov, score_ref, score_mov, parameters)
        return matched_ref, matched_mov

