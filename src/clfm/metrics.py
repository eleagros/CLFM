import os
import numpy as np
from PIL import Image
import tifffile
from tqdm import tqdm
from clfm import utils

def find_rmse(
    combinations: tuple,
    parameters: dict,
    images: dict,
    type_alignment: str,
    threshold: float
):
    """
    Compute RMSE and precision metrics for all tiles.

    Args:
        parameters_dict (dict): Dictionary with tile parameters.
        images (dict): Dictionary with loaded images.
        type_alignment (str): Alignment type.
        combinations (tuple): Tuple of registration method combinations.
        threshold (float): Maximum number of pixels to be considered a correct match.
        mm_per_pixel (float): Micrometers per pixel for the images.

    Returns:
        tuple: (rmses, metrics) dictionaries for each tile.
    """
    rmses = {}
    metrics = {}
                
    # Iterate over each of the images of the database
    for current_tile, value in tqdm(
        parameters.items(),
        total = len(parameters),
        desc ='Calculating RMSE'
    ):
        # Load the GT maps (generated automatically for HE or manually for polarimetry)
        path_gt_map = os.path.join(value['folder_gt_tile'], 'gt_backward.tif' if type_alignment == 'histology' else 'gt_map.tif')
        
        # Find the rmses for the current tile
        res = find_one_rmse(
            path_gt_map,
            combinations,
            images[current_tile],
            value,
            type_alignment,
            threshold=threshold
        )
        rmses[current_tile] = res[0]
        metrics[current_tile] = res[1]
    
    return rmses, metrics

def find_one_rmse(
    path_gt_map: str,
    combinations: tuple,
    images_tile: dict,
    value: dict,
    type_alignment: str,
    threshold: float
):
    """
    Compute RMSE and precision for a single tile and all registration method combinations.

    For each combination, this function:
      - Loads the ground truth mapping images (x and y maps).
      - Loads the matched coordinates for the current combination.
      - Propagates the ground truth mapping to the matched points.
      - Computes the Euclidean distance between the propagated and matched points.
      - Calculates RMSE and precision metrics.
      - Saves the distances to a file and generates a visualization.

    Args:
        path_gt_maps (str): Paths to ground truth mapping image.
        combinations (tuple): Tuple of registration method combinations.
        images_tile (dict): Dictionary with 'reference' and 'moving' images.
        value (dict): Dictionary with paths for coordinates, distances, and plots.
        type_alignment (str): Alignment type ('histology' or other).
        threshold (float): Maximum number of pixels to be considered a correct match.

    Returns:
        tuple: (rmses, metrics) dictionaries for each combination.
    """
    
    # Extract images from the tile
    reference_image = images_tile['reference']
    moving_image = images_tile['moving']
    
    # Extract paths and other parameters from the value dictionary
    path_coordinates = value['path_coordinates']
    path_distances = value['path_distances']
    path_plot = value['path_plot']
    
    rmses = {}
    metrics = {}
        
    # Load the ground truth mapping images
    gt_map = tifffile.imread(path_gt_map)
    img_x_gt = ((gt_map[0] / (2**16 - 1)) * reference_image.shape[1]).astype(np.uint16)
    img_y_gt = ((gt_map[1] / (2**16 - 1)) * reference_image.shape[0]).astype(np.uint16)

    for combination in combinations:

        # Load the coordinates for the current combination
        fp_full, mp_full = utils.load_coordinates(
            path_coordinates,
            combination
        )
        
        #  Initialize lists to store distances and matched points
        distances = []
        pt_annotated = []
        mp_current = []
        
        for (mp, fp) in zip(mp_full, fp_full):
            
            # Get the coordinates from the ground truth mapping images
            try:
                idx_new, idy_new = [img_x_gt[fp[1],fp[0]], img_y_gt[fp[1],fp[0]]]
            except:
                idx_new, idy_new = [0, 0]
                print(f'Error in {combination[0]}_{combination[1]}', path_coordinates)
            
            if (idx_new + idy_new == 0):
                pass
            else:
                # If the coordinates are valid, append them to the lists
                pt_annotated.append([idx_new, idy_new])
                mp_current.append(mp)
                
                # Calculate the Euclidean distance between the GT point and the matched point
                distance = utils.euclidean_distance(pt_annotated[-1], mp_current[-1])
                distances.append(distance)
           
        # Write the distances to a file
        with open(os.path.join(
            path_distances,
            f'{combination[0]}_{combination[1]}.txt'
        ), 'w') as f:
            for line in distances:
                f.write(f"{line}\n")
        
        # Calculate RMSE and precision metrics        
        rmse = utils.calculate_rmse(distances)
            
        precision = utils.calculate_precision(distances, threshold=threshold)
        
        rmses[combination] = rmse
        metrics[combination] = precision
            
        # Visualize the results
        utils.plot_results(
            moving_image,
            moving_image,
            np.array(pt_annotated),
            np.array(mp_current),
            os.path.join(
                path_plot, 
                f'{combination[0]}_{combination[1]}_propagated.pdf'
            ),
            distances = distances,
            histology = type_alignment == 'histology'
        )
        
    return rmses, metrics


def aggregate_and_bootstrap_metrics(
    parameters: dict,
    combinations: tuple,
    type_alignment: str,
    threshold: float,
    mm_per_pixel: float,
    num_samples=1000,
    
):
    """
    Aggregate distances for each combination across all tiles and compute bootstrapped RMSE and precision.

    Args:
        parameters_dict (dict): Dictionary with tile parameters (must contain 'path_distances').
        combinations (tuple): Tuple of registration method combinations.
        type_alignment (str): Alignment type ('polarimetry', 'histology', etc.).
        threshold (float): Maximum number of pixels to be considered a correct match.
        mm_per_pixel (float): Micrometers per pixel for the images.
        num_samples (int): Number of bootstrap samples for statistics.

    Returns:
        tuple: (all_distances, rmses, metrics)
            - all_distances: dict of all distances per combination
            - rmses: dict of bootstrapped RMSE results per combination
            - metrics: dict of bootstrapped precision results per combination
    """
    all_distances = {}
    # Aggregate distances for each combination
    for combination in combinations:
        distances = []
        for key, val in parameters.items():
            path_distances = val['path_distances']
            try:
                with open(
                    os.path.join(
                        path_distances,
                        f'{combination[0]}_{combination[1]}.txt'
                    )
                ) as file:
                    distance_tile = file.readlines()
                for dist in distance_tile:
                    distances.append(float(dist.strip()))
            except Exception:
                pass
        all_distances[combination] = distances

    rmses = {}
    metrics = {}
    
    # Compute bootstrapped RMSE and precision for each combination
    for key, distance in tqdm(
        all_distances.items(),
        total=len(all_distances),
        desc='Bootstrapping metrics'
    ):
        rmses[key] = utils.bootstrap_parameters(
            distance,
            type_alignment,
            threshold=threshold,
            mm_per_pixel=mm_per_pixel,
            parameter='rmse',
            num_samples=num_samples
        )
        metrics[key] = utils.bootstrap_parameters(
            distance,
            type_alignment,
            threshold=threshold,
            mm_per_pixel=mm_per_pixel,
            parameter='precision',
            num_samples=num_samples
        )

    return rmses, metrics

def print_bootstrap_results(
    rmses: dict,
    precisions: dict
):
    """
    Pretty-print and style the RMSE and precision bootstrap results for each combination.

    Args:
        rmses (dict): Dictionary with combinations as keys and RMSE tuples as values.
        metrics (dict): Dictionary with combinations as keys and precision tuples as values.
    """
    print("="*95)
    print("{:<25} | {:>10} ± {:<10} | {:>10} ± {:<10} | {:>8} ± {:<8}".format(
        "Combination", "RMSE(px)", "error", "RMSE(mm)", "error", "Prec(%)", "error"
    ))
    print("-"*95)
    for key in rmses:
        rmse_px, rmse_px_ci, rmse_mm, rmse_mm_ci = rmses[key]
        prec, prec_ci = precisions[key]
        comb_str = f"{key[0]} + {key[1]}"
        print("{:<25} | {:10.2f} ± {:<10.2f} | {:10.2f} ± {:<10.2f} | {:8.2f} ± {:<8.2f}".format(
            comb_str, rmse_px, rmse_px_ci, rmse_mm, rmse_mm_ci, prec, prec_ci
        ))
    print("="*95)