import os
import numpy as np
np.random.seed(42)
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import copy
import tifffile

# === I/O and File Utilities ===

def write_coordinates_txt(
    path_save: str,
    mp_fp: tuple
):
    """
    Write matched points to a text file in ImageJ format.

    Args:
        path_save (str): Path to save the text file.
        mp_fp (tuple): Tuple of (moving_points, fixed_points) arrays.
    """
    with open(path_save, 'w') as f:
        f.write(write_mp_fp_txt_format(mp_fp))

def write_mp_fp_txt_format(
    mp_fp: tuple
):
    """
    Formats matched points for ImageJ alignment in a tab-separated text format.

    Args:
        mp_fp (tuple): Tuple of (moving_points, fixed_points) arrays.

    Returns:
        str: Formatted string for writing to file.
    """
    text = 'Index\txSource\tySource\txTarget\tyTarget\n'
    # Only include pairs where the target point is not all zeros
    text += ''.join(
        f"{i}\t{round(src[0])}\t{round(src[1])}\t{round(tgt[0])}\t{round(tgt[1])}\n"
        for i, (src, tgt) in enumerate(zip(mp_fp[0], mp_fp[1])) if np.sum(tgt) != 0
    )
    return text

def load_coordinates(
    path_coordinates: str,
    combination: tuple
):
    """
    Load matched coordinates for a given registration method combination from a text file.

    Args:
        path_coordinates (str): Directory containing coordinate files.
        combination (tuple): Tuple specifying the registration method combination.

    Returns:
        tuple: (fp, mp) where fp are fixed/reference points and mp are moving points.
    """
    path_coordinates_combination = os.path.join(
        path_coordinates,
        'manual_selection.txt' if combination == 'manual' else f'{combination[0]}_{combination[1]}.txt'
    )
    with open(path_coordinates_combination, "r") as f:
        coordinates = f.readlines()[1:]  # Skip header line

    mp = []
    fp = []
    # Parse each line and extract coordinates
    for coord in coordinates:
        coord_splitted = coord.replace('\n', '').split('\t')[1:]
        fp.append([int(coord_splitted[0]), int(coord_splitted[1])])
        mp.append([int(coord_splitted[2]), int(coord_splitted[3])])
    return np.array(fp), np.array(mp)

def load_one_manual_coordinates(
    folder_output: str,
    current_tile: str
):
    """
    Load manually selected coordinates for a single tile from a .mat file.

    Args:
        folder_output (str): Path to the output folder containing the .mat file.
        current_tile (str): Identifier for the current tile.

    Returns:
        tuple: (mp, fp) where mp are moving points and fp are fixed/reference points.
    """
    mp_fp = sio.loadmat(
        os.path.join(
            folder_output,
            current_tile,
            'manual_selection.mat'
        )
    )
    mp, fp = mp_fp['mp'], mp_fp['fp']
    return mp, fp


# === Image Processing Utilities ===

def load_imgs(
    parameters_dict: dict
):
    """
    Load grayscale reference and moving images for all tiles in polarimetry experiments.

    Args:
        parameters_dict (dict): Dictionary where each key is a tile identifier and each value is a dictionary
                                containing 'path_image_fixed' and 'path_image_moving'.

    Returns:
        dict: Dictionary with the same keys as parameters_dict, each containing a dict with 'reference' and 'moving' images.
    """
    all_images = {}
    
    for key, parameters in parameters_dict.items():
        images = {}
        images['reference'] = cv2.imread(parameters['path_image_fixed'], cv2.IMREAD_GRAYSCALE)
        images['moving'] = cv2.imread(parameters['path_image_moving'], cv2.IMREAD_GRAYSCALE)
        all_images[key] = images
    
    return all_images

def resize_image(
    img: np.ndarray,
    img_moved: np.ndarray,
    resize_param: float
):
    """
    Resize two images by the same scaling factor.

    Args:
        img (np.ndarray): First image to resize.
        img_moved (np.ndarray): Second image to resize.
        resize_param (float): Scaling factor.

    Returns:
        tuple: (resized_img, resized_img_moved)
    """
    new_width = int(img.shape[1] * resize_param)
    new_height = int(img.shape[0] * resize_param)
    return (
        cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA),
        cv2.resize(img_moved, (new_width, new_height), interpolation=cv2.INTER_AREA)
    )


def get_bounding_box(
    mask: np.ndarray
):
    """
    Compute the bounding box of the largest object in a binary mask.

    Args:
        mask (np.ndarray): Binary mask image.

    Returns:
        tuple: (y_min, y_max, x_min, x_max) bounding box coordinates.
    """
    # Threshold the image to create a binary mask
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return y, y + h, x, x + w


def get_histology_img(
    path_mrxs: str,
    parameters: dict
):
    """
    Load and mask a histology image, returning a grayscale masked image and mask.

    Args:
        path_mrxs (str): Path to the MRXS file.
        parameters (dict): Parameters for loading the image.

    Returns:
        tuple: (masked grayscale image, rotated mask)
    """
    histology_image = HistologyImage(path_mrxs, parameters = parameters)
    rgb_image = rgba2rgb(histology_image.RGB_image)
    mask = histology_image.mask
    expanded_mask = mask[:, :, np.newaxis]
    
    rgb_image_mskd = np.rot90(np.where(expanded_mask, rgb_image, 0), k = -1)
    gs_image_mskd = rgb2gray(rgb_image_mskd)
    
    return gs_image_mskd, np.rot90(expanded_mask, k = -1)


def rgb2gray(
    rgb: np.ndarray 
):
    """
    Convert an RGB image to grayscale.

    Args:
        rgb (np.ndarray): RGB image.

    Returns:
        np.ndarray: Grayscale image.
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def rgba2rgb(
    rgba: np.ndarray,
    background: tuple =(255,255,255) 
):
    """
    Convert an RGBA image to RGB using a specified background color.

    Args:
        rgba (np.ndarray): RGBA image.
        background (tuple): Background color as (R, G, B).

    Returns:
        np.ndarray: RGB image.
    """
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')
    
def create_propagation_img(
    path_img: str
):
    """
    Create a coordinate map for resampling histology images, scaled for 
    ImageJ compatibility.

    Args:
        path_img (str): Path to the image file.

    Returns:
        list: [x_map, y_map] as uint16 arrays scaled to 16-bit range.
    """
    img = cv2.imread(path_img)
    # Initialize coordinate maps
    x_coords = np.arange(img.shape[1])
    y_coords = np.arange(img.shape[0])

    to_propagate = [np.tile(y_coords, (img.shape[1], 1)),
                    np.tile(x_coords, (img.shape[0], 1)).T]
            
    # Scale the coordinates for large histology images or polarimetric images
    scale = 2**16 - 1
    max_ = 2**16 - 1
    img_to_propagate = [
        np.clip(((to_propagate[1] / np.max(np.abs(to_propagate[1]))) * scale).T.astype('uint16'), 0, max_),
        np.clip(((to_propagate[0] / np.max(np.abs(to_propagate[0]))) * scale).T.astype('uint16'), 0, max_)
    ]
    return img_to_propagate

def stack_and_save_tiff(
    img_x_path: str,
    img_y_path: str,
    output_path: str
):
    """
    Load two single-channel TIFF images, stack them into a multi-channel array, and save as a multi-channel TIFF.

    Args:
        img_x_path (str): Path to the first TIFF image (e.g., registered_img_x.tif).
        img_y_path (str): Path to the second TIFF image (e.g., registered_img_y.tif).
        output_path (str): Path to save the combined multi-channel TIFF.
    """
    img_x = tifffile.imread(img_x_path)
    img_y = tifffile.imread(img_y_path)

    # Stack into a (H, W, 2) array
    print(f"Stacking images: {img_x_path} and {img_y_path}")
    print(f"Image shapes: img_x: {img_x.shape}, img_y: {img_y.shape}")
    stacked = np.stack([img_x[0], img_y[0]], axis=0)

    # Save as multi-channel TIFF
    tifffile.imwrite(output_path, stacked)
    
    
# === Plotting ===

def plot_results(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    matched_points_ref: np.ndarray,
    matched_points_mov_ori: np.ndarray,
    save_path: str,
    distance: int = 50,
    distances: list = None,
    histology: bool = False
):
    """
    Visualize and save the matching results between two images.

    This function creates a side-by-side plot of the reference and moving images,
    overlays the matched keypoints, and draws lines between corresponding matches.
    Matches are colored by distance (if provided) or by a color cycle.

    Args:
        reference_image (np.ndarray): The reference (fixed) image.
        moving_image (np.ndarray): The moving image.
        matched_points_ref (np.ndarray): Matched keypoints in the reference image (Nx2).
        matched_points_mov_ori (np.ndarray): Matched keypoints in the moving image (Nx2).
        save_path (str): Path to save the resulting plot.
        distance (int, optional): Horizontal gap between images in the plot. Default is 50.
        distances (list or np.ndarray, optional): Distances for each match (for coloring).
        histology (bool, optional): If True, use histology-specific thresholds/colors.

    Returns:
        None. The plot is saved to `save_path`.
    """
    plt.figure(figsize=(20, 20))

    # Deep copy to avoid modifying the original array
    matched_points_mov = copy.deepcopy(matched_points_mov_ori)

    # Limit the number of matches shown for clarity
    match_pts_number = 60
    if len(matched_points_ref) > match_pts_number:
        random_indices = np.random.choice(matched_points_ref.shape[0], size=match_pts_number, replace=False)
        matched_points_ref = matched_points_ref[random_indices]
        matched_points_mov = matched_points_mov[random_indices]
        if distances is not None:
            distances_mov = np.array(distances)[random_indices]
    else:
        distances_mov = distances

    # Assign colors to matches based on distance or a color cycle
    if distances is not None:
        rgb_cycle = []
        for dist in distances_mov:
            if histology:
                dist_thresh = [5, 10]
            else:
                dist_thresh = [30, 100]
            if dist < dist_thresh[0]:
                rgb_cycle.append([0, 1, 0])   # Green for close matches
            elif dist < dist_thresh[1]:
                rgb_cycle.append([0, 0, 1])   # Blue for medium matches
            else:
                rgb_cycle.append([1, 0, 0])   # Red for far matches
    else:
        # Use a color cycle if no distances are provided
        x = len(matched_points_ref)
        phi = np.linspace(0, 2 * np.pi, x)
        rgb_cycle = (np.stack((np.cos(phi), np.cos(phi + 2 * np.pi / 3), np.cos(phi - 2 * np.pi / 3))).T + 1) * 0.5

    # Shift moving points horizontally for side-by-side plotting
    if len(matched_points_mov) > 0:
        matched_points_mov[:, 0] = matched_points_mov[:, 0] + distance + reference_image.shape[1]

    # Create a blank canvas for both images
    array_plot = np.ones(
        (reference_image.shape[0], distance + reference_image.shape[1] + moving_image.shape[1])
    ) * 255

    # Place reference and moving images on the canvas
    array_plot[:, :reference_image.shape[1]] = reference_image
    array_plot[:, distance + reference_image.shape[1]:] = moving_image

    # Plot matched keypoints and lines
    if len(matched_points_mov) > 0:
        plt.scatter(
            *(matched_points_ref).T,
            s=70,
            c=rgb_cycle
        )
        plt.scatter(
            *(matched_points_mov).T,
            s=70,
            c=rgb_cycle
        )
        for mp, fp, rgb in zip(matched_points_ref, matched_points_mov, rgb_cycle):
            plt.plot(
                [mp[0], fp[0]],
                [mp[1], fp[1]],
                color=rgb,
                linewidth=1
            )

    plt.imshow(array_plot, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()


# === Metrics and Statistics ===

def bootstrap_parameters(
    distances: list,
    type_alignment: str,
    parameter: str = 'rmse',
    threshold: int = 30,
    num_samples: int =10000):
    """
    Perform bootstrapping to estimate the variance or precision of point distribution.

    This function resamples the input distances with replacement and computes the
    desired metric (RMSE or precision) for each bootstrap sample. It then returns
    the mean, confidence interval, and (for RMSE) the value scaled to physical units.

    Args:
        distances (list or np.ndarray): List of distances/errors.
        type_alignment (str): Alignment type ('HE' or other), used for scaling.
        parameter (str): Metric to bootstrap ('rmse' or 'precision'). Default is 'rmse'.
        threshold (float): Precision threshold (used if parameter='precision'). Default is 30.
        num_samples (int): Number of bootstrap samples. Default is 10000.

    Returns:
        tuple:
            - For 'rmse': (mean_rmse, ci_rmse, mean_rmse_scaled, ci_rmse_scaled)
            - For 'precision': (mean_precision, ci_precision)
    """
    bootstrap = []

    for _ in range(num_samples):
        # Sample distances with replacement
        resampled_points = np.random.choice(range(len(distances)), size=len(distances), replace=True)
        resampled_points_coords = [distances[i] for i in resampled_points]

        # Compute the metric for the resampled points
        if parameter == 'rmse':
            fun = calculate_rmse
        else:
            fun = calculate_precision
        bootstrap.append(fun(resampled_points_coords, threshold=threshold))

    # Compute mean and confidence intervals (2.5% and 97.5%)
    mean_variance = np.mean(bootstrap)
    lower_bound = np.percentile(bootstrap, 2.5)
    upper_bound = np.percentile(bootstrap, 97.5)

    # Set scaling factor for physical units
    # 1 pixel = 26um (histology), 34um (other)
    factor = 0.026 if type_alignment == 'HE' else 0.034

    if parameter == 'rmse':
        ci = (mean_variance - lower_bound + upper_bound - mean_variance) / 2
        return mean_variance, ci, mean_variance * factor, ci * factor
    else:
        ci = (mean_variance - lower_bound + upper_bound - mean_variance) / 2 * 100
        return mean_variance * 100, ci
    
def euclidean_distance(
    point1: np.ndarray,
    point2: np.ndarray
):
    """
    Compute the Euclidean distance between two points.

    Args:
        point1 (array-like): First point coordinates (x, y).
        point2 (array-like): Second point coordinates (x, y).

    Returns:
        float: Euclidean distance between point1 and point2.
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def calculate_rmse(
    distances: np.ndarray,
    threshold: int =None
):
    """
    Calculate the Root Mean Square Error (RMSE) from a list of errors.

    Args:
        distances (list or np.ndarray): List of distances values.
        threshold (float, optional): Not used. Included for compatibility.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean(np.square(distances)))

def calculate_precision(
    distances: np.ndarray,
    threshold: int = 30
):
    """
    Calculate the precision as the fraction of distances below a given threshold.

    Args:
        distances (list or np.ndarray): List of error values.
        threshold (float): Threshold value for precision calculation.

    Returns:
        float: Precision (fraction of errors < threshold).
    """
    distances = np.array(distances)
    return np.sum(distances < threshold) / len(distances) if len(distances) > 0 else 0


# === Third-party Path Helpers ===

def get_imgJ_script_path():
    """
    Get the path to the ImageJ script directory.

    Returns:
        str: Path to the ImageJ script directory.
    """
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'scripts',
        'imgJ_align.py'
    )

def get_third_party_path():
    """
    Get the path to the third-party directory containing external tools and libraries.

    Returns:
        str: Path to the third-party directory.
    """
    return os.path.join(os.path.realpath(__file__).split(f'src{os.sep}clfm')[0], 'third_party')

def get_OANet_path():
    return os.path.join(get_third_party_path(), 'OANet-master', 'demo')

def get_SuperPoint_path():
    return os.path.join(get_third_party_path(), 'SuperPoint')

def get_SuperGlue_path():
    return os.path.join(get_third_party_path(), 'SuperGluePretrainedNetwork')

def get_omniglue_path():
    return os.path.join(get_third_party_path(), 'omniglue')

def get_Fiji_path():
    return os.path.join(get_third_party_path(), 'Fiji.app')