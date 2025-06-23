import os, shutil
import cv2
from tqdm import tqdm

from clfm import utils

def register_images(
    image_pairs: dict,
    combinations: tuple
):
    """
    Aligns images based on provided image pairs and combinations of alignment methods.

    Args:
        image_pairs (dict): Dictionary where each key is a tile identifier, and each value is a dictionary
            containing the paths to the fixed and moving images, and the folder for output results.
        combinations (tuple): Tuple of method combinations to use for alignment.

    Returns:
        None
    """
    for _, image_pair in tqdm(image_pairs.items(), total = len(image_pairs), desc='Aligning images'):
        align_one_image(image_pair, combinations)
        
    
def align_one_image(
    parameters: dict,
    combinations: tuple,
    histo_histo: bool = False
):
    """
    Aligns one image pair based on the provided parameters and combinations of alignment methods.

    Args:
        parameters (dict): Dictionary containing the paths to the fixed and moving images,
            the folder for output results, and other necessary parameters.
        combinations (tuple): Tuple of method combinations to use for alignment.
        histo_histo (bool): Flag indicating whether the registration is histology to histology.

    Returns:
        None
    """
    for combination in combinations:
        
        # Create temporary directory for image registration
        shutil.rmtree('./tmp', ignore_errors=True)
        os.makedirs('./tmp', exist_ok=True)

        
        # Copy the fixed image to the temporary directory
        if histo_histo:
            raise ValueError('Histology to histology registration not implemented yet')
        else:
            src = parameters['path_image_fixed']
        dst = os.path.join('./tmp', 'global_no_bg.png')
        shutil.copy(src, dst)

        
        # Copy the moving image to the temporary directory
        if histo_histo:
            raise ValueError('Histology to histology registration not implemented yet')
        else:
            src = parameters['path_image_moving']
        dst = os.path.join('./tmp', 'moving.png')
        shutil.copy(src, dst)


        # Copy the coordinates file to the temporary directory
        if histo_histo:
            raise ValueError('Histology to histology registration not implemented yet')
            coordinates_path = os.path.join(folder_data, current_tile, 'superglue', 'coordinates.txt')
        elif type(combination) == str:
            coordinates_path = os.path.join(
                parameters['folder_gt_tile'],
                'manual_selection.txt'
            )
        else:
            raise ValueError('Combination type not recognized')
        dst = os.path.join('./tmp', 'coordinates.txt')
        shutil.copy(coordinates_path, dst)

        # Create the images that will be deformed and later used for resampling
        img_to_propagate = utils.create_propagation_img(
            os.path.join(
                './tmp',
                'global_no_bg.png'
            )
        )
        cv2.imwrite('./tmp/img_x.tif', img_to_propagate[0])
        cv2.imwrite('./tmp/img_y.tif', img_to_propagate[1])

        # Create the folder used to store the results
        if histo_histo:
            raise ValueError('Histology to histology registration not implemented yet')
            path_results = os.path.join(folder_data, current_tile, 'superglue')
        elif type(combination) == str:
            path_results = parameters['folder_gt_tile']
        else:
            raise ValueError('Combination type not recognized')
        os.makedirs(path_results, exist_ok=True)
                
        # Run registration if not already done
        gt_map = os.path.join(
            path_results,
            'gt_map.tif'
        )
        
        if not (os.path.exists(gt_map)):
            os.system(f'python {utils.get_imgJ_script_path()} ./tmp')
            utils.stack_and_save_tiff(
                './tmp/registered_img_x.tif',
                './tmp/registered_img_y.tif',
                gt_map
            )
            
    # Remove the temporary directory
    shutil.rmtree('./tmp', ignore_errors=True)