import os, shutil
import cv2
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
import math

from comparemethods import plot_results, helpers

def create_propagation_img(path_img):
    """
    creates a coordinate map for resampling histology images, scaled based on the processing type.
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
    img_to_propagate = [np.clip(((to_propagate[1] / np.max(np.abs(to_propagate[1]))) * scale).T.astype('uint16'), 0, max_),
                        np.clip(((to_propagate[0] / np.max(np.abs(to_propagate[0]))) * scale).T.astype('uint16'), 0, max_)]
        
    return img_to_propagate

def align_images(parameters_dict, combinations, coordinates = None):
    
    for key, value in tqdm(parameters_dict.items(), total = len(parameters_dict)):
        
        folder_data = value['folder_data']
        current_tile = key
        path_coordinates = value['path_coordinates']
        path_mapped = value['path_mapped']
        
        if coordinates is not None:
            folder_output = value['folder_output']
        else:
            folder_output = ''
            
        align_one_image(combinations, folder_data, current_tile, path_coordinates, path_mapped, folder_output = folder_output)
        
        
def align_one_image(combinations, folder_data, current_tile, path_coordinates, path_mapped, folder_output = '', histo_histo = False):
    
    for combination in combinations:

        shutil.rmtree('./tmp', ignore_errors=True)
        os.makedirs('./tmp', exist_ok=True)
        
        if histo_histo:
            src = os.path.join(folder_data, current_tile, 'histology', 'HE.png')
        else:
            src = os.path.join(folder_data, current_tile, 'global_no_bg.png')
        dst = os.path.join('./tmp', 'global_no_bg.png')
        shutil.copy(src, dst)
        
        if histo_histo:
            src = os.path.join(folder_data, current_tile, 'histology', 'LFB.png')
        else:
            src = os.path.join(folder_data, current_tile, 'moving.png')
        dst = os.path.join('./tmp', 'moving.png')
        shutil.copy(src, dst)

        
        if histo_histo:
            coordinates_path = os.path.join(folder_data, current_tile, 'superglue', 'coordinates.txt')
        elif type(combination) == str:
            coordinates_mat = f'{folder_output}/{current_tile}/manual_selection.mat'
            coordinates_path = f'{folder_output}/{current_tile}/manual_selection.txt'
            
            if os.path.exists(coordinates_mat):
                mp_fp = helpers.get_mp_fp(coordinates_mat)
                helpers.write_coordinates_txt(coordinates_path, [mp_fp['mp'], mp_fp['fp']])
            else:
                raise ValueError('Manual selection file not found.')
        else:
            coordinates_path = f'{path_coordinates}/{combination[0]}_{combination[1]}.txt'
        
        dst = os.path.join('./tmp', 'coordinates.txt')
        shutil.copy(coordinates_path, dst)
     
        img_to_propagate = create_propagation_img(os.path.join('./tmp', 'global_no_bg.png'))
        cv2.imwrite('./tmp/img_x.tif', img_to_propagate[0])
        cv2.imwrite('./tmp/img_y.tif', img_to_propagate[1])
        
        os.system('python imgJ_align.py ./tmp')
        
        if histo_histo:
            path_results = os.path.join(folder_data, current_tile, 'superglue')
        elif type(combination) == str:
            path_results = os.path.join(f'{folder_output}/{current_tile}')
        else:
            path_results = os.path.join(f'{path_mapped}/{combination[0]}_{combination[1]}')
            shutil.rmtree(path_results, ignore_errors=True)
        os.makedirs(path_results, exist_ok=True)
        
        src = './tmp/registered_img_x.tif'
        dst = os.path.join(path_results, 'registered_img_x.tif')
        shutil.copy(src, dst)
        
        src = './tmp/registered_img_y.tif'
        dst = os.path.join(path_results, 'registered_img_y.tif')
        shutil.copy(src, dst)
        
    shutil.rmtree('./tmp', ignore_errors=True)
    
    
def load_manual_coordinates(parameters_dict):
    coordinates = {}
    
    for key, value in parameters_dict.items():
        folder_output = value['folder_output']
        current_tile = key
        mp, fp = load_one_manual_coordinates(folder_output, current_tile)
        coordinates[key] = [mp, fp]
        
    return coordinates


def load_one_manual_coordinates(folder_output, current_tile):
    mp_fp = sio.loadmat(f'{folder_output}/{current_tile}/manual_selection.mat')
    mp, fp = mp_fp['mp'], mp_fp['fp']
    return mp, fp


def find_rmse(parameters_dict, images, combinations, type_alignment = 'histology'):
    rmses = {}
    metrics = {}
        
    for key, value in tqdm(parameters_dict.items(), total = len(parameters_dict)):
        current_tile = key
        if type_alignment == 'histology':
            folder_data = value['folder_data']
            path_gt_maps = [f'{folder_data}/{current_tile}/superglue/registered_img_inv_x.tif', 
                            f'{folder_data}/{current_tile}/superglue/registered_img_inv_y.tif']
            
        else:
            folder_data = value['folder_output']
            path_gt_maps = [f'{folder_data}/{current_tile}/registered_img_x.tif',
                            f'{folder_data}/{current_tile}/registered_img_y.tif']
            
        path_coordinates = value['path_coordinates']
        path_distances = value['path_distances']
        path_plot = value['path_plot']
        reference_image = images[key]['reference']
        moving_image = images[key]['moving']
        res = find_one_rmse(path_gt_maps, combinations, reference_image, moving_image, path_coordinates, path_distances, path_plot, type_alignment)
        rmses[key] = res[0]
        metrics[key] = res[1]
    
    return rmses, metrics

def find_one_rmse(path_gt_maps, combinations, reference_image, moving_image, path_coordinates, path_distances, path_plot, type_alignment):
    rmses = {}
    metrics = {}
        
    img_x_gt = ((np.array(Image.open(path_gt_maps[0])) / (2**16 - 1)) * reference_image.shape[1]).astype(np.uint16)
    img_y_gt = ((np.array(Image.open(path_gt_maps[1])) / (2**16 - 1)) * reference_image.shape[0]).astype(np.uint16)

    for combination in combinations:

        path_coordinates_combination = f'{path_coordinates}/{combination[0]}_{combination[1]}.txt'
        f = open(path_coordinates_combination, "r")
        coordinates = f.readlines()[1:]
        f.close()

        mp = []
        fp = []
        
        
        for coord in coordinates:
            coord_splitted = coord.replace('\n', '').split('\t')[1:]
            mp.append([int(coord_splitted[0]), int(coord_splitted[1])])
            fp.append([int(coord_splitted[2]), int(coord_splitted[3])])

        mp_full = np.array(mp)
        fp_full = np.array(fp)
        
        distances = []

        pt_annotated = []
        mp_current = []
        
        # with open(r'C:\Users\romai\Documents\repositories\compareMatchingMethods\notebooks\filename.pickle', 'wb') as handle:
        #     pickle.dump([reference_image, moving_image, mp_full, fp_full, img_x_gt, img_y_gt], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        for (mp, fp) in zip(mp_full, fp_full):
            
            try:
                idx_new, idy_new = [img_x_gt[mp[1],mp[0]], img_y_gt[mp[1],mp[0]]]
            except:
                print(mp, img_x_gt.shape)
                idx_new, idy_new = [0, 0]
                print(f'Error in {combination[0]}_{combination[1]}', path_coordinates)
            
            if (idx_new + idy_new == 0):
                pass
            else:
                pt_annotated.append([idx_new, idy_new])
                mp_current.append(fp)
                distance = euclidean_distance(pt_annotated[-1], mp_current[-1])
                distances.append(distance)
           
        with open(f'{path_distances}/{combination[0]}_{combination[1]}.txt', 'w') as f:
            for line in distances:
                f.write(f"{line}\n")
                
        rmse = calculate_rmse(distances)
        
        if type_alignment == 'histology':
            threshold = 10
        else:
            threshold = 30
            
        tp = np.array(distances) < threshold
        fp = np.array(distances) > threshold
        precision = np.sum(tp) / (np.sum(tp) + np.sum(fp))
        
        rmses[combination] = rmse
        metrics[combination] = precision
            
        plot_results.plot_results(moving_image, moving_image, np.array(pt_annotated), np.array(mp_current),
                                  f'{path_plot}/{combination[0]}_{combination[1]}_propagated.pdf', distances = distances, histology = type_alignment == 'HE')
        
    return rmses, metrics
    
    
import cv2
import numpy as np
from sklearn.neighbors import KDTree

def calculate_FN(fp, sp_sg_matched_points_ref, threshold=50):
    TP = 0
    matched_gt_indices = set()  # To avoid double counting

    # Create a KDTree for fast nearest neighbor search
    gt_points = np.array([f for f in fp])
    tree = KDTree(gt_points)

    # Calculate True Positives
    for detected in sp_sg_matched_points_ref:

        distances, indices = tree.query([detected], k=1)
        
        if distances[0][0] < threshold:
            TP += 1
            matched_gt_indices.add(indices[0][0])  # Mark this GT match as used

    # Count False Negatives
    FN = len(fp) - len(matched_gt_indices)

    return FN

"""
def find_rmse(parameters_dict, images, coordinates, combinations):
    all_rmses = {}
    all_metrics = {}
    
    for key, value in parameters_dict.items():
        res = find_one_rmse(combinations, images[key]['reference'], images[key]['moving'],
                      coordinates[key][0], coordinates[key][1], value['path_mapped'],
                      value['path_plot'], value['path_distances'], value['path_coordinates'], folder_output = os.path.join(value['folder_output'], key))
        all_rmses[key] = res[0]
        all_metrics[key] = res[1]
    
    return all_rmses, all_metrics
    
def find_one_rmse(combinations, reference_image, moving_image, mp, fp, path_mapped, path_plot, path_distances, path_coordinates, folder_output = ''):
    rmses = {}
    metrics = {}
    
    img_x_gt = ((np.array(Image.open(f'{folder_output}/registered_img_x.tif')) / (2**16 - 1)) * reference_image.shape[1]).astype(np.uint16)
    img_y_gt = ((np.array(Image.open(f'{folder_output}/registered_img_y.tif')) / (2**16 - 1)) * reference_image.shape[0]).astype(np.uint16)
    
    for combination in combinations:
                
        path_results = f'{path_mapped}/{combination[0]}_{combination[1]}'
        img_x = ((np.array(Image.open(os.path.join(path_results, 'registered_img_x.tif').replace('\\', '/'))) / (2**16 - 1)) * reference_image.shape[1]).astype(np.uint16)
        img_y = ((np.array(Image.open(os.path.join(path_results, 'registered_img_y.tif').replace('\\', '/'))) / (2**16 - 1)) * reference_image.shape[0]).astype(np.uint16)
        
        pt_annotated = []
        mp_current = []
        
        for pt_ref, _ in zip(fp, mp):

            x, y = np.where(np.logical_and(img_x == int(pt_ref[0]), img_y == int(pt_ref[1])))
            x_man, y_man = np.where(np.logical_and(img_x_gt == int(pt_ref[0]), img_y_gt == int(pt_ref[1])))

            if len(y) == 0 or len(y_man) == 0:
                pass
            
            else:
                if len(y_man) == 1:
                    pt_manual = [x_man[0], y_man[0]]
                else:
                    pt_manual = [int(np.mean(x_man)), int(np.mean(y_man))]
                    
                
                    
                if len(y) == 1:
                    pt_annotated.append([x[0], y[0]])
                    mp_current.append(pt_manual)
                else:
                    mp_current.append(pt_manual)
                    pt_annotated.append([int(np.mean(x)), int(np.mean(y))])
                 
        distances = []
        
        if len(pt_annotated) == len(mp_current) == 0:
            rmse = math.nan
        else:
            pt_annotated = np.array(pt_annotated)[:, [1, 0]]
            mp_current = np.array(mp_current)[:, [1, 0]]
            for pt, fp_c in zip(pt_annotated, mp_current):
                distances.append(euclidean_distance(pt, fp_c))
            rmse = calculate_rmse(distances)
        
        threshold = 30 # 30pixels = 1mm
    
        true_p = np.array(distances) < threshold
        false_p = np.array(distances) > threshold
    
        precision = np.sum(true_p) / (np.sum(true_p) + np.sum(false_p))
        
        metrics[combination] = precision
            
        with open(f'{path_distances}/{combination[0]}_{combination[1]}.txt', 'w') as f:
            for line in distances:
                f.write(f"{line}\n")
        
        rmses[combination] = rmse

        plot_results.plot_results(moving_image, moving_image, mp_current, pt_annotated,
                                  f'{path_plot}/{combination[0]}_{combination[1]}_propagated.pdf', distances = distances)
        
    return rmses, metrics
    """
    
def bootstrap_parameters(distances, type_alignment, parameter = 'rmse', threshold = 30, num_samples=10000):
    """
    Perform bootstrapping to estimate the variance of point distribution.

    Args:
    points (list of tuples): List of (x, y) coordinates representing points on the image.
    image_size (tuple): Image dimensions as (width, height).
    grid_size (tuple): Grid size as (num_cols, num_rows).
    num_samples (int): Number of bootstrap samples.

    Returns:
    tuple: Bootstrap mean variance, confidence interval (2.5%, 97.5%).
    """
    bootstrap = []

    for _ in range(num_samples):
        # Sample points with replacement
        resampled_points = np.random.choice(range(len(distances)), size=len(distances), replace=True)
        resampled_points_coords = [distances[i] for i in resampled_points]

        # Compute the variance for the resampled points
        if parameter == 'rmse':
            fun = calculate_rmse
        else:
            fun = calculate_precision
        bootstrap.append(fun(resampled_points_coords, threshold = threshold))
    
    # Compute the mean variance and confidence intervals (2.5% and 97.5%)
    mean_variance = np.mean(bootstrap)
    lower_bound = np.percentile(bootstrap, 2.5)
    upper_bound = np.percentile(bootstrap, 97.5)

    # 1 pixel = 0.0034 cm = 0.034 mm = 34 um
    # 1 pixel = 15um (histology)
    if type_alignment == 'histology':
        factor = 0.015
    else:
        factor = 0.034
    
    if parameter == 'rmse':
        return mean_variance, (mean_variance - lower_bound + upper_bound - mean_variance)/2, mean_variance * factor, (mean_variance - lower_bound + upper_bound - mean_variance)/2 * factor
    else:
        return mean_variance * 100, (mean_variance - lower_bound + upper_bound - mean_variance)/2 * 100


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def calculate_rmse(errors, threshold = None):
    return np.sqrt(np.mean(np.square(errors)))

def calculate_precision(errors, threshold = 30):
    return np.sum(np.array(errors) < threshold) / len(errors)