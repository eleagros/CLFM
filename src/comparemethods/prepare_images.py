import os
import pickle

import cv2
from PIL import Image
import SimpleITK as sitk

import numpy as np

from comparemethods import alignment
from poltilereconstructor.align_images.histology_image import HistologyImage


def prepare_imgs(parameters_dict, type_alignment):
    if type_alignment == 'HE':
        return prepare_img_histology(parameters_dict)
    else:
        return prepare_img_polarimetry(parameters_dict)


def prepare_img_polarimetry(parameters_dict):
    all_images = {}
    
    for key, parameters_key in parameters_dict.items():
        images = {}
        ref_img, mov_img = prepare_one_img_polarimetry(parameters_key['folder_data'], key)
        images['reference'] = ref_img
        images['moving'] = mov_img
        all_images[key] = images
    
    return all_images


def prepare_one_img_polarimetry(folder_data, current_tile):
    reference_image = cv2.imread(os.path.join(folder_data, current_tile, 'global_no_bg.png'), cv2.IMREAD_GRAYSCALE)
    moving_image = cv2.imread(os.path.join(folder_data, current_tile, 'moving.png'), cv2.IMREAD_GRAYSCALE)
    
    shape_A1 = (794,)
    if reference_image.shape[0]  == shape_A1[0]:
        pass
    else:
        ratio = shape_A1[0] / reference_image.shape[0]
        reference_image = cv2.resize(reference_image, (int(reference_image.shape[1] * ratio), int(reference_image.shape[0] * ratio)))
        cv2.imwrite(os.path.join(folder_data, current_tile, 'global_no_bg.png'), reference_image)
        
    if reference_image.shape[0]  == moving_image.shape[0]:
        pass
    else:
        ratio = reference_image.shape[0] / moving_image.shape[0]
        moving_image = cv2.resize(moving_image, (int(moving_image.shape[1] * ratio), int(moving_image.shape[0] * ratio)))
        cv2.imwrite(os.path.join(folder_data, current_tile, 'moving.png'), moving_image)
        
    cv2.imwrite(os.path.join(folder_data, current_tile, 'global_no_bg.png'), cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(folder_data, current_tile, 'moving.png'), cv2.cvtColor(moving_image, cv2.COLOR_BGR2RGB))
    
    return reference_image, moving_image
        
    
def prepare_img_histology(parameters_dict):
    all_images = {}
    
    for key, parameters_key in parameters_dict.items():
        images = {}
        ref_img, mov_img = prepare_one_img_histology(parameters_key['folder_data'], key, parameters_key['path_mrxs'], parameters_key['parameters'])
        images['reference'] = ref_img
        images['moving'] = mov_img
        all_images[key] = images
    
    return all_images
    
    
def prepare_one_img_histology(folder_data, current_tile, path_mrxs, parameters):
    
    path_superglue_script = f'./../src/comparemethods/superglue/demo_superglue.py'
    folder_output_superglue = os.path.abspath(os.path.join(os.path.abspath(folder_data), current_tile, 'superglue'))
    cmd = f"python {path_superglue_script} --input {os.path.join(os.path.abspath(folder_data), current_tile, 'histology')} --output_dir {folder_output_superglue} --no_display --resize -1 --histology 1"
    os.system(cmd)
        
    with open(f'{folder_output_superglue}/matches_000000_000001.pickle', 'rb') as handle:
        [matches_ref, matches_mov] = pickle.load(handle)
    
    to_propagate = alignment.create_propagation_img(f'{folder_data}/{current_tile}/histology/HE.png')
    registered_imgs = align_with_sitk(cv2.imread(f'{folder_data}/{current_tile}/histology/LFB.png', cv2.IMREAD_GRAYSCALE), cv2.imread(f'{folder_data}/{current_tile}/histology/HE.png', cv2.IMREAD_GRAYSCALE), to_propagate, [matches_mov, matches_ref])
    cv2.imwrite(f'{folder_data}/{current_tile}/superglue/registered_img_x.tif', registered_imgs[0])
    cv2.imwrite(f'{folder_data}/{current_tile}/superglue/registered_img_y.tif', registered_imgs[1])
    
    registered_imgs = align_with_sitk(cv2.imread(f'{folder_data}/{current_tile}/histology/HE.png', cv2.IMREAD_GRAYSCALE), cv2.imread(f'{folder_data}/{current_tile}/histology/LFB.png', cv2.IMREAD_GRAYSCALE), to_propagate, [matches_ref, matches_mov])
    cv2.imwrite(f'{folder_data}/{current_tile}/superglue/registered_img_inv_x.tif', registered_imgs[0])
    cv2.imwrite(f'{folder_data}/{current_tile}/superglue/registered_img_inv_y.tif', registered_imgs[1])
        
    if current_tile == 'A1':
        gs_image = get_histology_img(path_mrxs, parameters)[:,500:2400]
    else:
        gs_image = get_histology_img(path_mrxs, parameters)

    img_x_gt = ((np.array(Image.open(f'{folder_data}/{current_tile}/superglue/registered_img_x.tif')) / (2**16 - 1)) * gs_image.shape[1]).astype(np.uint16)
    img_y_gt = ((np.array(Image.open(f'{folder_data}/{current_tile}/superglue/registered_img_y.tif')) / (2**16 - 1)) * gs_image.shape[0]).astype(np.uint16)
    
    gs_image_moved = np.zeros(gs_image.shape)

    for idx, x in enumerate(gs_image_moved):
        for idy, y in enumerate(x):
            try:
                gs_image_moved[idx,idy] = gs_image[img_y_gt[idx,idy], img_x_gt[idx,idy]]
            except:
                pass
                
    Image.fromarray(gs_image).convert('L').save(f'{folder_data}/{current_tile}/global_no_bg.png')
    Image.fromarray(gs_image_moved).convert('L').save(f'{folder_data}/{current_tile}/moving.png')
        
    return np.array(gs_image), np.array(gs_image_moved)


def align_with_sitk(fixed_arr, moving_arr, to_propagate, matching_points):
    fixed_image = sitk.GetImageFromArray(fixed_arr)
    
    # set up the matching points
    fixed_points = matching_points[0]
    moving_points = matching_points[1]
    fixed_landmarks = fixed_points.astype(np.uint16).flatten().tolist()
    moving_landmarks = moving_points.astype(np.uint16).flatten().tolist()
    

    # set up the bspline transform
    transform = sitk.BSplineTransformInitializer(fixed_image, (2, 2), 3)
    landmark_initializer = sitk.LandmarkBasedTransformInitializerFilter()
    landmark_initializer.SetFixedLandmarks(fixed_landmarks)
    landmark_initializer.SetMovingLandmarks(moving_landmarks)
    landmark_initializer.SetBSplineNumberOfControlPoints(10)
    landmark_initializer.SetReferenceImage(fixed_image)
    landmark_initializer.Execute(transform)
    output_transform = landmark_initializer.Execute(transform)

    # resample the moving images
    interpolator = sitk.sitkNearestNeighbor
    moving_images = [sitk.GetImageFromArray(to_propagate[0]), sitk.GetImageFromArray(to_propagate[1]),
                     sitk.GetImageFromArray(moving_arr)]
    resampled_images = []
    for moving_img in moving_images:
        resampled_images.append(sitk.GetArrayFromImage(sitk.Resample(moving_img, fixed_image, output_transform, interpolator, 0)))
                                
    return resampled_images
                               
                                             
def get_histology_img(path_mrxs, parameters):
    histology_image = HistologyImage(path_mrxs, parameters = parameters)
    rgb_image = rgba2rgb(histology_image.RGB_image)
    mask = histology_image.mask
    expanded_mask = mask[:, :, np.newaxis]
    rgb_image_mskd = np.rot90(np.where(expanded_mask, rgb_image, 0), k = -1)
    gs_image_mskd = rgb2gray(rgb_image_mskd)
    return gs_image_mskd


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def rgba2rgb( rgba, background=(255,255,255) ):
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

    return np.asarray( rgb, dtype='uint8' )