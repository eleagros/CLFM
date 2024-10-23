import os
import numpy as np
import cv2
from FyeldGenerator import generate_field
import random
random.seed(42)
import scipy.ndimage as ndi
import pickle
import scipy.io as sio
from PIL import Image


def get_parameters(type_alignment, current_tiles):
    
    # histology parameters
    parameters = {'level': 6, 'histo_histo': True, 'border_size': 600}
    
    parameters_dict = {}
    
    for current_tile in current_tiles:
        parameters_current_tile = {}
        
        parameters_current_tile['type_alignment'] = type_alignment
        parameters_current_tile['parameters'] = parameters
        parameters_current_tile['path_mrxs'] = f'D:/Chicago/HT_sample/{type_alignment}/data/{current_tile}_{type_alignment}.mrxs'
        
        
        parameters_current_tile['folder_data'] = f'./{type_alignment}/data'
        parameters_current_tile['folder_output'] = f'./{type_alignment}/output'
        
        os.makedirs(parameters_current_tile['folder_output'], exist_ok = True)
        os.makedirs(f'{parameters_current_tile["folder_output"]}/{current_tile}', exist_ok = True)
        os.makedirs(f'{parameters_current_tile["folder_data"]}/{current_tile}', exist_ok = True)
        
        parameters_current_tile['path_coordinates'] = f'./{type_alignment}/results/{current_tile}/coordinates'
        parameters_current_tile['path_mapped'] = f'./{type_alignment}/results/{current_tile}/mapped'
        parameters_current_tile['path_plot'] = f'./{type_alignment}/results/{current_tile}/plots'
        parameters_current_tile['path_distances'] = f'./{type_alignment}/results/{current_tile}/distances'
        
        os.makedirs(f'./{type_alignment}/results/{current_tile}', exist_ok = True)
        os.makedirs(parameters_current_tile['path_coordinates'], exist_ok = True)
        os.makedirs(f'./{type_alignment}/results/{current_tile}/plots', exist_ok = True)
        os.makedirs(f'./{type_alignment}/results/{current_tile}/distances', exist_ok = True)
        
        parameters_dict[current_tile] = parameters_current_tile
    
    return parameters_dict
        
    
    
def write_coordinates_txt(path_save, mp_fp):
    with open(path_save, 'w') as f:
        f.write(write_mp_fp_txt_format(mp_fp))
    
def write_mp_fp_txt_format(mp_fp):
    """
    formats matched points for ImageJ alignment in a tab-separated text format.
    """
    text = 'Index\txSource\tySource\txTarget\tyTarget\n'
    text += ''.join(f"{i}\t{round(src[0])}\t{round(src[1])}\t{round(tgt[0])}\t{round(tgt[1])}\n"
                    for i, (src, tgt) in enumerate(zip(mp_fp[0], mp_fp[1])) if np.sum(tgt) != 0)
    return text
    

def rotate_array_with_opencv(arr, angle):
    # Get the center of the array
    center = (arr.shape[1] // 2, arr.shape[0] // 2)
    
    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated_arr = cv2.warpAffine(arr, rotation_matrix, (arr.shape[1], arr.shape[0]), flags=cv2.INTER_NEAREST)
    
    return rotated_arr


def displace(image, image_mskd, parameters = None):
    border_size = 600
    
    border_displaced = np.zeros(image.shape)
    
    if parameters is None:
        parameters = {}
        displacement_x = random.randint(-border_size, border_size)
        displacement_y = random.randint(-border_size, border_size)
        angle = random.uniform(-5, 5)

        parameters['displacement_x'] = displacement_x
        parameters['displacement_y'] = displacement_y
        parameters['angle'] = angle
    
    border_displaced[border_size + parameters['displacement_x']: border_displaced.shape[0] - border_size + parameters['displacement_x'], 
                    border_size + parameters['displacement_y']: border_displaced.shape[1] - border_size + parameters['displacement_y']] = image_mskd
        
    border_rotated = rotate_array_with_opencv(border_displaced, parameters['angle'])

    # Apply elastic transformation
    alpha = 10  # Scaling factor for deformation
    sigma = 10   # Standard deviation of the Gaussian filter
    if 'maps_deformation' in parameters:
        border_deformed, maps_deformation = elastic_transform(border_rotated, alpha, sigma, indices = parameters['maps_deformation'])
        return border_deformed, border_rotated
    else:
        border_deformed, maps_deformation = elastic_transform(border_rotated, alpha, sigma)
        parameters['maps_deformation'] = maps_deformation

    return border_deformed, parameters




def deform_image(img):
    
    # Helper that generates power-law power spectrum
    def Pkgen(n):
        def Pk(k):
            return np.power(k, -n)
        return Pk

    # Draw samples from a normal distribution
    def distrib(shape):
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return a + 1j * b

    shape = img.shape

    field_x = generate_field(distrib, Pkgen(3), shape)
    field_y = generate_field(distrib, Pkgen(3), shape)

    mapx_base, mapy_base = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    mapx = mapx_base + field_x*100
    mapy = mapy_base + field_y*100  
    
    return apply_maps(img, mapx, mapy), [mapx, mapy]

def apply_maps(img, mapx, mapy):
    return cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), interpolation = cv2.INTER_NEAREST)


def elastic_transform(image, alpha, sigma, random_state=None, indices=None):
    """
    Apply random elastic deformation to the input image.
    
    Parameters:
        image: Input 2D or 3D image (e.g., RGB, RGBA)
        alpha: Scaling factor for the intensity of deformation (larger = more intense deformation)
        sigma: Standard deviation of the Gaussian kernel (controls smoothness of deformation)
        random_state: Optional; random seed for reproducibility
        
    Returns:
        Transformed image with elastic deformation applied.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    if len(shape) == 2:  # Grayscale image
        h, w = shape
    elif len(shape) == 3:  # RGB/RGBA image
        h, w, c = shape
    
    # Generate random displacement fields for x and y directions
    dx = random_state.rand(h, w) * 2 - 1  # Random values between -1 and 1
    dy = random_state.rand(h, w) * 2 - 1
    
    # Smooth displacement fields using a Gaussian filter
    dx = ndi.gaussian_filter(dx, sigma, mode="constant", cval=0) * alpha
    dy = ndi.gaussian_filter(dy, sigma, mode="constant", cval=0) * alpha
    
    # Create a meshgrid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply the displacements to the meshgrid
    if indices is None:
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    print(indices[0].shape)
    
    # Map the image pixels according to the displacement
    if len(shape) == 2:
        # For grayscale images
        distorted_image = ndi.map_coordinates(image, indices, order=3, mode='nearest').reshape((h, w))
        print(np.where(distorted_image.astype(np.uint64) == 41))
    else:
        # For RGB/RGBA images, apply transformation to each channel
        distorted_image = np.zeros_like(image)
        for i in range(c):
            distorted_image[..., i] = ndi.map_coordinates(image[..., i], indices, order=1, mode='reflect').reshape((h, w))
    
    return distorted_image, indices


def get_mp_fp(path_coordinates) -> dict:
    """
    retrieves matched points from a .mat or text file into a Python dictionary format.
    """
    if path_coordinates.endswith('.mat'):
        mp_fp = sio.loadmat(path_coordinates)
        return {'mp': mp_fp['mp'], 'fp': mp_fp['fp']}
    
    with open(path_coordinates) as f:
        lines = [line.rstrip('\n') for line in f]
    
    mp = []
    fp = []
    for line in lines[1:]:
        splitted = line.split('\t')
        fp.append([int(splitted[1]), int(splitted[2])])
        mp.append([int(splitted[3]), int(splitted[4])])
    return {'mp': np.array(mp), 'fp': np.array(fp)}