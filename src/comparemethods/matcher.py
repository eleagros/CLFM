import time
import numpy as np
import torch
import os
import pickle
import cv2

from comparemethods import match_mnn, get_features, match_oanet, helpers, plot_results


def match_sift_mnn(current_tile, type_alignment):
    folder_data = f'./{type_alignment}/data'
    folder_output = f'./{type_alignment}/output/{current_tile}'
    [sift_reference, sift_desc_ref, sift_moving, sift_desc_mov] = get_features.get_sift_features(os.path.join(folder_data, current_tile), 
                                                                                ['global_no_bg.png', 'moving.png'], folder_output, current_tile)
    
    _, matched_points_ref, matched_points_mov = match_mnn.batched_estimate_homography(torch.tensor(sift_reference), 
                torch.tensor(sift_moving), torch.tensor(sift_desc_ref), torch.tensor(sift_desc_mov))
    # matched_points_ref, matched_points_mov = ransac(matched_points_ref, matched_points_mov)
    return matched_points_ref, matched_points_mov
    
def match_superpoint_mnn(current_tile, type_alignment):
    folder_data = f'./{type_alignment}/data'
    folder_output = f'./{type_alignment}/output/{current_tile}'
    path_script = f'./../src/comparemethods/SuperPoint/find_superpoints.py'

    os.system(f"python {path_script} {os.path.join(os.path.abspath(folder_data), current_tile)} global_no_bg.png moving.png {os.path.abspath(folder_output)}")

    with open(f'{folder_output}/superpoint_points.pickle', 'rb') as handle:
        [superpoint_reference, superpoint_desc_ref, _, superpoint_moving, superpoint_desc_mov, _] = pickle.load(handle)
    _, matched_points_ref, matched_points_mov = match_mnn.batched_estimate_homography(torch.tensor(superpoint_reference), 
                torch.tensor(superpoint_moving), torch.tensor(superpoint_desc_ref), torch.tensor(superpoint_desc_mov))
    # matched_points_ref, matched_points_mov = ransac(matched_points_ref, matched_points_mov)
    return matched_points_ref, matched_points_mov

def match_silk_mnn(current_tile, type_alignment):
    folder_data = f'./{type_alignment}/data'
    folder_output = f'./{type_alignment}/output/{current_tile}'
    
    get_features.get_silk_features(folder_data, folder_output, current_tile)
    
    with open(f'{folder_output}/silk_points.pickle', 'rb') as handle:
        [silk_reference, silk_desc_ref, silk_moving, silk_desc_mov] = pickle.load(handle)
    _, matched_points_ref, matched_points_mov = match_mnn.batched_estimate_homography(torch.tensor(silk_reference), 
            torch.tensor(silk_moving), torch.tensor(silk_desc_ref), torch.tensor(silk_desc_mov))
    
    # matched_points_ref, matched_points_mov = ransac(matched_points_ref, matched_points_mov)
    return matched_points_ref, matched_points_mov

def match_sift_oanet(current_tile, type_alignment):
    folder_data = f'./{type_alignment}/data'
    folder_output = f'./{type_alignment}/output/{current_tile}'
    
    _ = get_features.get_sift_features(os.path.join(folder_data, current_tile), ['global_no_bg.png', 'moving.png'], folder_output, current_tile)
    
    matched_points_ref, matched_points_mov = match_oanet.match_oanet(folder_output, 'sift', current_tile)
    # matched_points_ref, matched_points_mov = ransac(matched_points_ref.astype(np.float32), matched_points_mov.astype(np.float32))
    return matched_points_ref, matched_points_mov

def match_superpoint_oanet(current_tile, type_alignment):
    folder_data = f'./{type_alignment}/data'
    folder_output = f'./{type_alignment}/output/{current_tile}'
    path_script = f'./../src/comparemethods/SuperPoint/find_superpoints.py'
    os.system(f"python {path_script} {os.path.join(os.path.abspath(folder_data), current_tile)} global_no_bg.png moving.png {os.path.abspath(folder_output)}")
    matched_points_ref, matched_points_mov = match_oanet.match_oanet(folder_output, 'superpoint', current_tile)
    # matched_points_ref, matched_points_mov = ransac(matched_points_ref.astype(np.float32), matched_points_mov.astype(np.float32))
    return matched_points_ref, matched_points_mov

def match_silk_oanet(current_tile, type_alignment):
    folder_data = f'./{type_alignment}/data'
    folder_output = f'./{type_alignment}/output/{current_tile}'
    
    get_features.get_silk_features(folder_data, folder_output, current_tile)
    
    matched_points_ref, matched_points_mov = match_oanet.match_oanet(folder_output, 'silk', current_tile)
    
    # matched_points_ref, matched_points_mov = ransac(matched_points_ref.astype(np.float32), matched_points_mov.astype(np.float32))
    return matched_points_ref, matched_points_mov

def match_superpoint_superglue(current_tile, type_alignment):
    folder_data = f'./{type_alignment}/data'
    folder_output = f'./{type_alignment}/output/{current_tile}'
    
    path_script = f'./../src/comparemethods/SuperPoint/find_superpoints.py'
    os.system(f"python {path_script} {os.path.join(os.path.abspath(folder_data), current_tile)} global_no_bg.png moving.png {os.path.abspath(folder_output)}")
    with open(f'{folder_output}/superpoint_points.pickle', 'rb') as handle:
        [superpoint_reference, superpoint_desc_ref, scores_ref, superpoint_moving, superpoint_desc_mov, scores_mov] = pickle.load(handle)
    
    os.makedirs(f'{folder_output}/superglue', exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dict_ref = {}
    dict_ref['keypoints0'] = [torch.tensor(superpoint_reference).to(device).type(torch.float32)]
    dict_ref['descriptors0'] = [torch.tensor(superpoint_desc_ref.T).to(device).type(torch.float32)]
    dict_ref['scores0'] = (torch.tensor(scores_ref).to(device).type(torch.float32),)

    with open(f'{folder_output}/superglue/dict_ref.pickle', 'wb') as handle:
        pickle.dump(dict_ref, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    dict_mov = {}
    dict_mov['keypoints1'] = [torch.tensor(superpoint_moving).to(device).type(torch.float32)]
    dict_mov['descriptors1'] = [torch.tensor(superpoint_desc_mov.T).to(device).type(torch.float32)]
    dict_mov['scores1'] = (torch.tensor(scores_mov).to(device).type(torch.float32),)

    with open(f'{folder_output}/superglue/dict_mov.pickle', 'wb') as handle:
        pickle.dump(dict_mov, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    path_superglue_script = f'./../src/comparemethods/superglue/demo_superglue.py'
    folder_output_superglue = os.path.abspath(os.path.join(folder_output, 'superglue'))
    cmd = f"python {path_superglue_script} --input {os.path.join(os.path.abspath(folder_data), current_tile)} --output_dir {folder_output_superglue} --no_display --resize -1"
    os.system(cmd)
    
    with open(f'{folder_output}/superglue/matches_000000_000001.pickle', 'rb') as handle:
        [matched_points_ref, matched_points_mov] = pickle.load(handle)
    
    # matched_points_ref, matched_points_mov = ransac(matched_points_ref.astype(np.float32), matched_points_mov.astype(np.float32))
    
    return [matched_points_ref, matched_points_mov]

def match_sift_superglue(current_tile, type_alignment):
    folder_data = f'./{type_alignment}/data'
    folder_output = f'./{type_alignment}/output/{current_tile}'
    
    [sift_reference, sift_desc_ref, sift_moving, sift_desc_mov] = get_features.get_sift_features(os.path.join(folder_data, current_tile), 
                                                                                ['global_no_bg.png', 'moving.png'], folder_output, current_tile)
    with open(f'{folder_output}/silk_points.pickle', 'rb') as handle:
        [sift_reference, sift_desc_ref, sift_moving, sift_desc_mov] = pickle.load(handle)
        
        
    print(sift_reference.shape, sift_desc_ref.shape, sift_moving.shape, sift_desc_mov.shape)
    
    os.makedirs(f'{folder_output}/superglue', exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dict_ref = {}
    dict_ref['keypoints0'] = [torch.tensor(sift_reference).to(device).type(torch.float32)]
    dict_ref['descriptors0'] = [torch.tensor(sift_desc_ref.T).to(device).type(torch.float32)]
    dict_ref['scores0'] = (torch.tensor(np.ones(sift_reference.shape[0])).to(device).type(torch.float32),)

    with open(f'{folder_output}/superglue/dict_ref.pickle', 'wb') as handle:
        pickle.dump(dict_ref, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    dict_mov = {}
    dict_mov['keypoints1'] = [torch.tensor(sift_moving).to(device).type(torch.float32)]
    dict_mov['descriptors1'] = [torch.tensor(sift_desc_mov.T).to(device).type(torch.float32)]
    dict_mov['scores1'] = (torch.tensor(np.ones(sift_moving.shape[0])).to(device).type(torch.float32),)

    with open(f'{folder_output}/superglue/dict_mov.pickle', 'wb') as handle:
        pickle.dump(dict_mov, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    path_superglue_script = f'./../src/comparemethods/superglue/demo_superglue.py'
    folder_output_superglue = os.path.abspath(os.path.join(folder_output, 'superglue'))
    cmd = f"python {path_superglue_script} --input {os.path.join(os.path.abspath(folder_data), current_tile)} --output_dir {folder_output_superglue} --no_display --resize -1"
    os.system(cmd)
    
    with open(f'{folder_output}/superglue/matches_000000_000001.pickle', 'rb') as handle:
        [matched_points_ref, matched_points_mov] = pickle.load(handle)
    
    # matched_points_ref, matched_points_mov = ransac(matched_points_ref.astype(np.float32), matched_points_mov.astype(np.float32))
    
    return [matched_points_ref, matched_points_mov]



def ransac(points_image1, points_image2):

    # Use RANSAC to find the homography matrix and mask of inliers
    _, mask = cv2.findHomography(points_image1, points_image2, cv2.RANSAC)
    return points_image1[mask.ravel() == 1], points_image2[mask.ravel() == 1]

def affine(points_image1, points_image2):
    # Compute affine transformation using OpenCV
    affine_matrix, _ = cv2.estimateAffine2D(points_image1, points_image2)
    transformed_points = cv2.transform(points_image1.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)
    residuals = np.linalg.norm(transformed_points - points_image2, axis=1)

    threshold = 100  # You can tune this based on your application
    return points_image1[residuals < threshold], points_image2[residuals < threshold]



def match(feature_points = 'sift', matcher = 'mnn', repetitions = 1, parameters_dict = None, images = None):
    
    if feature_points == 'sift' and matcher == 'mnn':
        fun = match_sift_mnn
    
    if feature_points == 'superpoint' and matcher == 'mnn':
        fun = match_superpoint_mnn
        
    if feature_points == 'silk' and matcher == 'mnn':
        fun = match_silk_mnn
        
    if feature_points == 'sift' and matcher == 'oanet':
        fun = match_sift_oanet
        
    if feature_points == 'superpoint' and matcher == 'oanet':
        fun = match_superpoint_oanet
        
    if feature_points == 'superpoint' and matcher == 'superglue':
        fun = match_superpoint_superglue
        
    if feature_points == 'silk' and matcher == 'oanet':
        fun = match_silk_oanet
        
    if feature_points == 'sift' and matcher == 'superglue':
        fun = match_sift_superglue
        
    times = []
    results = {}
    
    for key, value in parameters_dict.items():
        current_tile = key
        type_alignment = value['type_alignment']
        
        for i in range(repetitions):
            start_time = time.time()
            results[key] = fun(current_tile, type_alignment)
            end_time = time.time()
            times.append(end_time - start_time)
            
        helpers.write_coordinates_txt(f"{value['path_coordinates']}/{feature_points}_{matcher}.txt", 
                                        [results[key][0], results[key][1]])
        
        plot_results.plot_results(images[key]['reference'], images[key]['moving'], 
                          results[key][0], results[key][1], f"{value['path_plot']}/{feature_points}_{matcher}.pdf")
        
    mean_time = np.mean(times)
    std_time = np.std(times)
    print('Processing time: {:.2f} ± {:.2f} s'.format(mean_time,std_time))
    
    return results