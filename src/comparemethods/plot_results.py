import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
np.random.seed(42)

def plot_results(reference_image, moving_image, matched_points_ref, matched_points_mov_ori, save_path, distance = 50, distances = None, histology = False):
    plt.figure(figsize = (20,20))
    
    matched_points_mov = copy.deepcopy(matched_points_mov_ori)
    
    if histology:
        match_pts_numnber = 100
    else:
        match_pts_numnber = 50
        
    if len(matched_points_ref) > match_pts_numnber:
        random_indices = np.random.choice(matched_points_ref.shape[0], size=match_pts_numnber, replace=False)
        matched_points_ref = matched_points_ref[random_indices]
        matched_points_mov = matched_points_mov[random_indices]
        if distances is not None:
            distances_mov = np.array(distances)[random_indices]
    else:
        distances_mov = distances
    
    if distances is not None:
        rgb_cycle = []
        for dist in distances_mov:
            if histology:
                dist_thresh = [5, 10]
            else:
                dist_thresh = [30, 100]
                
            if dist < dist_thresh[0]:
                rgb_cycle.append([0,1,0])
            elif dist < dist_thresh[1]:
                rgb_cycle.append([0,0,1])
            else:
                rgb_cycle.append([1,0,0])
    else:
        x = len(matched_points_ref)
        phi = np.linspace(0, 2*np.pi, x)
        x = np.sin(phi)
        y = np.cos(phi)
        rgb_cycle = (np.stack((np.cos(phi), np.cos(phi+2*np.pi/3), np.cos(phi-2*np.pi/3))).T + 1)*0.5
    
    if len(matched_points_mov) > 0:
        matched_points_mov[:,0] = matched_points_mov[:,0] + distance + reference_image.shape[1]

    array_plot = np.ones((reference_image.shape[0], distance + reference_image.shape[1] + moving_image.shape[1])) * 255

    array_plot[:,:reference_image.shape[1]] = reference_image
    array_plot[:,distance + reference_image.shape[1]:] = moving_image

    if len(matched_points_mov) > 0:
        plt.scatter(*(matched_points_ref).T, s = 70, c = rgb_cycle)
        plt.scatter(*(matched_points_mov).T, s = 70, c = rgb_cycle)

        for mp, fp, rgb in zip(matched_points_ref, matched_points_mov, rgb_cycle):
            plt.plot([mp[0], fp[0]], [mp[1], fp[1]], color=rgb, linewidth=1)

    plt.imshow(array_plot, cmap = 'gray')
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    
    if distances is not None:
        plt.close()
    