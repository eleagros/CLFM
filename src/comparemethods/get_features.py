import os
import pickle
import numpy as np
np.random.seed(42)
import cv2
import matplotlib.pyplot as plt
import shutil
import subprocess
    
def get_sift_features(folder_path, path_imgs, output_file, current_tile):
        
    path_script_rep = os.path.join(os.path.realpath(__file__).split('get_features.py')[0], 'OANet-master/demo').replace('\\', '/') 
    path_script = os.path.join(path_script_rep, 'get_sift_points.py').replace('\\', '/') 
    os.system(f'cd {path_script_rep} && python {path_script} ' + os.path.join(os.path.abspath(folder_path), path_imgs[0]) + ' ' + os.path.join(os.path.abspath(folder_path), path_imgs[1]) + ' ' + os.path.abspath(output_file))
    
    with open(os.path.join(output_file, 'sift_points.pickle'), 'rb') as handle:
        [sift_reference, sift_desc_ref, sift_moving, sift_desc_mov] = pickle.load(handle)
        
    match_pts_numnber = len(sift_reference) - 1
    if len(sift_reference) > match_pts_numnber:
        random_indices = np.random.choice(sift_reference.shape[0], size=match_pts_numnber, replace=False)
        sift_reference = sift_reference[random_indices]
        sift_desc_ref = sift_desc_ref[random_indices]
        
    match_pts_numnber = len(sift_moving) - 1
    
    if len(sift_moving) > match_pts_numnber:
        random_indices = np.random.choice(sift_moving.shape[0], size=match_pts_numnber, replace=False)
        sift_moving = sift_moving[random_indices]
        sift_desc_mov = sift_desc_mov[random_indices]
    
    return [sift_reference[:,:2], sift_desc_ref, sift_moving[:,:2], sift_desc_mov]


def get_silk_features(folder_data, folder_output, current_tile):
    
    src = f"{folder_data}/{current_tile}/global_no_bg.png"
    dst = r"\\wsl$\Ubuntu\home\romane\silk-main\scripts\examples\global_no_bg.png"
    shutil.copy(src, dst)
    src = f"{folder_data}/{current_tile}/moving.png"
    dst = r"\\wsl$\Ubuntu\home\romane\silk-main\scripts\examples\moving.png"
    shutil.copy(src, dst)
    
    wsl_directory = "/home/romane/silk-main/scripts/examples"
    script_name = "get_silk_points.py"
    conda_env_name = "silk"
    conda_init_path = "/home/romane/ENTER/etc/profile.d/conda.sh"  # Adjust this path if necessary

    # Function to run the Python script after activating Conda environment
    def run_wsl_python_script_with_conda():
        try:
            
            # Command to activate Conda and run the script
            command = f"source {conda_init_path} && conda activate {conda_env_name} && cd {wsl_directory} && python {script_name}"
            
            # Execute the command in WSL using bash
            result = subprocess.run(["wsl", "bash", "-c", command], capture_output=True, text=True)
            
            # Return the result if needed
            return result.stdout, result.stderr
        except Exception as e:
            print(f"An error occurred: {e}")

    # Run the script with Conda environment activated
    run_wsl_python_script_with_conda()
    
    if os.path.exists(f'{folder_output}/silk_points.pickle'):
        os.remove(f'{folder_output}/silk_points.pickle')
    src = r"\\wsl$\Ubuntu\home\romane\silk-main\scripts\examples\silk_points.pickle"
    dst = f"{folder_output}/silk_points.pickle"
    shutil.copy(src, dst)