import os
import pickle

def match_oanet(output_file, type_feature_points, current_tile):
    path_script_rep = os.path.join(os.path.realpath(__file__).split('match_oanet.py')[0], 'OANet-master/demo').replace('\\', '/') 
    path_script = os.path.join(path_script_rep, 'infer_oa_net.py').replace('\\', '/') 
    os.system(f'cd {path_script_rep} && python {path_script} {os.path.abspath(output_file)} {type_feature_points} {current_tile}')
    
    with open(os.path.join( os.path.abspath(output_file), 'matches_' + type_feature_points + '_oanet.pickle'), 'rb') as handle:
        matches = pickle.load(handle)
        
    matched_points_ref = matches[0][:,:2]
    matched_points_mov = matches[1][:,:2]
    return matched_points_ref, matched_points_mov