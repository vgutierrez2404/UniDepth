import os
import numpy as np
import cv2

from typing import Dict, List, Any

def save_to_npy(calibration_dictionary: Dict[str, Any], items_to_save: List[str], directory_path: str) -> None:
    
    saving_directory = os.path.join(directory_path, 'calibration_variables/')
    os.makedirs(saving_directory, exist_ok=True)

    for item in items_to_save:
        np.save(os.path.join(saving_directory, f'{item}.npy'), calibration_dictionary[item])
        print(f'Saving {item} into {saving_directory} directory') 

if __name__ == "__main__":

    dataset_folder = "/home/gaps-canteras-u22/Documents/repos/UniDepth/data"

    ### Adaptation from the original AkuDataLoader ###
    calibration_dictionary = {}
    items_to_save = ['Intrinsic_L', 'Intrinsic_R']

    subdirectories = [ folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder)) ]
    for folder in subdirectories: 
        directory_path = os.path.join(dataset_folder, folder)
        files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
        yaml_path = os.path.join(directory_path, 'AK_stereo_calibration.yaml')
        if os.path.isfile(yaml_path):
            fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
            calibration_dictionary[folder] = {
                'Intrinsic_L': np.array(fs.getNode('Intrinsic_L').mat()).astype(np.float32),
                'Intrinsic_R': np.array(fs.getNode('Intrinsic_R').mat()).astype(np.float32),
                'R': np.array(fs.getNode('Extrinsic_R').mat()).astype(np.float32),
                't': np.array(fs.getNode('Extrinsic_t').mat()).astype(np.float32),
                'B': fs.getNode('B').real(),
                'f': fs.getNode('f').real(),
            }
            fs.release()
    

        save_to_npy(calibration_dictionary[folder], items_to_save, directory_path)
