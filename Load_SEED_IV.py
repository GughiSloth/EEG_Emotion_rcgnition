import os
import numpy as np
import scipy.io as sio
import pandas as pd
from tqdm import tqdm

# Input and output directories
source_dir = 'DE_Data_SEED_IV/3'
output_dirs = {
    'LDS': 'SEED_IV_DE/DE_LDS/ses_3',
    'movingAve': 'SEED_IV_DE/DE_ma/ses_3'
}

# Ensure output folders exist
for out_dir in output_dirs.values():
    os.makedirs(out_dir, exist_ok=True)

# Load labels from Labels.csv (assumes one label per trial, in order)
labels_path = os.path.join(source_dir, 'Labels.csv')
labels = pd.read_csv(labels_path, header=None).squeeze().tolist()  # List of labels

# Loop through all .mat files in the source directory
for file_name in tqdm(os.listdir(source_dir)):
    if not file_name.endswith('.mat'):
        continue

    mat_path = os.path.join(source_dir, file_name)
    mat = sio.loadmat(mat_path)

    # Process both LDS and moving average smoothing types
    for smooth_type, out_folder in output_dirs.items():
        trial_dict = {}

        for key in mat:
            if key.startswith(f'de_{smooth_type}'):
                trial_id = key.replace(f'de_{smooth_type}', '')
                trial_index = int(trial_id)  # assuming trial1, trial2, etc.

                arr = mat[key]  # shape [62, W, 5]
                arr = np.transpose(arr, (1, 0, 2)).reshape(arr.shape[1], -1)  # -> [W, 310]

                # Store both label and sample
                trial_dict[f"trial{trial_index}"] = {
                    "label": labels[trial_index-1],     # assumes 0-based indexing
                    "sample": arr
                }

        # Save to .npz
        save_name = file_name.replace('.mat', '.npz')
        save_path = os.path.join(out_folder, save_name)
        np.savez(save_path, **trial_dict)