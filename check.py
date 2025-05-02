import numpy as np

# Load the npz file
file_path = 'SEED_IV_DE/DE_LDS/ses_2/1_20161125.npz'  # update this to your actual path
data = np.load(file_path, allow_pickle=True)

# List all trials
print("Trials in this file:", list(data.keys()))

# Inspect one trial
trial_key = list(data.keys())[0]  # for example, the first trial
trial_data = data[trial_key].item()  # unpack the dict

print(f"\nTrial: {trial_key}")
print("Label:", trial_data['label'])
print("Sample shape:", trial_data['sample'].shape)  # should be [W, 310]

# Optional: view part of the sample
print("\nFirst 2 time steps of sample:")
print(trial_data['sample'][:2])

