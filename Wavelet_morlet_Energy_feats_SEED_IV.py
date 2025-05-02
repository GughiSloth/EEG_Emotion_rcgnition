import numpy as np
import os
import pywt
import re
from tqdm import tqdm

# === Parameters ===
base_folder = "SEED_IV_DE/DE_ma"
wavelet = 'cmor1.5-1.0'
scales = np.arange(1, 21)

# === Storage ===
meta_info = []  # (partitioner, session, trial, features, label)

# === Helpers ===
def extract_numbers(text):
    return tuple(map(int, re.findall(r'\d+', text)))  # returns tuple of all numbers in string

# === Loop through all ses_* folders ===
for session_folder in sorted(os.listdir(base_folder)):
    session_path = os.path.join(base_folder, session_folder)
    if not os.path.isdir(session_path):
        continue

    session_id = extract_numbers(session_folder)[0]  # ses_3 → 3
    print(f"📁 Processing session {session_id}")

    for filename in sorted(os.listdir(session_path)):
        if not filename.endswith(".npz"):
            continue

        partitioner_id = extract_numbers(filename)[0]  # part_2.npz → 2
        file_path = os.path.join(session_path, filename)
        data = np.load(file_path, allow_pickle=True)

        for trial_key in sorted(data.keys()):
            trial_id = extract_numbers(trial_key)[0]  # trial3 → 3
            trial = data[trial_key].item()
            sample = trial["sample"]  # shape: (W, 310)
            label = trial["label"]

            # === Wavelet feature extraction ===
            feats = []
            for i in range(sample.shape[1]):
                signal = sample[:, i]
                try:
                    coefs, _ = pywt.cwt(signal, scales, wavelet)
                    power = np.abs(coefs) ** 2
                    energy_per_scale = np.sum(power, axis=1)
                    feats.extend(np.log1p(energy_per_scale))
                except Exception:
                    feats.extend([0.0] * len(scales))

            meta_info.append((partitioner_id, session_id, trial_id, feats, label))

# === Sort by (partitioner, session, trial) ===
meta_info.sort(key=lambda x: (x[0], x[1], x[2]))

# === Unpack features and labels ===
features = np.array([entry[3] for entry in meta_info])
labels = np.array([entry[4] for entry in meta_info])

# === Save to .npz ===
output_path = "wavelet_morlet_features_SEED_IV_ma.npz"
np.savez(output_path, features=features, labels=labels)
print(f"\n✅ Saved ordered dataset: {output_path}")
print(f"   → features shape: {features.shape}")
print(f"   → labels shape: {labels.shape}")