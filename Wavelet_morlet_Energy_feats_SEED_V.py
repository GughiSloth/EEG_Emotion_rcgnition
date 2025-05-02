import numpy as np
import os
import pywt
import pickle as pkl
from scipy.signal import lfilter

# === Parameters ===
folder_path = "DE_Data"  # Path to original npz files
wavelet = 'cmor1.5-1.0'
scales = np.arange(1, 21)  # 20 scale levels for CWT
window_size = 5            # For moving average
lds_order = 2              # Order of LDS

# === LDS filter (autoregressive model of order 2) ===
def apply_lds(signal, order=2):
    # Simple autoregressive LDS: y[t] = a1*y[t-1] + a2*y[t-2] + ...
    # We'll just use lfilter with preset coeffs (can be tuned)
    a = [1] + [0.25] * order  # example coefficients
    b = [1.0]
    return lfilter(b, a, signal)

# === Moving average smoothing ===
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# === Load all DE samples and labels ===
all_samples = []
labels = []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".npz"):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path, allow_pickle=True)

        if "data" in data:
            sample = pkl.loads(data["data"])
            label = pkl.loads(data["label"])
            for key in sorted(sample.keys()):
                all_samples.append(sample[key])  # shape: (time, 310)
                labels.append(label[key][0])

labels = np.array(labels)

# === Feature extraction wrapper ===
def extract_features_from_samples(samples, smoothing_method):
    all_feats = []

    for sample in samples:
        feats = []
        for i in range(sample.shape[1]):
            signal = sample[:, i]

            # Apply selected smoothing method
            if smoothing_method == 'lds':
                smoothed = apply_lds(signal, order=lds_order)
            elif smoothing_method == 'ma':
                smoothed = moving_average(signal, window_size=window_size)
            else:
                raise ValueError("Invalid smoothing method")

            try:
                coefs, _ = pywt.cwt(smoothed, scales, wavelet)
                power = np.abs(coefs) ** 2
                energy_per_scale = np.sum(power, axis=1)
                feats.extend(np.log1p(energy_per_scale))
            except Exception:
                feats.extend([0.0] * len(scales))  # Fallback
        all_feats.append(feats)

    return np.array(all_feats)

# === Extract and save LDS-smoothed version ===
features_lds = extract_features_from_samples(all_samples, smoothing_method='lds')
np.savez("wavelet_morlet_features_SEEDV_LDS.npz", features=features_lds, labels=labels)
print(f"✅ Saved: wavelet_morlet_features_SEEDV_LDS.npz → {features_lds.shape}")

# === Extract and save MA-smoothed version ===
features_ma = extract_features_from_samples(all_samples, smoothing_method='ma')
np.savez("wavelet_morlet_features_SEEDV_MA.npz", features=features_ma, labels=labels)
print(f"✅ Saved: wavelet_morlet_features_SEEDV_MA.npz → {features_ma.shape}")