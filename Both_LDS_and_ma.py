import numpy as np

# Load both datasets
lds_data = np.load("wavelet_morlet_features_SEEDV_LDS.npz")
ma_data = np.load("wavelet_morlet_features_SEEDV_MA.npz")

X_lds = lds_data['features']
X_ma = ma_data['features']
y_lds = lds_data['labels']
y_ma = ma_data['labels']

# === Sanity check: Ensure label alignment
assert np.array_equal(y_lds, y_ma), "Mismatch in labels between LDS and MA datasets!"

# === Concatenate features
X_combined = np.concatenate([X_lds, X_ma], axis=1)
y_combined = y_lds  # Same as y_ma

print("✅ Combined shape:", X_combined.shape)
print("🧾 Number of labels:", y_combined.shape[0])

# === Save to new .npz
np.savez("wavelet_morlet_features_SEED_V_combined.npz", features=X_combined, labels=y_combined)
print("📦 Saved: wavelet_morlet_features_SEED_V_combined.npz")