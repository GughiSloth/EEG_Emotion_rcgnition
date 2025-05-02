import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import f_classif
from sklearn.metrics import pairwise_distances, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KernelDensity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Add, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import gelu
import tensorflow as tf
import os

# === Load data ===
data = np.load("wavelet_morlet_features_SEED_V_combined.npz")
X = data['features']
y = data['labels']

# === Train-test split ===
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Preprocessing ===
qt = QuantileTransformer(output_distribution='normal', random_state=42)
scaler = StandardScaler()
X_trainval_qt = qt.fit_transform(X_trainval)
X_trainval_scaled = scaler.fit_transform(X_trainval_qt)
X_test_qt = qt.transform(X_test)
X_test_scaled = scaler.transform(X_test_qt)

# === LDA for confusion ===
lda_temp = LDA(n_components=2, solver='eigen', shrinkage=0.005)
X_trainval_lda2 = lda_temp.fit_transform(X_trainval_scaled, y_trainval)
centroids = [X_trainval_lda2[y_trainval == c].mean(axis=0) for c in np.unique(y_trainval)]
dists = pairwise_distances(centroids)
np.fill_diagonal(dists, np.inf)
pairs = []
for _ in range(2):
    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    pairs.append((i, j))
    dists[i, j] = dists[j, i] = np.inf
print(f"Most confused class pairs: {pairs}")

# === Density separation control ===
kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
kde.fit(X_trainval_lda2)
densities = kde.score_samples(X_trainval_lda2)
low_density_threshold = np.percentile(densities, 30)  # Lower 30% most sparse
boost_mask = densities <= low_density_threshold

# === Feature selection ===
all_top_feats = set()
for i, j in pairs:
    mask = (y_trainval == i) | (y_trainval == j)
    F, _ = f_classif(X_trainval_scaled[mask], y_trainval[mask])
    top_k = np.argsort(F)[-20:]
    all_top_feats.update(top_k.tolist())
top_k_union = sorted(all_top_feats)

# === Apply density-based boost ===
X_trainval_filtered = X_trainval_scaled.copy()
X_test_filtered = X_test_scaled.copy()
X_trainval_filtered[boost_mask, :][:, top_k_union] *= 1.75
X_trainval_filtered[~boost_mask, :][:, top_k_union] *= 0.75

# === Start CV ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_val_acc = 0
best_model_path = ""
best_lda = None
best_svd = None
fold = 1

for train_index, val_index in kf.split(X_trainval_filtered, y_trainval):
    print(f"\n📂 Fold {fold}")
    X_train, X_val = X_trainval_filtered[train_index], X_trainval_filtered[val_index]
    y_train, y_val = y_trainval[train_index], y_trainval[val_index]

    lda = LDA(n_components=4, solver='eigen', shrinkage=0.005)
    svd = TruncatedSVD(n_components=4)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_val_lda = lda.transform(X_val)
    X_test_lda = lda.transform(X_test_filtered)
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd = svd.transform(X_val)
    X_test_svd = svd.transform(X_test_filtered)

    X_train_combined = np.concatenate([X_train_lda, X_train_svd], axis=1)
    X_val_combined = np.concatenate([X_val_lda, X_val_svd], axis=1)
    X_test_combined = np.concatenate([X_test_lda, X_test_svd], axis=1)

    input_layer = Input(shape=(X_train_combined.shape[1],))
    x = Dense(128, activation=gelu, kernel_regularizer=l2(1e-5))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    res = Dense(128, kernel_regularizer=l2(1e-5))(x)
    x = Dense(128, kernel_regularizer=l2(1e-5))(x)
    x = Add()([x, res])
    x = tf.keras.layers.Activation(gelu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    attention_weights = Dense(x.shape[-1], activation='sigmoid')(x)
    x = Multiply()([x, attention_weights])
    x = LayerNormalization()(x)
    x = Dense(64, activation=gelu, kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation=gelu)(x)
    x = Dropout(0.2)(x)
    out = Dense(5, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True),
        ModelCheckpoint(f"fold_{fold}_model_V.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)
    ]

    history = model.fit(
        X_train_combined, y_train,
        validation_data=(X_val_combined, y_val),
        epochs=500,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"✅ Fold {fold} best val accuracy: {val_acc:.4f}")
    model.save(f"fold_{fold}_model_V.h5")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = f"fold_{fold}_model_V.h5"
        best_lda = lda
        best_svd = svd

    fold += 1

print("\n🧠 Evaluating on test set with model ensemble...")
X_test_combined = np.concatenate([best_lda.transform(X_test_filtered), best_svd.transform(X_test_filtered)], axis=1)
models = []
for i in range(1, 6):
    model_path = f"fold_{i}_model_V.h5"
    if os.path.exists(model_path):
        models.append(tf.keras.models.load_model(model_path))
    else:
        print(f"⚠️ Warning: {model_path} not found.")

print(f"🔍 Averaging predictions from {len(models)} models...")
y_pred_probs_all = np.mean([m.predict(X_test_combined) for m in models], axis=0)
y_pred = np.argmax(y_pred_probs_all, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()
