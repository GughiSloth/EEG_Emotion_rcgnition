import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import f_classif

# === Load data ===
data = np.load("wavelet_morlet_features_SEED_IV_combined.npz")
X = data['features']
y = data['labels']

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# === Preprocessing ===
qt = QuantileTransformer(output_distribution='normal', random_state=42)
scaler = StandardScaler()
X_train_qt = qt.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_qt)
X_test_qt = qt.transform(X_test)
X_test_scaled = scaler.transform(X_test_qt)

# === Plot Function ===
def plot_lda_2d(X_lda, y, title):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'orange', 'purple']
    class_names = ['Neutral  (0)','Sad (1)', 'Fear (2)', 'Happy (3)']
    classes = np.unique(y)
    for i, cls in enumerate(classes):
        plt.scatter(X_lda[y == cls, 0], X_lda[y == cls, 1],
                    label=class_names[int(cls)], alpha=0.7, color=colors[i % len(colors)])
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Original LDA Projection ===
lda = LDA(n_components=2, solver='eigen', shrinkage=0.005)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)
plot_lda_2d(X_train_lda, y_train, "Original LDA Projection (Train Set)")
plot_lda_2d(X_test_lda, y_test, "Original LDA Projection (Test Set)")

# === Compact class 3 and separate it from class 1 and 2 ===
def compact_and_separate(X_scaled, y, target_class, opposing_classes, boost=1.5, reduce=0.5, topk=15):
    X_new = X_scaled.copy()

    mask_target = y == target_class
    mask_opp = np.isin(y, opposing_classes)

    X_pair = np.vstack([X_scaled[mask_target], X_scaled[mask_opp]])
    y_pair = np.concatenate([
        np.zeros(np.sum(mask_target)),
        np.ones(np.sum(mask_opp))
    ])

    F, _ = f_classif(X_pair, y_pair)
    top_feats = np.argsort(F)[-topk:]

    X_new[mask_target][:, top_feats] *= boost
    for opp_class in opposing_classes:
        X_new[y == opp_class][:, top_feats] *= reduce

    return X_new, top_feats

# === Apply transformation ===
X_train_modified, important_features = compact_and_separate(
    X_train_scaled, y_train, target_class=2, opposing_classes=[0, 1], boost=1.75, reduce=0.75, topk=20)
X_test_modified, _ = compact_and_separate(
    X_test_scaled, y_test, target_class=2, opposing_classes=[0, 1], boost=1.75, reduce=0.75, topk=20)

# === LDA after transformation ===
lda2 = LDA(n_components=2, solver='eigen', shrinkage=0.005)
X_train_lda2 = lda2.fit_transform(X_train_modified, y_train)
X_test_lda2 = lda2.transform(X_test_modified)

plot_lda_2d(X_train_lda2, y_train, "LDA Projection After Feature Engineering (Train Set)")
plot_lda_2d(X_test_lda2, y_test, "LDA Projection After Feature Engineering (Test Set)")