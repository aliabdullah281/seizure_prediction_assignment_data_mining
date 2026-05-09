"""
============================================================
SEMESTER MAJOR ASSIGNMENT -- Advance Topics in Data Mining
Topic  : Seizure Prediction (EEG Classification)
============================================================
"""


# 0. IMPORTS & GLOBAL SETTINGS

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend - saves files without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import os, warnings, json
warnings.filterwarnings("ignore")

from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis, zscore
from scipy.fft import fft, fftfreq

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     learning_curve, StratifiedKFold,
                                     validation_curve)
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              average_precision_score,
                              precision_recall_curve, roc_curve)
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 100, "font.size": 11})

BASE   = Path(r"D:\Imsciences\Semester_ii\Data Minning\Assignment\data-mining-assignment-3")
DATA   = BASE / "dataset"
FIGS   = BASE / "results"
FIGS.mkdir(exist_ok=True)

FS = 173.6          # dataset sampling frequency (Hz)
N  = 512            # samples per segment

print("[OK] Imports and global settings done.")



# 1. DATA LOADING

print("\n" + "="*60)
print("SECTION 1 -- DATA LOADING")
print("="*60)

CLASS_MAP = {"Z": 0, "O": 1, "N": 2, "F": 3, "S": 4}
FOLDER    = {c: f"{c}_filtered_CWT" for c in CLASS_MAP}
CLASS_DESC = {
    "Z": "Healthy - eyes open",
    "O": "Healthy - eyes closed",
    "N": "Epileptic - seizure-free (non-focal)",
    "F": "Epileptic - seizure-free (focal)",
    "S": "Seizure (ictal)"
}

def load_class(data_dir: Path, folder: str, label: int):
    files = sorted(p for p in (data_dir / folder).iterdir() if p.suffix == ".txt")
    arr   = np.array([np.loadtxt(f) for f in files])
    return arr, np.full(len(arr), label, dtype=int)

X_parts, y_parts = [], []
for cls, lbl in CLASS_MAP.items():
    X_c, y_c = load_class(DATA, FOLDER[cls], lbl)
    X_parts.append(X_c); y_parts.append(y_c)
    print(f"  {cls} ({CLASS_DESC[cls]}): {len(X_c)} samples x {X_c.shape[1]} features")

X_full = np.vstack(X_parts)
y_full = np.concatenate(y_parts)
print(f"\n  >> Total: {X_full.shape[0]} samples x {X_full.shape[1]} features")

# D1: Binary - Healthy-open vs Seizure  (balanced 1:1)
mask_d1 = np.isin(y_full, [0, 4])
X_d1 = X_full[mask_d1]; y_d1 = (y_full[mask_d1] == 4).astype(int)

# D2: Binary - (N+F epileptic non-ictal) vs Seizure (imbalanced 2:1)
mask_d2 = np.isin(y_full, [2, 3, 4])
X_d2 = X_full[mask_d2]; y_d2 = (y_full[mask_d2] == 4).astype(int)

# D3: All vs Seizure - Full binary (highly imbalanced 4:1)
X_d3 = X_full.copy(); y_d3 = (y_full == 4).astype(int)

DATASETS = {
    "D1 - Healthy vs Seizure (1:1)":          (X_d1, y_d1),
    "D2 - Epileptic non-ictal vs Seizure (2:1)": (X_d2, y_d2),
    "D3 - All classes vs Seizure (4:1)":       (X_d3, y_d3),
}
print("\nDataset summary:")
for name, (X, y) in DATASETS.items():
    pos = y.sum(); neg = len(y) - pos
    print(f"  {name}  ->  {len(y)} samples  | 0:{neg}  1:{pos}  ratio {neg/pos:.1f}:1")



# 2. EXPLORATORY DATA ANALYSIS

print("\n" + "="*60)
print("SECTION 2 -- EXPLORATORY DATA ANALYSIS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
cls_keys = list(CLASS_MAP.keys())
colors   = sns.color_palette("tab10", 5)
t        = np.arange(N) / FS * 1000   # ms

for i, (cls, ax) in enumerate(zip(cls_keys, axes.flat)):
    sample = X_parts[i][0]
    ax.plot(t, sample, color=colors[i], lw=0.9)
    ax.set_title(f"Class {cls} -- {CLASS_DESC[cls]}", fontsize=10)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude (muV)")
axes.flat[-1].axis("off")
plt.suptitle("Figure 1 -- Representative EEG Segments (one per class)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIGS / "fig01_eda_signals.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig01_eda_signals.png")

# Class distribution bar
fig, ax = plt.subplots(figsize=(8, 4))
cls_names = [f"{c}\n({CLASS_DESC[c][:18]})" for c in cls_keys]
counts    = [len(X_parts[i]) for i in range(5)]
bars = ax.bar(cls_names, counts, color=colors)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, cnt + 10, str(cnt),
            ha="center", va="bottom", fontsize=10)
ax.set_title("Figure 2 -- Class Distribution (Bonn EEG Dataset)")
ax.set_ylabel("Number of Samples"); ax.set_ylim(0, max(counts)*1.15)
plt.tight_layout()
plt.savefig(FIGS / "fig02_class_distribution.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig02_class_distribution.png")

# Dataset imbalance pie charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, (_, y)) in zip(axes, DATASETS.items()):
    pos = y.sum(); neg = len(y) - pos
    ax.pie([neg, pos], labels=["Non-seizure", "Seizure"],
           autopct="%1.1f%%", colors=["#4878CF", "#D65F5F"], startangle=90)
    ax.set_title(name, fontsize=9)
plt.suptitle("Figure 3 -- Dataset Class Balance", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "fig03_dataset_balance.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig03_dataset_balance.png")

# Signal statistics table
stats_rows = []
for i, cls in enumerate(cls_keys):
    vals = X_parts[i].flatten()
    stats_rows.append({
        "Class": cls, "Description": CLASS_DESC[cls],
        "Mean": round(float(np.mean(vals)), 3),
        "Std":  round(float(np.std(vals)), 3),
        "Min":  round(float(np.min(vals)), 3),
        "Max":  round(float(np.max(vals)), 3),
        "Skew": round(float(skew(vals)), 3),
        "Kurt": round(float(kurtosis(vals)), 3),
    })
stats_df = pd.DataFrame(stats_rows)
print("\nClass statistics:\n", stats_df.to_string(index=False))
stats_df.to_csv(FIGS / "table01_class_statistics.csv", index=False)



# 3. PREPROCESSING PIPELINES

print("\n" + "="*60)
print("SECTION 3 -- PREPROCESSING PIPELINES")
print("="*60)

# ?? Pipeline B helper: feature extraction ?????
def extract_features(X_raw: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Extract 28 hand-crafted time/frequency features per EEG segment.
    """
    feats = []
    for seg in X_raw:
        f = []
        # Time-domain (14)
        f.append(np.mean(seg))
        f.append(np.std(seg))
        f.append(np.var(seg))
        f.append(float(np.min(seg)))
        f.append(float(np.max(seg)))
        f.append(float(np.max(seg) - np.min(seg)))
        f.append(float(np.median(seg)))
        f.append(float(np.sqrt(np.mean(seg**2))))          # RMS
        f.append(float(skew(seg)))
        f.append(float(kurtosis(seg)))
        # zero-crossing rate
        zcr = np.sum(np.diff(np.sign(seg)) != 0) / len(seg)
        f.append(float(zcr))
        f.append(float(np.sum(seg**2)))                    # energy
        # line length
        f.append(float(np.sum(np.abs(np.diff(seg)))))
        # peak-to-peak
        f.append(float(np.ptp(seg)))

        # Frequency-domain (14)
        freqs, psd = sp_signal.welch(seg, fs=fs, nperseg=min(256, len(seg)))
        # Band powers (delta theta alpha beta gamma)
        def band_power(lo, hi):
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            return float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0
        f.append(band_power(0.5,  4))    # delta
        f.append(band_power(4,    8))    # theta
        f.append(band_power(8,   13))    # alpha
        f.append(band_power(13,  30))    # beta
        f.append(band_power(30,  fs/2))  # gamma
        total_power = band_power(0.5, fs/2) + 1e-12
        # relative powers (3)
        f.append(band_power(0.5, 4)  / total_power)
        f.append(band_power(4,  13)  / total_power)
        f.append(band_power(13, fs/2)/ total_power)
        # spectral entropy
        psd_norm = psd / (psd.sum() + 1e-12)
        f.append(float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12))))
        # dominant frequency & its power
        dom_idx = np.argmax(psd)
        f.append(float(freqs[dom_idx]))
        f.append(float(psd[dom_idx]))
        # spectral centroid
        f.append(float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)))
        # median frequency
        cumulative = np.cumsum(psd)
        median_freq_idx = np.searchsorted(cumulative, cumulative[-1] / 2)
        f.append(float(freqs[min(median_freq_idx, len(freqs)-1)]))

        feats.append(f)
    return np.array(feats)

# ?? Pipeline A: StandardScaler -> SavGol filter -> SelectKBest ?
def pipeline_A(X_train, X_test, y_train, k=100):
    """
    Pipeline A -- raw signal approach.
    Step 1: StandardScaler normalisation
    Step 2: Savitzky-Golay smoothing (noise removal)
    Step 3: SelectKBest univariate feature selection
    Returns: X_train_proc, X_test_proc, selector
    """
    # Step 1 - Normalise
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    # Step 2 - Noise removal via Savitzky-Golay along each sample row
    Xtr = np.apply_along_axis(
        lambda s: sp_signal.savgol_filter(s, window_length=11, polyorder=3), 1, Xtr)
    Xte = np.apply_along_axis(
        lambda s: sp_signal.savgol_filter(s, window_length=11, polyorder=3), 1, Xte)
    # Step 3 - Feature selection
    selector = SelectKBest(f_classif, k=min(k, Xtr.shape[1]))
    Xtr = selector.fit_transform(Xtr, y_train)
    Xte = selector.transform(Xte)
    return Xtr, Xte, scaler, selector

# ?? Pipeline B: Feature extraction -> MinMaxScaler -> PCA ??????
def pipeline_B(X_train, X_test, y_train, n_comp=15):
    """
    Pipeline B -- feature engineering approach.
    Step 1: Extract 28 statistical/spectral features
    Step 2: MinMaxScaler normalisation
    Step 3: PCA dimensionality reduction
    Returns: X_train_proc, X_test_proc, pca
    """
    # Step 1 - Extract features
    Xtr = extract_features(X_train)
    Xte = extract_features(X_test)
    # Step 2 - Scale
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    # Step 3 - PCA
    pca = PCA(n_components=min(n_comp, Xtr.shape[1]))
    Xtr = pca.fit_transform(Xtr)
    Xte = pca.transform(Xte)
    return Xtr, Xte, scaler, pca

print("  Pipeline A: StandardScaler -> Savitzky-Golay -> SelectKBest(k=100)")
print("  Pipeline B: Feature Extraction (28 feats) -> MinMaxScaler -> PCA(15)")

# ?? Pipeline diagram ?????????????????????????
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (title, steps, colors_p) in zip(axes, [
    ("Pipeline A -- Raw Signal", ["StandardScaler\n(Normalise)", "Savitzky-Golay\nFilter\n(Noise Removal)", "SelectKBest\n(f_classif, k=100)\n(Feature Selection)"], ["#4C72B0", "#55A868", "#DD8452"]),
    ("Pipeline B -- Feature Engineering", ["Feature Extraction\n(28 statistical /\nspectral features)", "MinMaxScaler\n(Normalise)", "PCA\n(n=15 components)\n(Dim. Reduction)"], ["#C44E52", "#8172B2", "#937860"])
]):
    ax.set_xlim(0, 10); ax.set_ylim(-1, 3); ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    for j, (step, col) in enumerate(zip(steps, colors_p)):
        x = 1.5 + j * 3.2
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x - 1.2, 0.2), 2.4, 1.8,
                              boxstyle="round,pad=0.1", fc=col, ec="white", lw=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, 1.1, step, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if j < len(steps) - 1:
            ax.annotate("", xy=(x + 1.4, 1.1), xytext=(x + 1.2, 1.1),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))
plt.suptitle("Figure 4 -- Preprocessing Pipeline Diagrams", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "fig04_pipeline_diagram.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig04_pipeline_diagram.png")



# 4. BASELINE LOGISTIC REGRESSION

print("\n" + "="*60)
print("SECTION 4 -- BASELINE LOGISTIC REGRESSION")
print("="*60)

def evaluate_model(model, X_test, y_test, label=""):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    prauc  = average_precision_score(y_test, y_prob)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    prec   = precision_score(y_test, y_pred, zero_division=0)
    return {"Label": label, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "PR-AUC": prauc}

baseline_rows = []
PIPELINE_FUNCS = {"Pipeline A": pipeline_A, "Pipeline B": pipeline_B}

for ds_name, (X, y) in DATASETS.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           stratify=y, random_state=42)
    for pp_name, pp_func in PIPELINE_FUNCS.items():
        Xtr_p, Xte_p, *_ = pp_func(Xtr, Xte, ytr)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        model.fit(Xtr_p, ytr)
        row = evaluate_model(model, Xte_p, yte, f"{ds_name} | {pp_name}")
        baseline_rows.append(row)
        print(f"  {row['Label']}: Acc={row['Accuracy']:.3f}  F1={row['F1']:.3f}  PR-AUC={row['PR-AUC']:.3f}")

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(FIGS / "table02_baseline_results.csv", index=False)

# Baseline results heatmap
pivot = baseline_df.pivot_table(index="Label", values=["Accuracy", "F1", "PR-AUC"])
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
            linewidths=0.5, vmin=0.5, vmax=1.0)
ax.set_title("Figure 5 -- Baseline Logistic Regression Results (Accuracy / F1 / PR-AUC)")
plt.tight_layout()
plt.savefig(FIGS / "fig05_baseline_heatmap.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig05_baseline_heatmap.png")

# Confusion matrices for best pipeline (D1 + Pipeline A)
Xtr, Xte, ytr, yte = train_test_split(X_d1, y_d1, test_size=0.2, stratify=y_d1, random_state=42)
Xtr_p, Xte_p, *_ = pipeline_A(Xtr, Xte, ytr)
best_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
best_model.fit(Xtr_p, ytr)
y_pred_best = best_model.predict(Xte_p)
cm = confusion_matrix(yte, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Non-Seizure", "Seizure"],
            yticklabels=["Non-Seizure", "Seizure"])
ax.set_ylabel("True"); ax.set_xlabel("Predicted")
ax.set_title("Figure 6 -- Confusion Matrix (D1, Pipeline A, Baseline)")
plt.tight_layout()
plt.savefig(FIGS / "fig06_confusion_matrix.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig06_confusion_matrix.png")



# 5. OVERFITTING & UNDERFITTING DEMONSTRATION

print("\n" + "="*60)
print("SECTION 5 -- OVERFITTING & UNDERFITTING")
print("="*60)

# Use D1 (balanced, cleanest dataset) for this section
Xtr_full, Xte_full, ytr_full, yte_full = train_test_split(
    X_d1, y_d1, test_size=0.2, stratify=y_d1, random_state=42)
Xtr_A, Xte_A, *_ = pipeline_A(Xtr_full, Xte_full, ytr_full, k=100)

# ?? 5a. Validation curves vs regularisation (C) ??????????
C_range = np.logspace(-4, 4, 20)

train_scores, val_scores = validation_curve(
    LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
    Xtr_A, ytr_full, param_name="C", param_range=C_range,
    cv=StratifiedKFold(5), scoring="f1", n_jobs=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(C_range, train_scores.mean(1), "b-o", ms=4, label="Training F1")
ax.semilogx(C_range, val_scores.mean(1),   "r-o", ms=4, label="CV Validation F1")
ax.fill_between(C_range, train_scores.mean(1)-train_scores.std(1),
                train_scores.mean(1)+train_scores.std(1), alpha=0.15, color="blue")
ax.fill_between(C_range, val_scores.mean(1)-val_scores.std(1),
                val_scores.mean(1)+val_scores.std(1), alpha=0.15, color="red")
ax.axvspan(C_range[0], 0.005, alpha=0.10, color="orange", label="Underfitting region")
ax.axvspan(200, C_range[-1], alpha=0.10, color="red",    label="Overfitting region")
ax.set_xlabel("Regularisation parameter C (log scale)")
ax.set_ylabel("F1 Score")
ax.set_title("Figure 7 -- Validation Curve: Underfitting vs Overfitting")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "fig07_validation_curve.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig07_validation_curve.png")

# ?? 5b. Learning curves ??????????????????????
def plot_learning_curve(estimator, X, y, title, filename, cv=5):
    train_sizes, tr_sc, cv_sc = learning_curve(
        estimator, X, y, cv=StratifiedKFold(cv),
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="f1", n_jobs=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, tr_sc.mean(1), "b-o", ms=5, label="Training F1")
    ax.plot(train_sizes, cv_sc.mean(1), "r-o", ms=5, label="Cross-Val F1")
    ax.fill_between(train_sizes, tr_sc.mean(1)-tr_sc.std(1),
                    tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color="blue")
    ax.fill_between(train_sizes, cv_sc.mean(1)-cv_sc.std(1),
                    cv_sc.mean(1)+cv_sc.std(1), alpha=0.15, color="red")
    ax.set_xlabel("Training set size"); ax.set_ylabel("F1 Score")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGS / filename, bbox_inches="tight"); plt.close()
    print(f"  >> Saved {filename}")

# Underfitting: very strong L2 + only first 5 features
Xtr_under = Xtr_A[:, :5]; Xte_under = Xte_A[:, :5]
plot_learning_curve(
    LogisticRegression(C=0.0001, solver="lbfgs", max_iter=1000, random_state=42),
    Xtr_under, ytr_full,
    "Figure 8a -- Learning Curve: UNDERFITTING (C=0.0001, 5 features)",
    "fig08a_learning_curve_underfit.png")

# Good fit: optimal C
plot_learning_curve(
    LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42),
    Xtr_A, ytr_full,
    "Figure 8b -- Learning Curve: GOOD FIT (C=1.0, 100 features)",
    "fig08b_learning_curve_goodfit.png")

# Overfitting: no regularisation, full 512 features
Xtr_raw_sc = StandardScaler().fit_transform(Xtr_full)
Xte_raw_sc = StandardScaler().fit_transform(Xte_full)
plot_learning_curve(
    LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=42),
    Xtr_raw_sc, ytr_full,
    "Figure 8c -- Learning Curve: OVERFITTING (C=1e6, 512 raw features)",
    "fig08c_learning_curve_overfit.png")

# Summary bar
scenarios = ["Underfitting\n(C=0.0001, 5 feats)", "Good Fit\n(C=1.0, 100 feats)", "Overfitting\n(C=1e6, 512 feats)"]
train_f1s, val_f1s = [], []
for (C_val, X_fit) in [(0.0001, Xtr_under), (1.0, Xtr_A), (1e6, Xtr_raw_sc)]:
    m = LogisticRegression(C=C_val, solver="lbfgs", max_iter=2000, random_state=42)
    m.fit(X_fit, ytr_full)
    tr = f1_score(ytr_full, m.predict(X_fit), zero_division=0)
    X_test_fit = (Xte_under if C_val == 0.0001 else (Xte_A if C_val == 1.0 else Xte_raw_sc))
    vl = f1_score(yte_full, m.predict(X_test_fit), zero_division=0)
    train_f1s.append(tr); val_f1s.append(vl)
x = np.arange(3)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, train_f1s, 0.4, label="Train F1", color="#4C72B0")
ax.bar(x + 0.2, val_f1s,   0.4, label="Test F1",  color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(scenarios)
ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.1)
ax.set_title("Figure 9 -- Underfitting / Good Fit / Overfitting Comparison")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for xi, (tr, vl) in enumerate(zip(train_f1s, val_f1s)):
    ax.text(xi-0.2, tr+0.02, f"{tr:.3f}", ha="center", fontsize=9)
    ax.text(xi+0.2, vl+0.02, f"{vl:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "fig09_fit_comparison.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig09_fit_comparison.png")



# 6. REGULARISATION STUDY (L1, L2, Elastic Net)
print("\n" + "="*60)
print("SECTION 6 -- REGULARISATION STUDY")
print("="*60)

C_grid  = [0.01, 0.1, 1, 10, 100]
REGS    = {
    "L1 (Lasso)":    dict(penalty="l1", solver="liblinear"),
    "L2 (Ridge)":    dict(penalty="l2", solver="lbfgs"),
    "Elastic Net":   dict(penalty="elasticnet", solver="saga", l1_ratio=0.5),
}

reg_results = []
cv_strat    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold faster

# Pre-extract Pipeline B features for regularisation study (already shown to work well)
REG_PREPROC = {}  # store pre-processed data per dataset
for ds_name, (X, y) in DATASETS.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr_p, Xte_p, *_   = pipeline_B(Xtr, Xte, ytr)   # use B (fast, clean features)
    REG_PREPROC[ds_name] = (Xtr_p, Xte_p, ytr, yte)
    print(f"    Pre-processed {ds_name[:30]} -> Xtr={Xtr_p.shape}", flush=True)

for ds_name, (Xtr_p, Xte_p, ytr, yte) in REG_PREPROC.items():
    for reg_name, reg_kwargs in REGS.items():
        print(f"    {ds_name[:25]} | {reg_name} ...", flush=True)
        for C_val in C_grid:
            max_it = 300
            cv_f1s = cross_val_score(
                LogisticRegression(C=C_val, max_iter=max_it, random_state=42, **reg_kwargs),
                Xtr_p, ytr, cv=cv_strat, scoring="f1")
            model = LogisticRegression(C=C_val, max_iter=max_it, random_state=42, **reg_kwargs)
            model.fit(Xtr_p, ytr)
            y_prob = model.predict_proba(Xte_p)[:, 1]
            prauc  = average_precision_score(yte, y_prob)
            n_nonzero = int(np.sum(np.abs(model.coef_[0]) > 1e-6))
            reg_results.append({
                "Dataset": ds_name, "Regularisation": reg_name,
                "C": C_val, "CV_F1_mean": cv_f1s.mean(),
                "CV_F1_std": cv_f1s.std(), "PR_AUC": prauc,
                "Nonzero_coefs": n_nonzero,
            })
            print(f"      C={C_val} -> CV_F1={cv_f1s.mean():.3f}", flush=True)

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv(FIGS / "table03_regularisation_results.csv", index=False)

# F1 vs C for each dataset
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
line_styles = ["-o", "-s", "-^"]
pal = ["#4C72B0", "#DD8452", "#55A868"]
for ax, (ds_name, grp) in zip(axes, reg_df.groupby("Dataset")):
    for (reg_name, rgrp), ls, col in zip(grp.groupby("Regularisation"), line_styles, pal):
        best = rgrp.loc[rgrp["CV_F1_mean"].idxmax()]
        ax.semilogx(rgrp["C"], rgrp["CV_F1_mean"], ls, label=reg_name,
                    ms=6, color=col, lw=1.8)
        ax.fill_between(rgrp["C"],
                        rgrp["CV_F1_mean"] - rgrp["CV_F1_std"],
                        rgrp["CV_F1_mean"] + rgrp["CV_F1_std"],
                        alpha=0.12, color=col)
    ax.set_title(ds_name.split("-")[0].strip(), fontsize=10)
    ax.set_xlabel("C (log scale)"); ax.set_ylabel("CV F1")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("Cross-Validation F1 Score")
plt.suptitle("Figure 10 -- Regularisation: L1 vs L2 vs Elastic Net (F1 vs C)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "fig10_regularisation_curves.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig10_regularisation_curves.png")

# Sparsity analysis
best_reg = (reg_df.sort_values("CV_F1_mean", ascending=False)
            .groupby(["Dataset", "Regularisation"]).first().reset_index())
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (ds_name, grp) in zip(axes, best_reg.groupby("Dataset")):
    bars = ax.bar(grp["Regularisation"], grp["Nonzero_coefs"],
                  color=["#4C72B0", "#DD8452", "#55A868"])
    for bar, cnt in zip(bars, grp["Nonzero_coefs"]):
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 0.5, str(int(cnt)),
                ha="center", va="bottom", fontsize=10)
    ax.set_title(ds_name.split("-")[0].strip(), fontsize=10)
    ax.set_ylabel("Non-zero Coefficients"); ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=15)
plt.suptitle("Figure 11 -- Sparsity Analysis: Non-zero Coefficients at Best C", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "fig11_sparsity_analysis.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig11_sparsity_analysis.png")

# Stability heatmap (std of CV F1)
stability = reg_df.groupby(["Regularisation", "Dataset"])["CV_F1_std"].mean().unstack()
fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(stability, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax,
            linewidths=0.5)
ax.set_title("Figure 12 -- Stability: Mean CV F1 Std Dev (lower = more stable)")
plt.tight_layout()
plt.savefig(FIGS / "fig12_stability_heatmap.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig12_stability_heatmap.png")



# 7. HANDLING CLASS IMBALANCE

print("\n" + "="*60)
print("SECTION 7 -- CLASS IMBALANCE HANDLING")
print("="*60)

# Focus on D3 (most imbalanced 4:1) and D2 (2:1)
imbalance_results = []

for ds_name, (X, y) in list(DATASETS.items())[1:]:   # D2 and D3
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr_p, Xte_p, *_ = pipeline_A(Xtr, Xte, ytr, k=min(100, Xtr.shape[1]))
    best_C = 1.0

    techniques = {
        "No Resampling":       (Xtr_p, ytr),
        "SMOTE":               SMOTE(random_state=42).fit_resample(Xtr_p, ytr),
        "Undersampling":       RandomUnderSampler(random_state=42).fit_resample(Xtr_p, ytr),
    }

    for tech_name, (Xr, yr) in techniques.items():
        for cw in [None, "balanced"]:
            cw_label = "class_weight=balanced" if cw else "no weight"
            model = LogisticRegression(C=best_C, solver="lbfgs", max_iter=1000,
                                       class_weight=cw, random_state=42)
            model.fit(Xr, yr)
            y_pred = model.predict(Xte_p)
            y_prob = model.predict_proba(Xte_p)[:, 1]
            imbalance_results.append({
                "Dataset": ds_name.split("-")[0].strip(),
                "Technique": f"{tech_name} + {cw_label}",
                "Precision": precision_score(yte, y_pred, zero_division=0),
                "Recall":    recall_score(yte, y_pred, zero_division=0),
                "F1":        f1_score(yte, y_pred, zero_division=0),
                "PR-AUC":    average_precision_score(yte, y_prob),
            })

imb_df = pd.DataFrame(imbalance_results)
imb_df.to_csv(FIGS / "table04_imbalance_results.csv", index=False)
print(imb_df.to_string(index=False))

# Grouped bar chart: Precision vs Recall
for ds_label, grp in imb_df.groupby("Dataset"):
    fig, ax = plt.subplots(figsize=(12, 5))
    x  = np.arange(len(grp))
    ax.bar(x - 0.25, grp["Precision"], 0.25, label="Precision", color="#4C72B0")
    ax.bar(x,        grp["Recall"],    0.25, label="Recall",    color="#55A868")
    ax.bar(x + 0.25, grp["F1"],        0.25, label="F1",        color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(grp["Technique"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score"); ax.legend()
    ax.set_title(f"Figure 13 -- Class Imbalance Handling Comparison ({ds_label})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = f"fig13_imbalance_{ds_label.replace(' ','_').replace('/','_')}.png"
    plt.savefig(FIGS / fname, bbox_inches="tight"); plt.close()
    print(f"  >> Saved {fname}")

# PR curves comparison for D3
ds_name_pr, (X_pr, y_pr) = list(DATASETS.items())[2]
Xtr_pr, Xte_pr, ytr_pr, yte_pr = train_test_split(
    X_pr, y_pr, test_size=0.2, stratify=y_pr, random_state=42)
Xtr_pp, Xte_pp, *_ = pipeline_A(Xtr_pr, Xte_pr, ytr_pr, k=100)

fig, ax = plt.subplots(figsize=(8, 6))
configs = [
    (Xtr_pp, ytr_pr, None, "Baseline"),
    (SMOTE(random_state=42).fit_resample(Xtr_pp, ytr_pr)[0],
     SMOTE(random_state=42).fit_resample(Xtr_pp, ytr_pr)[1], None, "SMOTE"),
    (RandomUnderSampler(random_state=42).fit_resample(Xtr_pp, ytr_pr)[0],
     RandomUnderSampler(random_state=42).fit_resample(Xtr_pp, ytr_pr)[1], None, "Undersampling"),
    (Xtr_pp, ytr_pr, "balanced", "Class Weights"),
]
pr_colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
for (Xr, yr, cw, lbl), col in zip(configs, pr_colors):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                class_weight=cw, random_state=42)
    model.fit(Xr, yr)
    prec_c, rec_c, _ = precision_recall_curve(yte_pr, model.predict_proba(Xte_pp)[:, 1])
    ap = average_precision_score(yte_pr, model.predict_proba(Xte_pp)[:, 1])
    ax.plot(rec_c, prec_c, color=col, lw=2, label=f"{lbl} (AP={ap:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Figure 14 -- Precision-Recall Curves (D3: All vs Seizure)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "fig14_pr_curves.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig14_pr_curves.png")



# 8. COMPARATIVE ANALYSIS

print("\n" + "="*60)
print("SECTION 8 -- COMPARATIVE ANALYSIS")
print("="*60)

# ?? Q1: Does preprocessing order affect results? ??????????
print("\n  Q1: Pipeline A vs Pipeline B across all datasets")
q1_rows = []
for ds_name, (X, y) in DATASETS.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    for pp_name, pp_func in PIPELINE_FUNCS.items():
        Xtr_p, Xte_p, *_ = pp_func(Xtr, Xte, ytr)
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        m.fit(Xtr_p, ytr)
        row = evaluate_model(m, Xte_p, yte, f"{ds_name} | {pp_name}")
        q1_rows.append({"Dataset": ds_name.split("-")[0].strip(),
                        "Pipeline": pp_name, **{k: v for k, v in row.items() if k != "Label"}})
q1_df = pd.DataFrame(q1_rows)
q1_pivot = q1_df.pivot_table(index="Dataset", columns="Pipeline", values="F1")
print(q1_pivot)

fig, ax = plt.subplots(figsize=(9, 5))
q1_pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_xlabel("Dataset"); ax.set_ylabel("F1 Score")
ax.set_title("Figure 15 -- Q1: Does Preprocessing Order Affect Results?")
ax.legend(["Pipeline A (raw->norm->select)", "Pipeline B (extract->scale->PCA)"])
ax.set_ylim(0, 1.1); ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}",
                (p.get_x() + p.get_width()/2, p.get_height() + 0.01),
                ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "fig15_q1_pipeline_comparison.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig15_q1_pipeline_comparison.png")

# ?? Q2: Which regularisation generalises best? ???????????
print("\n  Q2: Best regularisation per dataset")
q2 = (reg_df.sort_values("CV_F1_mean", ascending=False)
      .groupby(["Dataset", "Regularisation"])
      .agg({"CV_F1_mean": "max", "PR_AUC": "max"})
      .reset_index())
q2_pivot = q2.pivot_table(index="Dataset", columns="Regularisation", values="CV_F1_mean")
print(q2_pivot)
fig, ax = plt.subplots(figsize=(10, 5))
q2_pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_xlabel("Dataset"); ax.set_ylabel("Best CV F1")
ax.set_title("Figure 16 -- Q2: Which Regularisation Generalises Best?")
ax.set_ylim(0, 1.1); ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3); ax.legend(title="Regularisation")
plt.tight_layout()
plt.savefig(FIGS / "fig16_q2_regularisation_generalisation.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig16_q2_regularisation_generalisation.png")

# ?? Q3: Does Elastic Net consistently outperform L1/L2? ??
print("\n  Q3: Elastic Net consistency across datasets")
q3 = reg_df.groupby(["Dataset", "Regularisation"])["CV_F1_mean"].max().unstack()
q3["EN_beats_L1"] = q3["Elastic Net"] > q3["L1 (Lasso)"]
q3["EN_beats_L2"] = q3["Elastic Net"] > q3["L2 (Ridge)"]
print(q3[["Elastic Net", "L1 (Lasso)", "L2 (Ridge)", "EN_beats_L1", "EN_beats_L2"]])

# Radar chart

cats  = ["D1", "D2", "D3"]
N_c   = len(cats)
angles = [n / float(N_c) * 2 * np.pi for n in range(N_c)]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for reg_name, col in zip(["L1 (Lasso)", "L2 (Ridge)", "Elastic Net"], ["#4C72B0", "#DD8452", "#55A868"]):
    vals = [q3.loc[d, reg_name] for d in q3.index]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", lw=2, label=reg_name, color=col)
    ax.fill(angles, vals, alpha=0.1, color=col)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(["D1\nHealthy vs Seizure", "D2\nEpileptic vs Seizure", "D3\nAll vs Seizure"])
ax.set_ylim(0, 1)
ax.set_title("Figure 17 -- Q3: Elastic Net vs L1/L2 Consistency\n(Spider Chart)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.savefig(FIGS / "fig17_q3_elastic_net_radar.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig17_q3_elastic_net_radar.png")

# ?? Q4: Imbalance handling x regularisation interaction ??
print("\n  Q4: Imbalance handling x regularisation interaction")
interaction_rows = []
# Use pre-processed D3 (Pipeline B)
Xtr_d3, Xte_d3, ytr_d3, yte_d3 = REG_PREPROC["D3 - All classes vs Seizure (4:1)"]
for reg_name, reg_kwargs in REGS.items():
    smote_X, smote_y = SMOTE(random_state=42).fit_resample(Xtr_d3, ytr_d3)
    under_X, under_y = RandomUnderSampler(random_state=42).fit_resample(Xtr_d3, ytr_d3)
    for tech_name, (Xr, yr) in {
        "SMOTE":          (smote_X, smote_y),
        "Undersampling":  (under_X, under_y),
        "Class Weights":  (Xtr_d3, ytr_d3),
        "No Resampling":  (Xtr_d3, ytr_d3),
    }.items():
        cw = "balanced" if tech_name == "Class Weights" else None
        m = LogisticRegression(C=1.0, max_iter=300, class_weight=cw,
                               random_state=42, **reg_kwargs)
        m.fit(Xr, yr)
        f1_val = f1_score(yte_d3, m.predict(Xte_d3), zero_division=0)
        interaction_rows.append({"Regularisation": reg_name,
                                  "Imbalance Technique": tech_name,
                                  "F1": f1_val})

int_df   = pd.DataFrame(interaction_rows)
int_pivot = int_df.pivot_table(index="Regularisation", columns="Imbalance Technique", values="F1")
print(int_pivot)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(int_pivot, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.5, ax=ax, vmin=0.5, vmax=1.0)
ax.set_title("Figure 18 -- Q4: Regularisation x Imbalance Handling Interaction (F1)")
plt.tight_layout()
plt.savefig(FIGS / "fig18_q4_interaction_heatmap.png", bbox_inches="tight"); plt.close()
print("  >> Saved fig18_q4_interaction_heatmap.png")



# 9. FINAL SUMMARY TABLE

print("\n" + "="*60)
print("SECTION 9 -- FINAL SUMMARY TABLE")
print("="*60)

summary_rows = []
for ds_name, (Xtr_p, Xte_p, ytr, yte) in REG_PREPROC.items():
    # Use pre-processed Pipeline B features
    for reg_name, reg_kwargs in REGS.items():
        best_c = (reg_df[(reg_df["Dataset"] == ds_name) &
                         (reg_df["Regularisation"] == reg_name)]
                  .sort_values("CV_F1_mean", ascending=False)["C"].iloc[0])
        m = LogisticRegression(C=best_c, max_iter=300, random_state=42, **reg_kwargs)
        m.fit(Xtr_p, ytr)
        row = evaluate_model(m, Xte_p, yte, f"{ds_name} | {reg_name}")
        summary_rows.append({
            "Dataset": ds_name.split("-")[0].strip(),
            "Regularisation": reg_name,
            "Best C": best_c,
            **{k: round(v, 4) for k, v in row.items() if k != "Label"}
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(FIGS / "table05_final_summary.csv", index=False)
print(summary_df.to_string(index=False))

# Overall best per dataset
best_per_ds = summary_df.loc[summary_df.groupby("Dataset")["F1"].idxmax()]
print("\n  >> Best regularisation per dataset:")
print(best_per_ds[["Dataset", "Regularisation", "Best C", "Accuracy", "F1", "PR-AUC"]].to_string(index=False))



# 10. SAVE RESULTS JSON

results_json = {
    "baseline":       baseline_df.to_dict(orient="records"),
    "regularisation": reg_df.to_dict(orient="records"),
    "imbalance":      imb_df.to_dict(orient="records"),
    "summary":        summary_df.to_dict(orient="records"),
    "q1_pipeline":    q1_df.to_dict(orient="records"),
    "q4_interaction": int_df.to_dict(orient="records"),
}
with open(FIGS / "all_results.json", "w") as jf:
    json.dump(results_json, jf, indent=2)
print("\n  >> Saved all_results.json")

print("\n" + "="*60)
print("?  ALL SECTIONS COMPLETE -- Results saved in:", FIGS)
print("    Figures: fig01 ? fig18")
print("    Tables:  table01 ? table05")
print("="*60)
