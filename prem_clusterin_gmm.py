# -*- coding: utf-8 -*-
"""
Created on Wed June 3 14:23:54 2026

@author: jthon
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# %% UNSUPERVISED CLUSTERING (SAME DATASET & PREPRECOSSING AS K-MEANS FILE)
# Merge all club csv files
# Change directory depdning on where dataset stored
DATA_DIR = r"C:\Users\jthon\Downloads\Premier League Datasets\Premier League Datasets"
 
def load_all_clubs(data_dir: str) -> pd.DataFrame:
    """
    Reads every CSV in data_dir, strips the two summary rows
    ('Squad Total', 'Opponent Total'), parses comma-formatted numbers,
    tags each row with its club name, and returns one combined DataFrame. *Used ChatGPT for this
    """
    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
 
    frames = []
    for path in sorted(csv_paths):
        club_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, thousands=",")   # thousands="," converts "2,790" → 2790
        df["Club"] = club_name
        # Drop summary rows that are not real players
        df = df[~df["Player"].isin(["Squad Total", "Opponent Total"])]
        frames.append(df)
 
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csv_paths)} clubs → {combined.shape[0]} players, "
          f"{combined.shape[1]} columns")
    return combined
 
 
raw_df = load_all_clubs(DATA_DIR)

# %% PREPROCESSING
# Remove players with very few minutes played
# Threshold to remove noise from players who have little contribution over the course of a season
MIN_MINUTES = 1080 # ~11 games of gametime out of the 38 in the season

#remove players with very low minutes played (noise)
if "Minutes_Played" in raw_df.columns:
    raw_df = raw_df[raw_df["Minutes_Played"] >= MIN_MINUTES].copy()

print(f"Players after filtering: {len(raw_df)}")

# Save player information separately (not features)
meta_cols = ["Player", "Nation", "Pos", "Age", "Club"]
meta_df = raw_df[[c for c in meta_cols if c in raw_df.columns]].copy()

# all other info are used as features for clustering
features = raw_df.drop(columns=[c for c in meta_cols if c in raw_df.columns]).copy()
features = raw_df.drop(
    columns=[c for c in meta_cols if c in raw_df.columns]
).copy()

# Remove non performance related variables (+ any redundant ones))
cols_to_remove = [
    "Matches_Played",
    "Starts",
    "Minutes_Played",
    "90s",
    "90s.1"
]
features = features.drop(
    columns=[c for c in cols_to_remove if c in features.columns]
)

# Convert percentage columns
percentage_features = features.filter(like='%', axis=1).columns

for col in percentage_features:
    features[col] = (features[col].astype(str).str.replace('%', '', regex=False).replace('nan', np.nan).astype(float)/ 100)

# Keep only numeric columns
features = features.select_dtypes(include=np.number)

# Fill missing values with median
features = features.fillna(features.median())

print(f"Features used: {features.shape[1]}")

# Standardize to mean=0 std=1
for col in features.columns:

    mean = features[col].mean()
    std = features[col].std()

    if std > 0:
        features[col] = (features[col] - mean) / std

# Dataset used for clustering
X = features

print(f"Final clustering matrix shape: {X.shape}")


## CLUSTERING GMM
K_RANGE = range(7, 18)

N_RUNS = 10 # different random seeds per K
N_INIT = 3 # given random seed, run algorithm 3 separate times w/ diff init & keep the highest likelihood
COV_TYPE = "full"

#init
best_sil_per_k = [] 
mean_sil_per_k = []
std_sil_per_k = []

best_k = None
best_sil_overall = -np.inf
best_gmm = None
best_hard_labels = None

print("\n" + "-"*60)
print("GMM model selection over K = 7 … 15")
print("Selection criterion: silhouette score (consistent with K-Means)")
print("-"*60)

#alg
for k in K_RANGE: #iterate through value of K

    print(f"\nTesting K = {k}")

    run_scores = []

    best_run_sil = -np.inf
    best_run_gmm = None
    best_run_labels = None

    for run in range(N_RUNS): # iterate through different random seeds for each K

        gmm = GaussianMixture(
            n_components=k,
            covariance_type=COV_TYPE,
            n_init=N_INIT,
            random_state=run,
            max_iter=300
        )

        gmm.fit(X)

        labels = gmm.predict(X)

        sizes = np.bincount(labels)

        # Skip runs where any cluster has fewer than 3 members
        if sizes.min() < 3:
            print(f"  seed={run}  SKIP (min cluster size={sizes.min()})")
            continue

        sil = silhouette_score(X, labels)

        run_scores.append(sil)

        print(f"  seed={run}  sil={sil:.4f}")

        if sil > best_run_sil:
            best_run_sil = sil
            best_run_gmm = gmm
            best_run_labels = labels

    if not run_scores:

        print(f"  K={k} skipped (all runs degenerate)")

        best_sil_per_k.append(-1)
        mean_sil_per_k.append(-1)
        std_sil_per_k.append(0)

        continue

    best_sil_per_k.append(best_run_sil)
    mean_sil_per_k.append(np.mean(run_scores))
    std_sil_per_k.append(np.std(run_scores))

    print(
        f"  → best={best_run_sil:.4f}"
        f"  mean={np.mean(run_scores):.4f}"
        f"  std={np.std(run_scores):.4f}"
    )

    if np.mean(run_scores) > best_sil_overall:

        best_sil_overall = np.mean(run_scores)

        best_k = k
        best_gmm = best_run_gmm
        best_hard_labels = best_run_labels

print("\n" + "="*60)
print(f"Best K by mean silhouette = {best_k}")
print(f"Mean silhouette = {best_sil_overall:.4f}")
print(f"  -> Using K={best_k} (silhouette) to match K-Means selection method")
print("="*60)

# Reset index so meta_df rows align 0...N-1
meta_df = meta_df.reset_index(drop=True)

# Attach hard labels to metadata
meta_df["Cluster"] = best_hard_labels

# Print Clusters
for cluster_id in sorted(meta_df["Cluster"].unique()):
    cluster_players = meta_df[meta_df["Cluster"] == cluster_id].copy()
    print("\n" + "-"*60)
    print(f"CLUSTER {cluster_id}")
    print(f"Total Players: {len(cluster_players)}")

    pos_counts = {}
    for pos in cluster_players["Pos"]:
        if pd.isna(pos):
            continue
        for p in [s.strip() for s in str(pos).split(",")]:
            pos_counts[p] = pos_counts.get(p, 0) + 1

    print("\nPosition Breakdown:")
    for pos, count in sorted(pos_counts.items()):
        print(f"  {pos}: {count}")

    for _, row in cluster_players.iterrows():
        player = str(row.get("Player", "Unknown"))
        club   = str(row.get("Club",   "Unknown"))
        pos    = str(row.get("Pos",    "Unknown"))
        print(f"{player:<25} | {club:<20} | {pos:<8}")

##### FIGURES ######
# Silhouette score
plt.figure(figsize=(8, 5))

plt.errorbar(
    list(K_RANGE),
    mean_sil_per_k,
    yerr=std_sil_per_k,
    marker='o',
    linewidth=2,
    capsize=5,
    label='Mean silhouette ± std'
)

plt.axvline(
    best_k,
    linestyle=':',
    linewidth=2,
    label=f'Best K={best_k}'
)

plt.xlabel("Number of Components (K)")
plt.ylabel("Mean Silhouette Score (higher = better)")
plt.title(f"GMM: Mean Silhouette Score vs K ({N_RUNS} runs per K)")

plt.legend()
plt.grid(True)

plt.tight_layout()

# Project GMM means into 2D PCA space
pca_2d     = PCA(n_components=2, random_state=42)
X_pca2     = pca_2d.fit_transform(X)
means_pca2 = pca_2d.transform(best_gmm.means_)
var2       = pca_2d.explained_variance_ratio_
cmap_c = plt.colormaps['tab20'].resampled(best_k)

fig, ax2d = plt.subplots(figsize=(10, 8))
for c in range(best_k):
    mask = best_hard_labels == c
    ax2d.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                 color=cmap_c(c), alpha=0.6, s=30, label=f'C{c}')

for i, (x, y) in enumerate(means_pca2):
    ax2d.text(x, y, str(i), fontsize=13, fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='white', alpha=0.85, edgecolor='black'))

ax2d.set_title(f'GMM: Player Clusters  (K={best_k})\n'
               f'PC1+PC2 = {100*sum(var2):.1f}% variance')
ax2d.set_xlabel(f'PC1 ({var2[0]*100:.1f}%)')
ax2d.set_ylabel(f'PC2 ({var2[1]*100:.1f}%)')
ax2d.legend(loc='best', title='Cluster', framealpha=0.9, fontsize=8)
ax2d.grid(True, alpha=0.3)
plt.tight_layout()

# 3D PCA plot with Gaussian ellipsoids
pca_3d = PCA(n_components=3, random_state=42)
X_pca3 = pca_3d.fit_transform(X)

W = pca_3d.components_
var3 = pca_3d.explained_variance_ratio_
means_pca3 = pca_3d.transform(best_gmm.means_)
cmap_c = plt.colormaps['tab20'].resampled(best_k)

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
sphere = np.stack([
    np.outer(np.cos(u), np.sin(v)),
    np.outer(np.sin(u), np.sin(v)),
    np.outer(np.ones_like(u), np.cos(v))
], axis=-1)

fig3d = plt.figure(figsize=(14, 10))
ax3d = fig3d.add_subplot(111, projection='3d')

for c in range(best_k):
    color = cmap_c(c)
    mask = best_hard_labels == c

    ax3d.scatter(
        X_pca3[mask, 0],
        X_pca3[mask, 1],
        X_pca3[mask, 2],
        color=color,
        alpha=0.55,
        s=18,
        label=f'C{c} ({mask.sum()})'
    )

    C_full = best_gmm.covariances_[c]
    C_pca = W @ C_full @ W.T

    try:
        L = np.linalg.cholesky(C_pca)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(C_pca)
        eigvals = np.maximum(eigvals, 1e-6)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    pts = sphere @ L.T
    mu = means_pca3[c]

    ax3d.plot_surface(
        mu[0] + pts[:, :, 0],
        mu[1] + pts[:, :, 1],
        mu[2] + pts[:, :, 2],
        color=color,
        alpha=0.12,
        linewidth=0
    )

    ax3d.scatter(*mu, color=color, s=120, marker='*',
                 edgecolors='black', linewidths=0.5)

    ax3d.text(mu[0], mu[1], mu[2], f' {c}',
              fontsize=8, fontweight='bold')

ax3d.set_xlabel(f'PC1 ({var3[0]*100:.1f}%)', labelpad=6)
ax3d.set_ylabel(f'PC2 ({var3[1]*100:.1f}%)', labelpad=6)
ax3d.set_zlabel(f'PC3 ({var3[2]*100:.1f}%)', labelpad=6)
ax3d.set_title(
    f'GMM Clusters in 3-D PCA Space  (K={best_k})\n'
    f'Ellipsoids = 1-sigma Gaussian  |  Stars = component means  |  '
    f'{sum(var3)*100:.1f}% variance explained',
    fontsize=10
)
ax3d.legend(loc='upper left', bbox_to_anchor=(-0.18, 1.0),
            fontsize=7, framealpha=0.8, title='Cluster')
ax3d.view_init(elev=20, azim=45)
plt.tight_layout()
plt.show()