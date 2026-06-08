# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:04:52 2026

@author: jthon
FOR KMEANS ++
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% UNSUPERVISED CLUSTERING
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

# %% CLUSTERING: K-MEANS
def getKMeansPlusPlusCentroids(dataset, k):

    X = dataset.values
    n_samples = len(X)

    centroids = []

    # First centroid chosen uniformly at random
    first_idx = np.random.choice(n_samples)
    centroids.append(X[first_idx])

    # Choose remaining centroids
    for _ in range(1, k):

        distances = np.zeros(n_samples)

        for i, point in enumerate(X):

            # distance to nearest existing centroid
            min_dist = np.inf

            for centroid in centroids:

                dist = np.sum((point - centroid) ** 2)

                if dist < min_dist:
                    min_dist = dist

            distances[i] = min_dist

        probabilities = distances / np.sum(distances)

        next_idx = np.random.choice(
            n_samples,
            p=probabilities
        )

        centroids.append(X[next_idx])

    return np.array(centroids)

# Returns a label for each piece of data in the dataset.
def getLabels(dataset, centroids):
    """
    For each element in the dataset, chose the closest centroid.
    Make that centroid the element's label
    """
    k = len(centroids)
    labelDict = {i: [] for i in range(k)}
    for idx in range(dataset.shape[0]):
        point = dataset.iloc[idx].values
        dists = np.zeros(k)
        for i, centroid in enumerate(centroids):
            dists[i] = np.sum((point - centroid) ** 2)
        closest = np.argmin(dists)
        labelDict[closest].append(idx)
    return labelDict

def getCentroids(dataset, labels, numFeatures, k):
    """
    Each centroid is the geometric mean of the points that
    have that centroid's label. Important: If a centroid is empty (no points have
    that centroid's label) you should randomly re-initialize it.
    """
    centroids = np.zeros((k, numFeatures))
    for cluster_idx, point_indices in labels.items():
        if len(point_indices) == 0:
            centroids[cluster_idx] = np.random.uniform(low=-1.0, high=1.0, size=(numFeatures,))
        else:
            centroids[cluster_idx] = np.mean(dataset.iloc[point_indices],axis=0)
    return centroids


def kMeans(dataset, k, iterations=50):
    numFeatures = dataset.shape[1]
    centroids = getKMeansPlusPlusCentroids(dataset, k)

    allCentroids = []

    for i in range(iterations):
        allCentroids.append(centroids.copy())

        labels = getLabels(dataset, centroids)

        centroids = getCentroids(dataset, labels, numFeatures,k)

        # print(f"Iteration {i+1}/{iterations}")

    return centroids, labels, allCentroids

# %% FIND BEST K ( + RUN KMEANS)
from sklearn.metrics import silhouette_score

k_values = range(6, 16)

n_runs = 10

mean_scores = []
std_scores = []

best_k = None
best_mean_score = -np.inf

best_centroids = None
best_labels = None
best_allCentroids = None

for k in k_values:

    print("\n" + "="*60)
    print(f"Testing K = {k}")

    run_scores = []

    best_run_score = -np.inf

    current_best_centroids = None
    current_best_labels = None
    current_best_allCentroids = None

    for run in range(n_runs):

        np.random.seed(run)

        print(f"  Run {run+1}/{n_runs}")

        centroids, labels, allCentroids = kMeans(
            X,
            k=k,
            iterations=20
        )

        cluster_labels = np.zeros(len(X), dtype=int)

        for cluster_id, player_indices in labels.items():
            for idx in player_indices:
                cluster_labels[idx] = cluster_id

        score = silhouette_score(X, cluster_labels)

        run_scores.append(score)

        print(f"     Silhouette Score = {score:.4f}")

        if score > best_run_score:

            best_run_score = score

            current_best_centroids = centroids.copy()
            current_best_labels = labels.copy()
            current_best_allCentroids = allCentroids.copy()

    mean_score = np.mean(run_scores)
    std_score = np.std(run_scores)

    mean_scores.append(mean_score)
    std_scores.append(std_score)

    print(
        f"K={k} | Mean={mean_score:.4f} | Std={std_score:.4f}"
    )

    if mean_score > best_mean_score:

        best_mean_score = mean_score

        best_k = k

        best_centroids = current_best_centroids
        best_labels = current_best_labels
        best_allCentroids = current_best_allCentroids

print("\n" + "="*60)
print(
    f"K={k} | Run {run+1}/{n_runs} | "
    f"Silhouette = {score:.4f}"
)

centroids = best_centroids
labels = best_labels
allCentroids = best_allCentroids

print(f"Final clustering complete with K={best_k}")

# %% PLAYER CLUSTERS

cluster_labels = np.zeros(len(X), dtype=int)

for cluster_id, player_indices in labels.items():
    for idx in player_indices:
        cluster_labels[idx] = cluster_id
meta_df["Cluster"] = cluster_labels

# Print all clusters and players
for cluster_id in sorted(meta_df["Cluster"].unique()):
    cluster_players = meta_df[meta_df["Cluster"] == cluster_id].copy()
    print("\n" + "-"*60)
    print(f"CLUSTER {cluster_id}")

    # Number of players
    print(f"Total Players: {len(cluster_players)}")
    # Position counts
    pos_counts = {}

    for pos in cluster_players["Pos"]:
        if pd.isna(pos):
            continue
        # Handle positions like "MF,FW"
        positions = [p.strip() for p in str(pos).split(",")]

        for p in positions:
            if p not in pos_counts:
                pos_counts[p] = 0
            pos_counts[p] += 1
    print("\nPosition Breakdown:")

    for pos, count in sorted(pos_counts.items()):
        print(f"  {pos}: {count}")
    print("\nPlayers:")
    print("-"*60)
    
    cluster_players = cluster_players.sort_values("Player")

    for _, row in cluster_players.iterrows():
        player = str(row.get("Player", "Unknown"))
        club = str(row.get("Club", "Unknown"))
        pos = str(row.get("Pos", "Unknown"))
        
        if player == "nan":
            player = "Unknown"

        if club == "nan":
            club = "Unknown"

        if pos == "nan":
            pos = "Unknown"

        print(f"{player:<25} | {club:<20} | {pos}")

# PCA 2-D scatter with K-Means ++ components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Variance explained:",round(100*np.sum(pca.explained_variance_ratio_),2),"%")

# Convert centroids to DataFrame so feature names match PCA input
centroids_df = pd.DataFrame(
    centroids,
    columns=X.columns
)

centroids_pca = pca.transform(centroids_df)
plt.figure(3, figsize=(10,8))
num_clusters = len(np.unique(cluster_labels))
cmap = plt.cm.get_cmap('tab20', num_clusters)

for cluster_id in range(num_clusters):
    mask = cluster_labels == cluster_id
    plt.scatter(
        X_pca[mask,0],
        X_pca[mask,1],
        color=cmap(cluster_id),
        alpha=0.6,
        label=f'Cluster {cluster_id}'
    )

for i, (x, y) in enumerate(centroids_pca):
    plt.text(
        x,
        y,
        str(i),
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='center',
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='black'
        )
    )

plt.title('Player Clusters with Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.legend(
    loc='best',
    title='Clusters',
    framealpha=0.9
)

plt.tight_layout()

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=X.columns
)

print("\n")
print("="*70)
print("PCA LOADINGS")
print("="*70)

print(loadings)

# Silhouette score

plt.figure(4, figsize=(8,5))

plt.errorbar(
    list(k_values),
    mean_scores,
    yerr=std_scores,
    marker='o',
    linewidth=2,
    capsize=5
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Average Silhouette Score")
plt.title("K-Means Performance vs Number of Clusters")

plt.grid(True)

plt.tight_layout()
plt.show()

