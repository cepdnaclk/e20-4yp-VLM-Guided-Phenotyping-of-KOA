"""
Phenotype Clustering Pipeline
==============================
1. Load embeddings + metadata from .npz
2. Fuse image embeddings with clinical features
3. Run UMAP dimensionality reduction
4. Run HDBSCAN clustering WITHIN each KL grade
5. Save cluster assignments + UMAP coordinates for visualization

Usage:
    pip install umap-learn hdbscan scikit-learn matplotlib seaborn
    python clustering.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
NPZ_PATH   = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/embeddings/embeddings_V00_finetuned.npz")
OUTPUT_DIR = Path("/scratch1/e20-fyp-vlm-knee-osteo/e20378-4yp-VLM-Guided-Phenotyping-of-KOA/clustering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── UMAP + HDBSCAN settings ───────────────────────────────────────────────────
UMAP_PARAMS = dict(
    n_components   = 2,
    n_neighbors    = 30,
    min_dist       = 0.1,
    metric         = "cosine",
    random_state   = 42,
)

HDBSCAN_PARAMS = dict(
    min_cluster_size  = 15,   # minimum knees per cluster
    min_samples       = 5,
    metric            = "euclidean",
    cluster_selection_method = "eom",
)

# Weight for clinical features vs image embeddings (tune as needed)
CLINICAL_WEIGHT = 2.0


def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    mask = d["matched"]

    embeddings  = d["embeddings"][mask]
    kl_grades   = d["kl_grades"][mask].astype(int)
    pain        = d["pain_scores"][mask]
    bmi         = d["bmis"][mask]
    age         = d["ages"][mask]
    jsn_lat     = d["jsn_lateral"][mask]
    jsn_med     = d["jsn_medial"][mask]
    subject_ids = d["subject_ids"][mask]
    knee_sides  = d["knee_sides"][mask]
    image_paths = d["image_paths"][mask]

    print(f"Loaded {len(embeddings)} matched embeddings")
    print(f"KL distribution: { {k: int((kl_grades==k).sum()) for k in range(5)} }")
    return {
        "embeddings":  embeddings,
        "kl_grades":   kl_grades,
        "pain":        pain,
        "bmi":         bmi,
        "age":         age,
        "jsn_lat":     jsn_lat,
        "jsn_med":     jsn_med,
        "subject_ids": subject_ids,
        "knee_sides":  knee_sides,
        "image_paths": image_paths,
    }


def build_fused_features(data, indices):
    """
    Fuse image embeddings with scaled clinical features.
    Clinical features: pain, BMI, age, JSN lateral, JSN medial
    """
    emb = data["embeddings"][indices]

    # Stack clinical features
    clinical = np.column_stack([
        data["pain"][indices],
        data["bmi"][indices],
        data["age"][indices],
        data["jsn_lat"][indices],
        data["jsn_med"][indices],
    ])

    # Handle NaN — fill with column median
    for col in range(clinical.shape[1]):
        col_data = clinical[:, col]
        median   = np.nanmedian(col_data)
        clinical[np.isnan(col_data), col] = median

    # Scale clinical features to unit variance
    scaler  = StandardScaler()
    clinical_scaled = scaler.fit_transform(clinical) * CLINICAL_WEIGHT

    # Concatenate: image embedding (512) + clinical (5 * weight)
    fused = np.hstack([emb, clinical_scaled])
    return fused


def cluster_kl_grade(data, kl, output_dir):
    """Run full clustering pipeline for one KL grade."""
    indices = np.where(data["kl_grades"] == kl)[0]
    n = len(indices)
    print(f"\n── KL Grade {kl} ({n} knees) ──────────────────────────")

    if n < 30:
        print(f"  Too few samples ({n}), skipping.")
        return None

    # Build fused features
    fused = build_fused_features(data, indices)

    # UMAP
    print(f"  Running UMAP...")
    reducer   = umap.UMAP(**UMAP_PARAMS)
    umap_2d   = reducer.fit_transform(fused)

    # HDBSCAN
    print(f"  Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    labels    = clusterer.fit_predict(umap_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise} ({n_noise/n*100:.1f}%)")

    # Silhouette score (exclude noise)
    valid = labels != -1
    if valid.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(umap_2d[valid], labels[valid])
        print(f"  Silhouette score: {sil:.3f}")
    else:
        sil = np.nan
        print(f"  Silhouette score: N/A (only {n_clusters} cluster)")

    # Cluster profiles — mean clinical features per cluster
    df_kl = pd.DataFrame({
        "cluster":    labels,
        "pain":       data["pain"][indices],
        "bmi":        data["bmi"][indices],
        "age":        data["age"][indices],
        "jsn_lat":    data["jsn_lat"][indices],
        "jsn_med":    data["jsn_med"][indices],
        "knee_side":  data["knee_sides"][indices],
        "subject_id": data["subject_ids"][indices],
        "umap_x":     umap_2d[:, 0],
        "umap_y":     umap_2d[:, 1],
    })

    # Print cluster profiles
    print(f"\n  Cluster profiles (mean values):")
    profile = df_kl[df_kl["cluster"] != -1].groupby("cluster")[
        ["pain", "bmi", "age", "jsn_lat", "jsn_med"]
    ].mean().round(2)
    print(profile.to_string())

    # ── Plot UMAP ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"KL Grade {kl} — {n} knees — {n_clusters} clusters", fontsize=14)

    # Plot 1: Cluster labels
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("tab10", n_colors=max(n_clusters, 1))
    color_map = {-1: (0.7, 0.7, 0.7)}
    for i, c in enumerate([l for l in unique_labels if l != -1]):
        color_map[c] = palette[i]

    colors = [color_map[l] for l in labels]
    axes[0].scatter(umap_2d[:, 0], umap_2d[:, 1], c=colors, s=8, alpha=0.6)
    axes[0].set_title("Clusters")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    # Add cluster number labels at centroids
    for c in unique_labels:
        if c == -1:
            continue
        mask_c = labels == c
        cx, cy = umap_2d[mask_c, 0].mean(), umap_2d[mask_c, 1].mean()
        axes[0].text(cx, cy, str(c), fontsize=12, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Plot 2: Pain score overlay
    sc = axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1],
                         c=data["pain"][indices], cmap="RdYlGn_r",
                         s=8, alpha=0.6, vmin=0, vmax=20)
    plt.colorbar(sc, ax=axes[1])
    axes[1].set_title("Pain score (WOMAC)")
    axes[1].set_xlabel("UMAP 1")

    # Plot 3: JSN medial overlay
    sc2 = axes[2].scatter(umap_2d[:, 0], umap_2d[:, 1],
                          c=data["jsn_med"][indices], cmap="YlOrRd",
                          s=8, alpha=0.6, vmin=0, vmax=3)
    plt.colorbar(sc2, ax=axes[2])
    axes[2].set_title("JSN medial score")
    axes[2].set_xlabel("UMAP 1")

    plt.tight_layout()
    plot_path = output_dir / f"finetuned_umap_KL{kl}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {plot_path}")

    return {
        "kl":         kl,
        "n":          n,
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "silhouette": sil,
        "df":         df_kl,
        "umap_2d":    umap_2d,
        "labels":     labels,
        "indices":    indices,
    }


def main():
    data    = load_data(NPZ_PATH)
    results = {}

    for kl in range(5):
        result = cluster_kl_grade(data, kl, OUTPUT_DIR)
        if result:
            results[kl] = result

    # Save all cluster assignments to CSV
    all_rows = []
    for kl, res in results.items():
        df = res["df"].copy()
        df["kl_grade"] = kl
        all_rows.append(df)

    final_df = pd.concat(all_rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "cluster_assignments_finetuned.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"\nAll cluster assignments saved → {csv_path}")

    # Summary table
    print(f"\n{'='*55}")
    print(f"{'KL':>4} {'N':>6} {'Clusters':>9} {'Noise%':>8} {'Silhouette':>11}")
    print(f"{'='*55}")
    for kl, res in results.items():
        noise_pct = res["n_noise"] / res["n"] * 100
        sil = f"{res['silhouette']:.3f}" if not np.isnan(res["silhouette"]) else "  N/A"
        print(f"{kl:>4} {res['n']:>6} {res['n_clusters']:>9} {noise_pct:>7.1f}% {sil:>11}")
    print(f"{'='*55}")
    print(f"\nDone! Copy plots to local:")
    print(f"  scp -r e20378@ada:{OUTPUT_DIR} ./clustering_results")


if __name__ == "__main__":
    main()