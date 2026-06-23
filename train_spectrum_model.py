"""
Shopper Spectrum — Model Training Pipeline
==========================================
Trains the KMeans spectrum detector (RFM clustering) and item-based
collaborative filtering recommender. Run after CustomerEDA.ipynb has
produced the cleaned CSV files in data/.

Usage:
    python train_spectrum_model.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ─── Paths (all relative to this script) ─────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

RFM_PATH = DATA_DIR / "rfm_features.csv"
CLEANED_PATH = DATA_DIR / "online_retail_cleaned.csv"
OPTIMAL_K = 4
K_RANGE = range(2, 12)
RANDOM_STATE = 42

# Segment colour palette used in visualizations and the Streamlit app
SEGMENT_PALETTE = {
    "High-Value": "#2ECC71",
    "Regular": "#3498DB",
    "Occasional": "#F39C12",
    "At-Risk": "#E74C3C",
}


def ensure_dirs() -> None:
    """Create output folders if they do not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def auto_label_clusters(centroids_df: pd.DataFrame) -> dict[int, str]:
    """
    Map arbitrary KMeans cluster IDs to business segment names
    by inspecting centroid RFM values.

    High-Value  → low recency, high frequency & monetary
    At-Risk     → high recency, low frequency & monetary
    Regular     → mid-range on all three dimensions
    Occasional  → remaining cluster (typically infrequent buyers)
    """
    df = centroids_df.copy()

    # Composite score: lower recency and higher F/M are better
    df["value_score"] = (
        -df["Recency"] / (df["Recency"].max() + 1)
        + df["Frequency"] / (df["Frequency"].max() + 1)
        + df["Monetary"] / (df["Monetary"].max() + 1)
    )
    high_value_cluster = int(df["value_score"].idxmax())

    remaining = df.drop(index=high_value_cluster)
    # At-risk: highest recency relative to spend
    remaining = remaining.assign(
        risk_score=remaining["Recency"] / (remaining["Recency"].max() + 1)
        - remaining["Monetary"] / (remaining["Monetary"].max() + 1)
    )
    at_risk_cluster = int(remaining["risk_score"].idxmax())

    remaining2 = remaining.drop(index=at_risk_cluster)
    # Regular: closest to median profile across R, F, M
    medians = centroids_df[["Recency", "Frequency", "Monetary"]].median()
    remaining2 = remaining2.assign(
        regular_dist=(
            (remaining2["Recency"] - medians["Recency"]).abs()
            + (remaining2["Frequency"] - medians["Frequency"]).abs()
            + (remaining2["Monetary"] - medians["Monetary"]).abs()
        )
    )
    regular_cluster = int(remaining2["regular_dist"].idxmin())
    occasional_cluster = int(
        [c for c in centroids_df.index if c not in (high_value_cluster, at_risk_cluster, regular_cluster)][0]
    )

    return {
        high_value_cluster: "High-Value",
        regular_cluster: "Regular",
        occasional_cluster: "Occasional",
        at_risk_cluster: "At-Risk",
    }


def train_kmeans_spectrum(rfm: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, KMeans, dict]:
  """Normalize RFM, select K, train KMeans, label segments, and save artifacts."""
  print("\n" + "=" * 50)
  print(" PHASE 1 — KMeans Spectrum Detector (RFM Clustering)")
  print("=" * 50)

  feature_cols = ["Recency", "Frequency", "Monetary"]
  rfm_features = rfm[feature_cols].copy()

  # ── StandardScaler: equalize feature scales for distance-based clustering
  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm_features)
  rfm_scaled_df = pd.DataFrame(
      rfm_scaled,
      columns=["R_scaled", "F_scaled", "M_scaled"],
      index=rfm.index,
  )
  print("Scaled RFM means:", rfm_scaled_df.mean().round(4).tolist())
  print("Scaled RFM stds :", rfm_scaled_df.std().round(4).tolist())

  # ── Elbow method: plot WCSS vs K
  wcss = []
  for k in K_RANGE:
      km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
      km.fit(rfm_scaled)
      wcss.append(km.inertia_)

  plt.figure(figsize=(9, 5))
  plt.plot(list(K_RANGE), wcss, "bo-", linewidth=2, markersize=8)
  plt.xlabel("Number of Clusters (K)")
  plt.ylabel("WCSS (Inertia)")
  plt.title("Elbow Method — Optimal K Selection")
  plt.xticks(list(K_RANGE))
  plt.axvline(x=OPTIMAL_K, color="red", linestyle="--", label=f"Chosen K={OPTIMAL_K}")
  plt.legend()
  plt.tight_layout()
  plt.savefig(FIGURES_DIR / "elbow_curve.png", dpi=150)
  plt.close()
  print(f"Saved elbow curve: {FIGURES_DIR / 'elbow_curve.png'}")

  # ── Silhouette scores for each K
  silhouette_scores = []
  for k in K_RANGE:
      km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
      labels = km.fit_predict(rfm_scaled)
      score = silhouette_score(rfm_scaled, labels)
      silhouette_scores.append(score)
      print(f"  K={k:2d} | Silhouette Score = {score:.4f}")

  plt.figure(figsize=(9, 5))
  plt.plot(list(K_RANGE), silhouette_scores, "gs-", linewidth=2, markersize=8)
  plt.xlabel("Number of Clusters (K)")
  plt.ylabel("Silhouette Score")
  plt.title("Silhouette Scores for Different K Values")
  plt.xticks(list(K_RANGE))
  plt.tight_layout()
  plt.savefig(FIGURES_DIR / "silhouette_scores.png", dpi=150)
  plt.close()

  best_k = list(K_RANGE)[silhouette_scores.index(max(silhouette_scores))]
  print(f"Best K by Silhouette: {best_k} (using K={OPTIMAL_K} per project guide)")

  # ── Train final KMeans model
  kmeans_model = KMeans(
      n_clusters=OPTIMAL_K,
      init="k-means++",
      n_init=20,
      max_iter=500,
      tol=1e-4,
      random_state=RANDOM_STATE,
  )
  rfm = rfm.copy()
  rfm["Cluster"] = kmeans_model.fit_predict(rfm_scaled)

  centroids_original = pd.DataFrame(
      scaler.inverse_transform(kmeans_model.cluster_centers_),
      columns=feature_cols,
  )
  centroids_original.index.name = "Cluster"
  print("\nCluster Centroids (original scale):")
  print(centroids_original.round(2))

  cluster_labels = auto_label_clusters(centroids_original)
  rfm["Segment"] = rfm["Cluster"].map(cluster_labels)

  # ── Evaluation metrics
  labels = kmeans_model.labels_
  sil = silhouette_score(rfm_scaled, labels)
  dbi = davies_bouldin_score(rfm_scaled, labels)
  chi = calinski_harabasz_score(rfm_scaled, labels)
  print("\n" + "=" * 40)
  print(" CLUSTERING EVALUATION REPORT")
  print("=" * 40)
  print(f" K (clusters)       : {OPTIMAL_K}")
  print(f" WCSS (Inertia)     : {kmeans_model.inertia_:.2f}")
  print(f" Silhouette Score   : {sil:.4f}")
  print(f" Davies-Bouldin Idx : {dbi:.4f} (lower=better)")
  print(f" Calinski-Harabasz  : {chi:.2f} (higher=better)")
  print("=" * 40)
  print("\nSegment Distribution:")
  print(rfm["Segment"].value_counts())
  print("\nSegment Profiles (mean RFM):")
  print(rfm.groupby("Segment")[feature_cols].mean().round(2))

  # ── Visualizations: 2D scatter, 3D scatter, radar chart
  _plot_cluster_scatters(rfm)
  _plot_radar_chart(rfm)

  # ── Persist models (keys as strings for JSON compatibility in Streamlit)
  joblib.dump(scaler, MODELS_DIR / "rfm_scaler.pkl")
  joblib.dump(kmeans_model, MODELS_DIR / "kmeans_model.pkl")
  labels_json = {str(k): v for k, v in cluster_labels.items()}
  with open(MODELS_DIR / "cluster_labels.json", "w", encoding="utf-8") as f:
      json.dump(labels_json, f, indent=2)

  # Save RFM with segment assignments for reference
  rfm.to_csv(DATA_DIR / "rfm_with_segments.csv", index=False)
  print(f"\nSaved scaler, KMeans model, cluster labels: {MODELS_DIR}")

  return rfm, scaler, kmeans_model, cluster_labels


def _plot_cluster_scatters(rfm: pd.DataFrame) -> None:
    """2D scatter plots of customer segments in RFM space."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for segment, group in rfm.groupby("Segment"):
        color = SEGMENT_PALETTE.get(segment, "gray")
        axes[0].scatter(
            group["Recency"], group["Monetary"],
            c=color, label=segment, alpha=0.5, s=20, edgecolors="none",
        )
        axes[1].scatter(
            group["Frequency"], group["Monetary"],
            c=color, label=segment, alpha=0.5, s=20, edgecolors="none",
        )
    axes[0].set(xlabel="Recency (days)", ylabel="Monetary (£)", title="Recency vs Monetary")
    axes[1].set(xlabel="Frequency (orders)", ylabel="Monetary (£)", title="Frequency vs Monetary")
    for ax in axes:
        ax.legend()
    plt.suptitle("Customer Segments — RFM Scatter Plots", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_scatter.png", dpi=150)
    plt.close()


def _plot_radar_chart(rfm: pd.DataFrame) -> None:
    """Radar chart comparing average RFM profiles per segment."""
    profile = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
    norm = MinMaxScaler()
    profile_norm = pd.DataFrame(
        norm.fit_transform(profile),
        index=profile.index,
        columns=profile.columns,
    )
    categories = ["Recency\n(low=recent)", "Frequency", "Monetary"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for segment in profile_norm.index:
        values = profile_norm.loc[segment].tolist() + [profile_norm.loc[segment].tolist()[0]]
        ax.plot(angles, values, linewidth=2, label=segment, color=SEGMENT_PALETTE.get(segment))
        ax.fill(angles, values, alpha=0.1, color=SEGMENT_PALETTE.get(segment))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title("Segment Profiles — Radar Chart", fontsize=14)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "radar_chart.png", dpi=150)
    plt.close()


def train_collaborative_filtering() -> None:
    """Build item-based cosine similarity matrix and save recommendation data."""
    print("\n" + "=" * 50)
    print(" PHASE 2 — Item-Based Collaborative Filtering")
    print("=" * 50)

    df = pd.read_csv(CLEANED_PATH)
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Pivot: rows=customers, columns=products, values=total quantity
    product_matrix = (
        df.groupby(["CustomerID", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    print(f"Customer-Product matrix shape: {product_matrix.shape}")

    # Transpose so rows are products for pairwise similarity
    product_T = product_matrix.T
    similarity_matrix = cosine_similarity(product_T)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=product_T.index,
        columns=product_T.index,
    )
    print(f"Similarity matrix shape: {similarity_df.shape}")

    # Save full matrix for fast lookup in Streamlit
    similarity_df.to_pickle(MODELS_DIR / "similarity_matrix.pkl")

    # Also save compact top-10 recommendations per product (memory-efficient fallback)
    top_recs: dict[str, dict[str, float]] = {}
    for product in similarity_df.index:
        top_recs[product] = (
            similarity_df[product]
            .sort_values(ascending=False)
            .drop(product)
            .head(10)
            .to_dict()
        )
    with open(MODELS_DIR / "top_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(top_recs, f)

    print(f"Saved similarity matrix: {MODELS_DIR / 'similarity_matrix.pkl'}")
    print(f"Saved top-10 recs per product: {MODELS_DIR / 'top_recommendations.json'}")

    # Quick validation on a sample product
    sample_products = [p for p in similarity_df.index if "BEAKER" in p.upper()][:1]
    if sample_products:
        sample = sample_products[0]
        recs = (
            similarity_df[sample]
            .sort_values(ascending=False)
            .drop(sample)
            .head(5)
        )
        print(f"\nSample recommendations for '{sample}':")
        for i, (name, score) in enumerate(recs.items(), 1):
            print(f"  {i}. {name} (similarity: {score:.4f})")


def main() -> None:
    """Run the full training pipeline."""
    ensure_dirs()

    if not RFM_PATH.exists():
        raise FileNotFoundError(
            f"RFM features not found at {RFM_PATH}. "
            "Run CustomerEDA.ipynb first to prepare the data."
        )
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned retail data not found at {CLEANED_PATH}. "
            "Run CustomerEDA.ipynb first."
        )

    rfm = pd.read_csv(RFM_PATH)
    train_kmeans_spectrum(rfm)
    train_collaborative_filtering()

    print("\nTraining complete. Launch the GUI with:")
    print(f"   cd {BASE_DIR}")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
