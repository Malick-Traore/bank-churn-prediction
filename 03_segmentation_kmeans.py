"""
================================================================================
03_segmentation_kmeans.py
================================================================================
Project      : Bank Customer Analysis — Finance / Banking / Insurance
Author       : Malick Traore
Date         : 2026
Description  : Unsupervised customer segmentation pipeline using K-Means
               clustering. Loads the enriched dataset produced by
               02_kpis_analysis.py, determines the optimal number of clusters
               via the Elbow method and Silhouette scoring, fits the final
               K-Means model, profiles each cluster, assigns business-readable
               segment names, and exports a fully labelled dataset for Power BI.

Usage:
    python notebooks/03_segmentation_kmeans.py

Inputs:
    data/bank_churn_enriched.csv       (produced by 02_kpis_analysis.py)

Outputs:
    data/bank_churn_segmented.csv      (full dataset + cluster labels)
    data/cluster_profiles.csv          (one row per cluster, all KPIs)
    dashboard/screenshots/03_*.png     (all visualisations)
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent
DATA_IN       = ROOT_DIR / "data" / "bank_churn_enriched.csv"
DATA_SEG_OUT  = ROOT_DIR / "data" / "bank_churn_segmented.csv"
PROFILE_OUT   = ROOT_DIR / "data" / "cluster_profiles.csv"
SCREENSHOTS   = ROOT_DIR / "dashboard" / "screenshots"

CLUSTERING_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "active_member",
    "estimated_salary",
]

K_RANGE     = range(2, 11)   # Range of K values to evaluate
K_OPTIMAL   = 3              # Override after running elbow/silhouette analysis
RANDOM_STATE = 42
N_INIT       = 15            # Number of K-Means initialisations (higher = more stable)
MAX_ITER     = 500           # Maximum iterations per run

CLUSTER_PALETTE = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12",
                   "#9B59B6", "#1ABC9C", "#E67E22", "#34495E", "#EC407A"]

PLOT_DPI = 150

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  I. DATA LOADING & FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def load_enriched_data(path: Path) -> pd.DataFrame:
    """
    Load the enriched dataset produced by 02_kpis_analysis.py.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
    """
    if not path.exists():
        log.error(f"Input file not found: {path}")
        log.error("Run 02_kpis_analysis.py first to generate bank_churn_enriched.csv")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info(f"Dataset loaded  —  {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    return df


def build_feature_matrix(df: pd.DataFrame,
                         features: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract the feature matrix and apply StandardScaler normalisation.

    StandardScaler transforms each feature to zero mean and unit variance:
        z = (x - mean(x)) / std(x)

    This is mandatory before K-Means to prevent high-magnitude features
    (e.g. balance in € vs products_number in [1,4]) from dominating
    the Euclidean distance computation.

    Parameters
    ----------
    df       : pd.DataFrame
    features : list of column names to include in clustering

    Returns
    -------
    X_scaled : np.ndarray  (n_samples, n_features) — normalised matrix
    X_raw    : pd.DataFrame — original (unscaled) values for profiling
    """
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]

    if missing:
        log.warning(f"Features not found in dataset (skipped): {missing}")

    X_raw = df[available].copy()

    n_missing = X_raw.isnull().sum().sum()
    if n_missing > 0:
        log.warning(f"Imputing {n_missing} missing values with column medians")
        X_raw = X_raw.fillna(X_raw.median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    log.info(f"Feature matrix  :  {X_scaled.shape[0]:,} rows  x  {X_scaled.shape[1]} features")
    log.info(f"Features used   :  {available}")
    log.info(f"Post-scaling mean  (should be ~0) : {X_scaled.mean(axis=0).round(3)}")
    log.info(f"Post-scaling std   (should be ~1) : {X_scaled.std(axis=0).round(3)}")

    return X_scaled, X_raw


# ══════════════════════════════════════════════════════════════════════════════
#  II. OPTIMAL K SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_elbow_silhouette(X_scaled: np.ndarray,
                             k_range: range) -> pd.DataFrame:
    """
    For each K in k_range, fit K-Means and record:
      - Inertia (WCSS) : sum of squared distances to nearest centroid
      - Silhouette score : cohesion vs separation metric in [-1, +1]

    The optimal K is where:
      1. The elbow curve shows a sharp "bend" (diminishing return in inertia)
      2. The silhouette score is maximised

    Parameters
    ----------
    X_scaled : np.ndarray
    k_range  : range of K values to evaluate

    Returns
    -------
    pd.DataFrame with columns [k, inertia, silhouette]
    """
    records = []
    log.info(f"Evaluating K from {k_range.start} to {k_range.stop - 1} …")

    for k in k_range:
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
            max_iter=MAX_ITER,
        )
        labels = km.fit_predict(X_scaled)
        sil    = silhouette_score(X_scaled, labels)

        records.append({
            "k"          : k,
            "inertia"    : round(km.inertia_, 2),
            "silhouette" : round(sil, 4),
        })
        log.info(f"  K={k:2d}  |  Inertia={km.inertia_:,.1f}  |  Silhouette={sil:.4f}")

    results = pd.DataFrame(records)
    best_k  = results.loc[results["silhouette"].idxmax(), "k"]
    log.info(f"Best K by silhouette score : {best_k}")
    return results


def compute_silhouette_per_sample(X_scaled: np.ndarray,
                                  k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit K-Means with the given K and compute per-sample silhouette scores.
    Used to draw a silhouette diagram.

    Parameters
    ----------
    X_scaled : np.ndarray
    k        : int — number of clusters

    Returns
    -------
    labels         : np.ndarray of cluster assignments
    sample_scores  : np.ndarray of per-sample silhouette values
    """
    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )
    labels        = km.fit_predict(X_scaled)
    sample_scores = silhouette_samples(X_scaled, labels)
    return labels, sample_scores


# ══════════════════════════════════════════════════════════════════════════════
#  III. FINAL MODEL & CLUSTER PROFILING
# ══════════════════════════════════════════════════════════════════════════════

def fit_kmeans(X_scaled: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]:
    """
    Fit the final K-Means model with the chosen optimal K.

    Parameters
    ----------
    X_scaled : np.ndarray
    k        : int — optimal number of clusters

    Returns
    -------
    model  : fitted KMeans object
    labels : np.ndarray of cluster assignments (one integer per client)
    """
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )
    labels = model.fit_predict(X_scaled)

    log.info(f"K-Means fitted  :  K={k}")
    log.info(f"Final inertia   :  {model.inertia_:,.2f}")
    log.info(f"Iterations used :  {model.n_iter_}")
    log.info(
        "Cluster sizes   :  "
        + "  |  ".join(
            f"Cluster {i}: {(labels == i).sum():,}" for i in range(k)
        )
    )
    return model, labels


def build_cluster_profiles(df: pd.DataFrame,
                            labels: np.ndarray,
                            features: List[str]) -> pd.DataFrame:
    """
    Compute per-cluster descriptive statistics and business KPIs.

    For each cluster, computes:
      - Size (n_clients, pct_of_portfolio)
      - Mean and median of all clustering features
      - Churn rate, active rate, credit card rate
      - Mean balance, mean credit score, mean age, mean products

    Parameters
    ----------
    df       : pd.DataFrame (enriched, with churn column)
    labels   : np.ndarray of cluster assignments
    features : list of feature column names used in clustering

    Returns
    -------
    pd.DataFrame with one row per cluster.
    """
    df = df.copy()
    df["cluster"] = labels

    profile_cols  = [f for f in features if f in df.columns]
    business_cols = [c for c in ["churn", "credit_card", "active_member"]
                     if c in df.columns]

    agg_dict = {col: ["mean", "median"] for col in profile_cols}
    for col in business_cols:
        agg_dict[col] = "mean"

    profiles = df.groupby("cluster").agg(agg_dict)
    profiles.columns = ["_".join(c).strip("_") for c in profiles.columns]

    profiles["n_clients"]        = df.groupby("cluster").size()
    profiles["pct_of_portfolio"] = (profiles["n_clients"] / len(df) * 100).round(2)

    if "churn_mean" in profiles.columns:
        profiles["churn_rate_%"]   = (profiles["churn_mean"] * 100).round(2)
        profiles = profiles.drop(columns=["churn_mean"])
    if "active_member_mean" in profiles.columns:
        profiles["active_rate_%"]  = (profiles["active_member_mean"] * 100).round(2)
        profiles = profiles.drop(columns=["active_member_mean"])
    if "credit_card_mean" in profiles.columns:
        profiles["cr_card_rate_%"] = (profiles["credit_card_mean"] * 100).round(2)
        profiles = profiles.drop(columns=["credit_card_mean"])

    profiles = profiles.round(2).reset_index()

    print("\n" + "=" * 70)
    print("  CLUSTER PROFILES")
    print("=" * 70)
    print(profiles.to_string(index=False))
    print("=" * 70 + "\n")

    return profiles


def assign_segment_names(profiles: pd.DataFrame,
                         df: pd.DataFrame,
                         labels: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Automatically assign a business-readable segment name to each cluster
    based on its profile characteristics (balance level, age, activity, churn).

    The naming logic follows this priority:
      1. High churn rate  → prefix "A Risque"
      2. High balance + active → "Premium Actif"
      3. Low balance + inactive → "Passif Faible Valeur"
      4. Young clients → "Jeune Entrant"
      5. Default → "Profil Standard"

    Parameters
    ----------
    profiles : pd.DataFrame (output of build_cluster_profiles)
    df       : pd.DataFrame (enriched dataset)
    labels   : np.ndarray

    Returns
    -------
    profiles_named : pd.DataFrame with "segment_name" column
    df_labelled    : pd.DataFrame with "cluster" and "segment_name" columns
    """
    profiles = profiles.copy()
    df       = df.copy()
    df["cluster"] = labels

    global_churn   = df["churn"].mean() * 100
    global_balance = df["balance"].mean()
    global_age     = df["age"].mean()

    names = {}
    for _, row in profiles.iterrows():
        k = int(row["cluster"])

        churn_col   = "churn_rate_%"   if "churn_rate_%" in row.index else None
        balance_col = "balance_mean"   if "balance_mean" in row.index else None
        active_col  = "active_rate_%"  if "active_rate_%" in row.index else None
        age_col     = "age_mean"       if "age_mean" in row.index else None

        churn   = row[churn_col]   if churn_col   else 0
        balance = row[balance_col] if balance_col else 0
        active  = row[active_col]  if active_col  else 50
        age     = row[age_col]     if age_col     else 40

        if churn > global_churn * 1.5:
            name = f"C{k} — À Risque Élevé"
        elif balance > global_balance * 1.3 and active > 55:
            name = f"C{k} — Premium Actif"
        elif balance < global_balance * 0.5 and active < 45:
            name = f"C{k} — Passif Faible Valeur"
        elif age < global_age - 5:
            name = f"C{k} — Jeune Entrant"
        else:
            name = f"C{k} — Profil Standard"

        names[k] = name
        log.info(f"  Cluster {k} → '{name}'")

    profiles["segment_name"] = profiles["cluster"].map(names)
    df["segment_name"]       = df["cluster"].map(names)

    return profiles, df


# ══════════════════════════════════════════════════════════════════════════════
#  IV. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    """Save figure and release memory."""
    path = SCREENSHOTS / name
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved  →  {path.relative_to(ROOT_DIR)}")


def plot_elbow_silhouette(results: pd.DataFrame) -> None:
    """
    Side-by-side Elbow curve (inertia) and Silhouette score curve.
    Vertical dashed line marks the best K by silhouette score.
    """
    best_k = int(results.loc[results["silhouette"].idxmax(), "k"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 — Elbow curve
    axes[0].plot(results["k"], results["inertia"],
                 marker="o", color="#3498DB", linewidth=2, markersize=7)
    axes[0].axvline(best_k, color="#E74C3C", linestyle="--", linewidth=1.5,
                    label=f"K optimal = {best_k}")
    axes[0].fill_between(results["k"], results["inertia"],
                          alpha=0.08, color="#3498DB")
    axes[0].set_title("Méthode du Coude (Elbow Method)", fontsize=13, pad=10)
    axes[0].set_xlabel("Nombre de clusters K", fontsize=11)
    axes[0].set_ylabel("Inertie (WCSS)", fontsize=11)
    axes[0].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Panel 2 — Silhouette score
    axes[1].plot(results["k"], results["silhouette"],
                 marker="s", color="#2ECC71", linewidth=2, markersize=7)
    axes[1].axvline(best_k, color="#E74C3C", linestyle="--", linewidth=1.5,
                    label=f"K optimal = {best_k}")
    for _, row in results.iterrows():
        axes[1].annotate(
            f"{row['silhouette']:.3f}",
            (row["k"], row["silhouette"]),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=8, color="#2C3E50",
        )
    axes[1].set_title("Score de Silhouette par K", fontsize=13, pad=10)
    axes[1].set_xlabel("Nombre de clusters K", fontsize=11)
    axes[1].set_ylabel("Silhouette Score (max = meilleur)", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Sélection du Nombre Optimal de Clusters",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "03_elbow_silhouette.png")


def plot_silhouette_diagram(X_scaled: np.ndarray,
                             labels: np.ndarray,
                             sample_scores: np.ndarray,
                             k: int) -> None:
    """
    Draw a silhouette diagram: horizontal bars per sample, grouped by cluster,
    sorted by silhouette value. Shows cohesion and separation of each cluster.
    The red dashed line marks the global average silhouette score.
    """
    global_score = silhouette_score(X_scaled, labels)
    fig, ax      = plt.subplots(figsize=(10, 6))
    y_lower      = 10

    for i in range(k):
        cluster_scores = np.sort(sample_scores[labels == i])
        size_i         = len(cluster_scores)
        y_upper        = y_lower + size_i

        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_scores,
            facecolor=color, edgecolor=color, alpha=0.85,
        )
        ax.text(-0.05, y_lower + size_i / 2, f"C{i}",
                ha="right", va="center", fontsize=10, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(global_score, color="#E74C3C", linestyle="--", linewidth=2,
               label=f"Score global = {global_score:.3f}")
    ax.set_xlim(-0.2, 1.0)
    ax.set_xlabel("Score de Silhouette individuel", fontsize=11)
    ax.set_ylabel("Clients (groupés par cluster)", fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"Diagramme de Silhouette — K={k}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    _save(fig, "03_silhouette_diagram.png")


def plot_cluster_heatmap(profiles: pd.DataFrame, features: List[str]) -> None:
    """
    Heatmap of cluster profiles: rows = clusters, columns = features.
    Cell colour = min-max normalised value (relative intensity per feature).
    Cell text = actual mean value.

    This is the key chart for interpreting what each cluster represents.
    """
    display_features = [f for f in features if f + "_mean" in profiles.columns]
    raw_cols         = [f + "_mean" for f in display_features]

    subset = profiles.set_index("segment_name")[raw_cols].copy()
    subset.columns = [c.replace("_mean", "").replace("_", " ").title()
                      for c in subset.columns]

    subset_norm = (subset - subset.min()) / (subset.max() - subset.min())

    fig, ax = plt.subplots(figsize=(12, max(3, len(profiles) * 1.2)))
    sns.heatmap(
        subset_norm,
        annot=subset.round(1),
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.8,
        ax=ax,
        cbar_kws={"label": "Intensité relative (0→1)", "shrink": 0.75},
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_title(
        "Profils des Clusters — Valeurs moyennes réelles (couleur = intensité relative)",
        fontsize=12, pad=12,
    )
    plt.tight_layout()
    _save(fig, "03_cluster_heatmap.png")


def plot_churn_by_cluster(df_labelled: pd.DataFrame) -> None:
    """
    Bar chart of churn rate per cluster, with global average reference line.
    Bars are coloured red if above global average, blue otherwise.
    """
    global_churn  = df_labelled["churn"].mean() * 100
    churn_cluster = (
        df_labelled.groupby("segment_name")["churn"].mean() * 100
    ).sort_values(ascending=False)

    colors = [
        "#E74C3C" if v > global_churn else "#3498DB"
        for v in churn_cluster.values
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(churn_cluster)), churn_cluster.values,
        color=colors, edgecolor="white", width=0.6,
    )
    ax.axhline(global_churn, color="#7F8C8D", linestyle="--", linewidth=1.8,
               label=f"Moyenne globale : {global_churn:.1f}%")

    for bar, val in zip(bars, churn_cluster.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(range(len(churn_cluster)))
    ax.set_xticklabels(churn_cluster.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Taux de Churn (%)", fontsize=11)
    ax.set_ylim(0, churn_cluster.max() * 1.25)
    ax.set_title("Taux de Churn par Segment Client", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _save(fig, "03_churn_by_cluster.png")


def plot_pca_scatter(X_scaled: np.ndarray,
                     labels: np.ndarray,
                     k: int) -> None:
    """
    Project the high-dimensional feature space onto 2 principal components (PCA)
    and plot each client as a coloured dot according to its cluster.

    PCA finds the two directions of maximum variance in the data, allowing
    visual inspection of cluster separation in 2D.

    A 20% random sample is used to avoid overplotting.

    Parameters
    ----------
    X_scaled : np.ndarray — normalised feature matrix (n_samples, n_features)
    labels   : np.ndarray — cluster assignments
    k        : int — number of clusters
    """
    pca        = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca      = pca.fit_transform(X_scaled)
    var_ratio  = pca.explained_variance_ratio_

    sample_idx = np.random.default_rng(RANDOM_STATE).choice(
        len(X_pca), size=int(len(X_pca) * 0.3), replace=False,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(k):
        mask = labels[sample_idx] == i
        ax.scatter(
            X_pca[sample_idx][mask, 0],
            X_pca[sample_idx][mask, 1],
            c=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)],
            alpha=0.5, s=18, label=f"Cluster {i}",
            edgecolors="none",
        )

    ax.set_xlabel(
        f"Composante Principale 1  ({var_ratio[0]*100:.1f}% de variance expliquée)",
        fontsize=10,
    )
    ax.set_ylabel(
        f"Composante Principale 2  ({var_ratio[1]*100:.1f}% de variance expliquée)",
        fontsize=10,
    )
    ax.set_title(
        f"Visualisation PCA des Clusters (K={k})  —  "
        f"{sum(var_ratio)*100:.1f}% de variance totale expliquée",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, "03_pca_scatter.png")


def plot_radar_profiles(profiles: pd.DataFrame, features: List[str]) -> None:
    """
    Radar chart (spider chart) comparing normalised cluster profiles
    across all clustering features simultaneously.

    Each cluster is drawn as a filled polygon on the same axes.
    Useful for comparing the "shape" of each cluster at a glance.
    """
    display_features = [f for f in features if f + "_mean" in profiles.columns]
    raw_cols         = [f + "_mean" for f in display_features]
    labels_feat      = [f.replace("_", " ").title() for f in display_features]

    subset = profiles.set_index("segment_name")[raw_cols].copy()

    subset_norm = (subset - subset.min()) / (subset.max() - subset.min())
    subset_norm = subset_norm.fillna(0)

    N     = len(display_features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})

    for i, (name, row) in enumerate(subset_norm.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        color  = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_feat, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Profils Radar des Clusters (normalisé 0→1)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "03_radar_profiles.png")


# ══════════════════════════════════════════════════════════════════════════════
#  V. EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_outputs(df_labelled: pd.DataFrame,
                   profiles: pd.DataFrame) -> None:
    """
    Export labelled dataset and cluster profiles for Power BI.

    Files produced
    --------------
    bank_churn_segmented.csv : full dataset with cluster and segment_name columns
    cluster_profiles.csv     : one row per cluster, all aggregated KPIs

    Parameters
    ----------
    df_labelled : pd.DataFrame
    profiles    : pd.DataFrame
    """
    df_labelled.to_csv(DATA_SEG_OUT, index=False)
    log.info(f"Segmented dataset →  {DATA_SEG_OUT.relative_to(ROOT_DIR)}")

    profiles.to_csv(PROFILE_OUT, index=False)
    log.info(f"Cluster profiles  →  {PROFILE_OUT.relative_to(ROOT_DIR)}")

    log.info("Power BI: import bank_churn_segmented.csv as main table")
    log.info("Power BI: import cluster_profiles.csv for segment-level visuals")


# ══════════════════════════════════════════════════════════════════════════════
#  VI. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Full K-Means segmentation pipeline:
      1. Load enriched dataset
      2. Build and normalise feature matrix (StandardScaler)
      3. Evaluate K from 2 to 10 (Elbow + Silhouette)
      4. Fit final K-Means with optimal K
      5. Build cluster profiles
      6. Assign business-readable segment names
      7. Produce all visualisations (elbow, silhouette diagram, heatmap,
         churn by cluster, PCA scatter, radar)
      8. Export outputs for Power BI
    """
    log.info("=" * 60)
    log.info("  PIPELINE START — 03_segmentation_kmeans.py")
    log.info("=" * 60)

    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.family"] = "DejaVu Sans"
    np.random.seed(RANDOM_STATE)

    # ── 1. Load ───────────────────────────────────────────────────────────
    df = load_enriched_data(DATA_IN)

    # ── 2. Feature matrix ─────────────────────────────────────────────────
    X_scaled, X_raw = build_feature_matrix(df, CLUSTERING_FEATURES)

    # ── 3. K selection ────────────────────────────────────────────────────
    results = compute_elbow_silhouette(X_scaled, K_RANGE)
    plot_elbow_silhouette(results)

    # ── 4. Fit final model ────────────────────────────────────────────────
    k = K_OPTIMAL
    log.info(f"Using K={k} (set in K_OPTIMAL constant — adjust based on elbow chart)")
    model, labels = fit_kmeans(X_scaled, k)

    # ── 5. Silhouette diagram ─────────────────────────────────────────────
    _, sample_scores = compute_silhouette_per_sample(X_scaled, k)
    plot_silhouette_diagram(X_scaled, labels, sample_scores, k)

    # ── 6. Cluster profiles ───────────────────────────────────────────────
    profiles = build_cluster_profiles(df, labels, CLUSTERING_FEATURES)

    # ── 7. Segment naming ─────────────────────────────────────────────────
    profiles, df_labelled = assign_segment_names(profiles, df, labels)

    # ── 8. Visualisations ─────────────────────────────────────────────────
    log.info("Generating visualisations …")
    plot_cluster_heatmap(profiles, CLUSTERING_FEATURES)
    plot_churn_by_cluster(df_labelled)
    plot_pca_scatter(X_scaled, labels, k)
    plot_radar_profiles(profiles, CLUSTERING_FEATURES)

    # ── 9. Export ─────────────────────────────────────────────────────────
    export_outputs(df_labelled, profiles)

    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
