"""
================================================================================
01_cleaning_eda.py
================================================================================
Project      : Bank Customer Analysis — Finance / Banking / Insurance
Author       : Malick Traore
Date         : 2026
Description  : Data cleaning and exploratory data analysis (EDA)
               pipeline for the Bank Customer Churn dataset.
               Produces a cleaned CSV and a full set of EDA visualizations.

Usage:
    python notebooks/01_cleaning_eda.py

Outputs:
    data/bank_churn_clean.csv
    dashboard/screenshots/01_*.png
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import logging
from pathlib import Path
from typing import Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent
DATA_RAW       = ROOT_DIR / "data" / "bank_churn.csv"
DATA_CLEAN_OUT = ROOT_DIR / "data" / "bank_churn_clean.csv"
SCREENSHOTS    = ROOT_DIR / "dashboard" / "screenshots"

COLUMNS_TO_DROP = ["customer_id"]

NUMERIC_FEATURES = [
    "credit_score", "age", "tenure",
    "balance", "products_number", "estimated_salary",
]

CATEGORICAL_FEATURES = [
    "country", "gender", "credit_card",
    "active_member", "churn",
]

IQR_MULTIPLIER = 1.5   # Standard Tukey fence
PLOT_DPI       = 150
PALETTE        = "muted"

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  I. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame and print a basic shape report.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset.
    """
    if not path.exists():
        log.error(f"File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info(f"Dataset loaded  —  {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    log.info(f"Columns: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  II. DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column quality summary: dtype, missing count/%, unique values,
    and basic descriptive stats for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Quality report table, also printed to stdout.
    """
    report = pd.DataFrame({
        "dtype"       : df.dtypes,
        "missing"     : df.isnull().sum(),
        "missing_%"   : (df.isnull().mean() * 100).round(2),
        "unique"      : df.nunique(),
        "min"         : df.min(numeric_only=True),
        "max"         : df.max(numeric_only=True),
        "mean"        : df.mean(numeric_only=True).round(2),
        "std"         : df.std(numeric_only=True).round(2),
    })

    n_dup = df.duplicated().sum()
    log.info(f"Duplicate rows  : {n_dup}")

    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        log.info("Missing values  : none ✓")
    else:
        log.warning(f"Missing values  : {total_missing} cells across "
                    f"{(df.isnull().sum() > 0).sum()} column(s)")

    print("\n" + "=" * 70)
    print("  DATA QUALITY REPORT")
    print("=" * 70)
    print(report.to_string())
    print("=" * 70 + "\n")
    return report


# ══════════════════════════════════════════════════════════════════════════════
#  III. CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations:
      1. Drop irrelevant identifier columns
      2. Standardise column names to snake_case
      3. Remove exact duplicate rows
      4. Coerce numeric columns to correct dtypes
      5. Drop rows where the target variable (churn) is null

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for analysis.
    """
    df_c = df.copy()
    original_shape = df_c.shape

    # 1. Drop identifier columns (case-insensitive match)
    cols_present = [
        c for c in df_c.columns
        if c in COLUMNS_TO_DROP or c.lower() in [x.lower() for x in COLUMNS_TO_DROP]
    ]
    df_c = df_c.drop(columns=cols_present)
    log.info(f"Dropped columns : {cols_present}")

    # 2. Standardise column names
    df_c.columns = (
        df_c.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    # 3. Remove duplicates
    before = len(df_c)
    df_c = df_c.drop_duplicates()
    removed = before - len(df_c)
    if removed:
        log.warning(f"Removed {removed} duplicate row(s)")
    else:
        log.info("Duplicates      : none ✓")

    # 4. Coerce numeric types
    for col in NUMERIC_FEATURES:
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

    # 5. Drop rows with null target
    if "churn" in df_c.columns:
        null_target = df_c["churn"].isnull().sum()
        if null_target:
            log.warning(f"Dropping {null_target} row(s) with null churn")
            df_c = df_c.dropna(subset=["churn"])

    log.info(
        f"Cleaning done   :  {original_shape} → {df_c.shape}  "
        f"({original_shape[0] - df_c.shape[0]} rows removed)"
    )
    return df_c


# ══════════════════════════════════════════════════════════════════════════════
#  IV. OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers_iqr(df: pd.DataFrame,
                        columns: list,
                        k: float = IQR_MULTIPLIER) -> pd.DataFrame:
    """
    Detect outliers using the Tukey IQR fence method.

    A value x is flagged as an outlier if:
        x  <  Q1 - k * IQR   OR   x  >  Q3 + k * IQR
    where IQR = Q3 - Q1, k = 1.5 by default (standard), k = 3.0 = extreme.

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list of numeric column names to inspect
    k       : float, fence multiplier (default 1.5)

    Returns
    -------
    pd.DataFrame
        Summary table: column, Q1, Q3, IQR, lower fence, upper fence,
        outlier count, outlier %.
    """
    records = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr    = q3 - q1
        lower  = q1 - k * iqr
        upper  = q3 + k * iqr
        n_out  = ((series < lower) | (series > upper)).sum()
        records.append({
            "column"       : col,
            "Q1"           : round(q1, 2),
            "Q3"           : round(q3, 2),
            "IQR"          : round(iqr, 2),
            "lower_fence"  : round(lower, 2),
            "upper_fence"  : round(upper, 2),
            "n_outliers"   : n_out,
            "outlier_%"    : round(n_out / len(series) * 100, 2),
        })

    summary = pd.DataFrame(records).set_index("column")
    print("\n  OUTLIER REPORT (IQR method, k=" + str(k) + ")")
    print(summary.to_string())
    print()
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  V. FEATURE ENGINEERING — Derived columns for EDA
# ══════════════════════════════════════════════════════════════════════════════

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add analytical columns that enrich EDA without modifying raw variables.

    New columns created:
      - age_group       : categorical age bands (18-30, 31-40, ...)
      - credit_segment  : 5-tier credit score bracket (banking standard)
      - balance_group   : balance quartile label
      - tenure_group    : tenure band (0-2, 3-5, 6-8, 9-10 years)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[17, 30, 40, 50, 60, 120],
            labels=["18-30", "31-40", "41-50", "51-60", "60+"],
        )

    if "credit_score" in df.columns:
        df["credit_segment"] = pd.cut(
            df["credit_score"],
            bins=[299, 579, 669, 739, 799, 900],
            labels=["Très Faible (<580)", "Faible (580-669)",
                    "Correct (670-739)", "Bon (740-799)", "Excellent (800+)"],
        )

    if "balance" in df.columns:
        df["balance_group"] = pd.qcut(
            df["balance"].replace(0, np.nan),
            q=4,
            labels=["Q1 — Bas", "Q2 — Moyen-bas", "Q3 — Moyen-haut", "Q4 — Haut"],
            duplicates="drop",
        )

    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[-1, 2, 5, 8, 10],
            labels=["0-2 ans", "3-5 ans", "6-8 ans", "9-10 ans"],
        )

    log.info("Derived features added: age_group, credit_segment, balance_group, tenure_group")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  VI. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to screenshots directory and close it."""
    path = SCREENSHOTS / name
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved  →  {path.relative_to(ROOT_DIR)}")


def plot_outlier_boxplots(df: pd.DataFrame, columns: list) -> None:
    """
    Grid of boxplots for outlier visualisation, one panel per numeric column.
    """
    cols_present = [c for c in columns if c in df.columns]
    n = len(cols_present)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_present):
        bp = ax.boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="#AED6F1", color="#2471A3"),
            medianprops=dict(color="#E74C3C", linewidth=2),
            whiskerprops=dict(color="#2471A3"),
            capprops=dict(color="#2471A3"),
            flierprops=dict(marker="o", color="#E74C3C", alpha=0.4, markersize=3),
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=10, pad=8)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Détection des Valeurs Aberrantes — Boxplots (Méthode IQR de Tukey)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "01_outliers_boxplot.png")


def plot_numeric_distributions(df: pd.DataFrame, columns: list) -> None:
    """
    Grid of histograms for all numeric features, with mean and median lines.
    """
    cols_present = [c for c in columns if c in df.columns]
    n = len(cols_present)
    ncols = 3
    nrows = -(-n // ncols)   # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols_present):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=35, color="#5DADE2", edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="#E74C3C", lw=1.8, ls="--",
                   label=f"Moyenne : {data.mean():,.1f}")
        ax.axvline(data.median(), color="#27AE60", lw=1.8, ls="-.",
                   label=f"Médiane : {data.median():,.1f}")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.1f}"
        ))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution des Variables Numériques", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "01_numeric_distributions.png")


def plot_categorical_distributions(df: pd.DataFrame, columns: list) -> None:
    """
    Horizontal bar charts for all categorical/binary features.
    """
    cols_present = [c for c in columns if c in df.columns]
    n = len(cols_present)
    palette = ["#5DADE2", "#E59866", "#82E0AA", "#F1948A", "#BB8FCE", "#F9E79F"]

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col, color in zip(axes, cols_present, palette):
        counts = df[col].value_counts()
        pcts   = counts / counts.sum() * 100
        bars   = ax.barh(
            counts.index.astype(str)[::-1],
            counts.values[::-1],
            color=color, edgecolor="white",
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=11, pad=8)
        ax.set_xlabel("Effectif")
        for bar, val, pct in zip(bars, counts.values[::-1], pcts.values[::-1]):
            ax.text(
                bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)",
                va="center", fontsize=8.5,
            )
        ax.set_xlim(0, counts.max() * 1.25)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Distribution des Variables Catégorielles", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "01_categorical_distributions.png")


def plot_correlation_matrix(df: pd.DataFrame, columns: list) -> None:
    """
    Lower-triangle Pearson correlation heatmap.
    Only includes numeric columns present in the DataFrame.
    """
    cols_present = [c for c in columns if c in df.columns and
                    pd.api.types.is_numeric_dtype(df[c])]

    corr = df[cols_present].corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        annot_kws={"size": 9},
    )
    ax.set_title("Matrice de Corrélation de Pearson", fontsize=13,
                 fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, "01_correlation_matrix.png")

    # Print sorted correlations with churn
    if "churn" in corr.columns:
        print("\n  CORRELATIONS WITH TARGET — churn (sorted)")
        print(
            corr["churn"]
            .drop("churn")
            .sort_values(key=abs, ascending=False)
            .round(4)
            .to_string()
        )
        print()


def plot_churn_by_derived_features(df: pd.DataFrame) -> None:
    """
    4-panel chart showing churn rate broken down by each derived feature:
    age_group, credit_segment, balance_group, tenure_group.
    """
    derived_cols = ["age_group", "credit_segment", "balance_group", "tenure_group"]
    available    = [c for c in derived_cols if c in df.columns]
    if not available:
        log.warning("No derived features found — run add_derived_features() first.")
        return

    n    = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    global_churn = df["churn"].mean() * 100

    for ax, col in zip(axes, available):
        churn_rate = (
            df.groupby(col, observed=True)["churn"]
            .mean()
            .mul(100)
        )
        colors = ["#E74C3C" if v > global_churn else "#3498DB"
                  for v in churn_rate.values]
        bars = ax.bar(
            range(len(churn_rate)),
            churn_rate.values,
            color=colors,
            edgecolor="white",
            width=0.65,
        )
        ax.axhline(global_churn, color="gray", ls="--", lw=1.5,
                   label=f"Global avg: {global_churn:.1f}%")
        ax.set_xticks(range(len(churn_rate)))
        ax.set_xticklabels(churn_rate.index.astype(str), rotation=25, ha="right", fontsize=8)
        ax.set_title(f"Taux de Churn par {col.replace('_', ' ').title()}",
                     fontsize=11, pad=8)
        ax.set_ylabel("Taux de Churn (%)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, churn_rate.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha="center", fontsize=8, fontweight="bold",
            )

    fig.suptitle("Churn Rate by Derived Customer Segments",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "01_churn_by_segments.png")


# ══════════════════════════════════════════════════════════════════════════════
#  VII. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Full EDA pipeline:
      1. Load raw data
      2. Data quality report
      3. Clean data
      4. Outlier detection
      5. Add derived features
      6. Produce all visualisations
      7. Export clean dataset
    """
    log.info("═" * 60)
    log.info("  PIPELINE START — 01_cleaning_eda.py")
    log.info("═" * 60)

    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams["font.family"] = "DejaVu Sans"

    # ── 1. Load ────────────────────────────────────────────────────────────
    df_raw = load_data(DATA_RAW)

    # ── 2. Quality report ──────────────────────────────────────────────────
    quality_report(df_raw)

    # ── 3. Clean ───────────────────────────────────────────────────────────
    df_clean = clean_data(df_raw)

    # ── 4. Outliers ────────────────────────────────────────────────────────
    cols_num = [c for c in NUMERIC_FEATURES if c in df_clean.columns]
    detect_outliers_iqr(df_clean, cols_num)
    plot_outlier_boxplots(df_clean, cols_num)

    # ── 5. Derived features ────────────────────────────────────────────────
    df_enriched = add_derived_features(df_clean)

    # ── 6. Visualisations ─────────────────────────────────────────────────
    log.info("Generating visualisations …")
    cols_cat = [c for c in CATEGORICAL_FEATURES if c in df_enriched.columns]

    plot_numeric_distributions(df_enriched, cols_num)
    plot_categorical_distributions(df_enriched, cols_cat)
    plot_correlation_matrix(df_enriched, cols_num + ["churn"])
    plot_churn_by_derived_features(df_enriched)

    # ── 7. Export ──────────────────────────────────────────────────────────
    df_enriched.to_csv(DATA_CLEAN_OUT, index=False)
    log.info(f"Clean dataset exported  →  {DATA_CLEAN_OUT.relative_to(ROOT_DIR)}")
    log.info(f"Final shape             :  {df_enriched.shape}")
    log.info("═" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
