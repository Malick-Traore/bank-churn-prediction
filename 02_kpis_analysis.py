"""
================================================================================
02_kpis_analysis.py
================================================================================
Project      : Bank Customer Analysis — Finance / Banking / Insurance
Author       : Malick Traore
Date         : 2026
Description  : KPI computation and business-oriented visualisation pipeline.
               Loads the cleaned dataset produced by 01_cleaning_eda.py and
               produces all key performance indicators, multi-dimensional churn
               breakdowns, balance / credit / product analyses, and a final
               export ready for Power BI ingestion.

Usage:
    python notebooks/02_kpis_analysis.py

Inputs:
    data/bank_churn_clean.csv          (produced by 01_cleaning_eda.py)

Outputs:
    data/bank_churn_enriched.csv       (enriched dataset for Power BI)
    data/kpis_summary.csv              (flat KPI table)
    data/kpis_by_country.csv           (KPIs aggregated per country)
    dashboard/screenshots/02_*.png     (all visualisations)
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import sys
import logging
from pathlib import Path
from typing import Dict, List

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
DATA_IN     = ROOT_DIR / "data" / "bank_churn_clean.csv"
DATA_OUT    = ROOT_DIR / "data" / "bank_churn_enriched.csv"
KPI_OUT     = ROOT_DIR / "data" / "kpis_summary.csv"
COUNTRY_OUT = ROOT_DIR / "data" / "kpis_by_country.csv"
SCREENSHOTS = ROOT_DIR / "dashboard" / "screenshots"

CREDIT_BINS   = [299, 579, 669, 739, 799, 900]
CREDIT_LABELS = [
    "Très Faible (<580)", "Faible (580-669)",
    "Correct (670-739)", "Bon (740-799)", "Excellent (800+)",
]

AGE_BINS   = [17, 30, 40, 50, 60, 120]
AGE_LABELS = ["18-30", "31-40", "41-50", "51-60", "60+"]

PALETTE_COUNTRY  = ["#3498DB", "#E74C3C", "#2ECC71"]
PALETTE_BINARY   = ["#E74C3C", "#2ECC71"]
PALETTE_PRODUCTS = ["#1ABC9C", "#F39C12", "#E74C3C", "#8E44AD"]
PALETTE_AGE      = ["#5DADE2", "#48C9B0", "#F0B27A", "#EC7063", "#AF7AC5"]

PLOT_DPI = 150

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  I. DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def load_clean_data(path: Path) -> pd.DataFrame:
    """
    Load the cleaned dataset produced by 01_cleaning_eda.py and validate
    that all expected columns are present before continuing.

    Parameters
    ----------
    path : Path
        Path to bank_churn_clean.csv.

    Returns
    -------
    pd.DataFrame
        Validated dataset.

    Raises
    ------
    SystemExit
        If the file is missing or required columns are absent.
    """
    if not path.exists():
        log.error(f"Input file not found: {path}")
        log.error("Run 01_cleaning_eda.py first to generate bank_churn_clean.csv")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info(f"Dataset loaded  —  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    required = [
        "credit_score", "country", "age", "tenure", "balance",
        "products_number", "estimated_salary",
        "credit_card", "active_member", "churn",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        log.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    log.info("Column validation passed ✓")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  II. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived categorical columns needed for KPI breakdowns.
    All new columns are non-destructive (original columns untouched).

    New columns
    -----------
    age_group      : age bands (18-30, 31-40, 41-50, 51-60, 60+)
    credit_segment : 5-tier banking credit score bracket
    balance_tier   : labelled quartile of non-zero balances
    tenure_band    : tenure grouped in 3-year bands
    high_value     : boolean flag — balance in top 25% AND active member
    at_risk        : boolean flag — churned OR (inactive AND products > 2)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"], bins=AGE_BINS, labels=AGE_LABELS,
    )

    df["credit_segment"] = pd.cut(
        df["credit_score"], bins=CREDIT_BINS, labels=CREDIT_LABELS,
    )

    df["balance_tier"] = pd.qcut(
        df["balance"].replace(0, np.nan), q=4,
        labels=["Q1 — Faible", "Q2 — Moyen-bas", "Q3 — Moyen-haut", "Q4 — Élevé"],
        duplicates="drop",
    )

    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[-1, 2, 5, 8, 10],
        labels=["0-2 ans", "3-5 ans", "6-8 ans", "9-10 ans"],
    )

    balance_q75 = df["balance"].quantile(0.75)
    df["high_value"] = (
        (df["balance"] >= balance_q75) & (df["active_member"] == 1)
    ).astype(int)

    df["at_risk"] = (
        (df["churn"] == 1) |
        ((df["active_member"] == 0) & (df["products_number"] > 2))
    ).astype(int)

    log.info("Features enriched: age_group, credit_segment, balance_tier, "
             "tenure_band, high_value, at_risk")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  III. KPI COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_global_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the full set of global portfolio KPIs.

    KPIs computed
    -------------
    - Total clients, churners, retained
    - Churn rate (%), retention rate (%)
    - Average and median balance
    - Average credit score
    - Average tenure
    - Average number of products
    - Active member rate (%)
    - Credit card penetration rate (%)
    - Average estimated salary
    - High-value client share (%)
    - At-risk client share (%)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Flat dictionary of {kpi_name: value}.
    """
    n = len(df)
    n_churn = int(df["churn"].sum())

    kpis = {
        "total_clients"              : n,
        "total_churners"             : n_churn,
        "total_retained"             : n - n_churn,
        "churn_rate_%"               : round(df["churn"].mean() * 100, 2),
        "retention_rate_%"           : round((1 - df["churn"].mean()) * 100, 2),
        "balance_mean_eur"           : round(df["balance"].mean(), 2),
        "balance_median_eur"         : round(df["balance"].median(), 2),
        "balance_std_eur"            : round(df["balance"].std(), 2),
        "credit_score_mean"          : round(df["credit_score"].mean(), 1),
        "credit_score_median"        : round(df["credit_score"].median(), 1),
        "tenure_mean_years"          : round(df["tenure"].mean(), 2),
        "products_per_client_mean"   : round(df["products_number"].mean(), 3),
        "active_member_rate_%"       : round(df["active_member"].mean() * 100, 1),
        "credit_card_penetration_%"  : round(df["credit_card"].mean() * 100, 1),
        "estimated_salary_mean_eur"  : round(df["estimated_salary"].mean(), 2),
        "high_value_clients_%"       : round(df["high_value"].mean() * 100, 1),
        "at_risk_clients_%"          : round(df["at_risk"].mean() * 100, 1),
    }

    print("\n" + "=" * 60)
    print("  GLOBAL KPIs DASHBOARD")
    print("=" * 60)
    for name, value in kpis.items():
        label = name.replace("_", " ").title()
        if isinstance(value, float):
            print(f"  {label:<38} {value:>12,.2f}")
        else:
            print(f"  {label:<38} {value:>12,}")
    print("=" * 60 + "\n")

    return kpis


def compute_kpis_by_group(df: pd.DataFrame,
                          groupby_col: str,
                          observed: bool = True) -> pd.DataFrame:
    """
    Aggregate KPIs by a categorical grouping column.

    Metrics per group
    -----------------
    n_clients, churn_rate_%, retention_rate_%, balance_mean, balance_median,
    credit_score_mean, tenure_mean, products_mean, active_rate_%,
    credit_card_rate_%

    Parameters
    ----------
    df         : pd.DataFrame
    groupby_col: str — column to group by (country, age_group, credit_segment…)
    observed   : bool — passed to groupby for Categorical columns

    Returns
    -------
    pd.DataFrame
        One row per group, columns = metrics.
    """
    g = df.groupby(groupby_col, observed=observed)

    result = pd.DataFrame({
        "n_clients"          : g["churn"].count(),
        "churn_rate_%"       : (g["churn"].mean() * 100).round(2),
        "retention_rate_%"   : ((1 - g["churn"].mean()) * 100).round(2),
        "balance_mean"       : g["balance"].mean().round(0),
        "balance_median"     : g["balance"].median().round(0),
        "credit_score_mean"  : g["credit_score"].mean().round(1),
        "tenure_mean"        : g["tenure"].mean().round(2),
        "products_mean"      : g["products_number"].mean().round(3),
        "active_rate_%"      : (g["active_member"].mean() * 100).round(1),
        "credit_card_rate_%" : (g["credit_card"].mean() * 100).round(1),
    }).reset_index()

    result["pct_of_portfolio"] = (
        result["n_clients"] / result["n_clients"].sum() * 100
    ).round(2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  IV. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to screenshots dir and release memory."""
    path = SCREENSHOTS / name
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved  →  {path.relative_to(ROOT_DIR)}")


def _add_bar_labels(ax: plt.Axes,
                    values,
                    fmt: str = "{:.1f}%",
                    offset_ratio: float = 0.01,
                    fontsize: int = 9) -> None:
    """
    Annotate each bar in an Axes with its numeric value.

    Parameters
    ----------
    ax           : matplotlib Axes containing bar patches
    values       : iterable of numeric values matching the bars
    fmt          : Python format string (default "{:.1f}%")
    offset_ratio : fraction of the max value used as vertical offset
    fontsize     : label font size
    """
    max_val = max(values) if max(values) != 0 else 1
    offset  = max_val * offset_ratio
    for bar, val in zip(ax.patches, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(val),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
        )


def plot_churn_multidimensional(df: pd.DataFrame) -> None:
    """
    4-panel figure : churn rate broken down by country, gender,
    number of products, and age group.
    Each bar is compared against the global churn average (dashed line).
    """
    global_churn = df["churn"].mean() * 100
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    breakdowns = [
        ("country",          "Churn par Pays",                PALETTE_COUNTRY),
        ("gender",           "Churn par Genre",               PALETTE_BINARY),
        ("products_number",  "Churn par Nombre de Produits",  PALETTE_PRODUCTS),
        ("age_group",        "Churn par Tranche d'Âge",       PALETTE_AGE),
    ]

    for ax, (col, title, palette) in zip(axes, breakdowns):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        rates  = df.groupby(col, observed=True)["churn"].mean() * 100
        colors = [
            palette[i % len(palette)] for i in range(len(rates))
        ]
        bars = ax.bar(
            rates.index.astype(str), rates.values,
            color=colors, edgecolor="white", width=0.6,
        )
        ax.axhline(
            global_churn, color="#7F8C8D", linestyle="--", linewidth=1.5,
            label=f"Moyenne : {global_churn:.1f}%",
        )
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel("Taux de Churn (%)")
        ax.set_ylim(0, rates.max() * 1.25)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        _add_bar_labels(ax, rates.values)

    fig.suptitle("Analyse Multidimensionnelle du Churn",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "02_churn_multidimensional.png")


def plot_balance_analysis(df: pd.DataFrame) -> None:
    """
    3-panel figure analysing the balance distribution:
    - Mean balance per country (horizontal bars)
    - Balance distribution: churners vs retained (overlapping histograms)
    - Mean balance: active vs inactive members (bar chart)
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1 — Solde moyen par pays
    balance_country = (
        df.groupby("country")["balance"]
        .mean()
        .sort_values(ascending=True)
    )
    axes[0].barh(
        balance_country.index, balance_country.values,
        color="#5DADE2", edgecolor="white",
    )
    axes[0].set_title("Solde Moyen par Pays (€)", fontsize=12)
    axes[0].set_xlabel("Solde moyen (€)")
    axes[0].xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x:,.0f}€")
    )
    for bar, val in zip(axes[0].patches, balance_country.values):
        axes[0].text(
            val + balance_country.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}€", va="center", fontsize=9,
        )
    axes[0].grid(axis="x", alpha=0.25)

    # Panel 2 — Distribution du solde : churners vs retenus
    retained = df[df["churn"] == 0]["balance"]
    churners = df[df["churn"] == 1]["balance"]
    axes[1].hist(retained, bins=40, alpha=0.6, color="#3498DB", label="Retenus")
    axes[1].hist(churners, bins=40, alpha=0.6, color="#E74C3C", label="Churners")
    axes[1].axvline(retained.mean(), color="#2471A3", linestyle="--", linewidth=1.5,
                    label=f"Moy. retenus : {retained.mean():,.0f}€")
    axes[1].axvline(churners.mean(), color="#C0392B", linestyle="--", linewidth=1.5,
                    label=f"Moy. churners : {churners.mean():,.0f}€")
    axes[1].set_title("Distribution du Solde : Retenus vs Churners", fontsize=12)
    axes[1].set_xlabel("Solde (€)")
    axes[1].set_ylabel("Nombre de clients")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # Panel 3 — Solde moyen : actifs vs inactifs
    balance_active = df.groupby("active_member")["balance"].mean()
    balance_active.index = ["Inactif (0)", "Actif (1)"]
    colors = PALETTE_BINARY
    bars = axes[2].bar(
        balance_active.index, balance_active.values,
        color=colors, edgecolor="white", width=0.5,
    )
    axes[2].set_title("Solde Moyen : Actifs vs Inactifs (€)", fontsize=12)
    axes[2].set_ylabel("Solde moyen (€)")
    axes[2].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x:,.0f}€")
    )
    axes[2].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[2], balance_active.values, fmt="{:,.0f}€", offset_ratio=0.01)

    fig.suptitle("Analyse du Solde Bancaire", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "02_balance_analysis.png")


def plot_credit_score_analysis(df: pd.DataFrame) -> None:
    """
    2-panel figure : volume distribution and churn rate per credit segment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    credit_palette = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#1ABC9C"]

    # Panel 1 — Répartition par segment
    vol = df["credit_segment"].value_counts().sort_index()
    bars = axes[0].bar(
        range(len(vol)), vol.values,
        color=credit_palette, edgecolor="white",
    )
    axes[0].set_xticks(range(len(vol)))
    axes[0].set_xticklabels(
        vol.index.astype(str), rotation=20, ha="right", fontsize=9,
    )
    axes[0].set_title("Répartition par Score de Crédit", fontsize=12)
    axes[0].set_ylabel("Nombre de clients")
    axes[0].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[0], vol.values, fmt="{:,.0f}", offset_ratio=0.01)

    # Panel 2 — Taux de churn par segment
    global_churn = df["churn"].mean() * 100
    churn_credit = (
        df.groupby("credit_segment", observed=True)["churn"].mean() * 100
    )
    axes[1].bar(
        range(len(churn_credit)), churn_credit.values,
        color=credit_palette, edgecolor="white",
    )
    axes[1].axhline(
        global_churn, color="#7F8C8D", linestyle="--", linewidth=1.5,
        label=f"Moyenne : {global_churn:.1f}%",
    )
    axes[1].set_xticks(range(len(churn_credit)))
    axes[1].set_xticklabels(
        churn_credit.index.astype(str), rotation=20, ha="right", fontsize=9,
    )
    axes[1].set_title("Taux de Churn par Score de Crédit", fontsize=12)
    axes[1].set_ylabel("Taux de churn (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(0, churn_credit.max() * 1.25)
    _add_bar_labels(axes[1], churn_credit.values)

    fig.suptitle("Analyse du Score de Crédit", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "02_credit_score_analysis.png")


def plot_product_tenure_analysis(df: pd.DataFrame) -> None:
    """
    2-panel figure : product penetration (pie chart) and
    mean tenure per country (horizontal bar chart).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1 — Pénétration produit
    prod_dist = df["products_number"].value_counts().sort_index()
    wedges, texts, autotexts = axes[0].pie(
        prod_dist.values,
        labels=[f"{p} produit(s)" for p in prod_dist.index],
        autopct="%1.1f%%",
        colors=PALETTE_PRODUCTS[:len(prod_dist)],
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    axes[0].set_title("Pénétration Produit (% du portefeuille)", fontsize=12)

    # Panel 2 — Ancienneté moyenne par pays
    tenure_country = (
        df.groupby("country")["tenure"].mean().sort_values(ascending=True)
    )
    axes[1].barh(
        tenure_country.index, tenure_country.values,
        color=PALETTE_COUNTRY[:len(tenure_country)], edgecolor="white",
    )
    axes[1].set_title("Ancienneté Moyenne par Pays (années)", fontsize=12)
    axes[1].set_xlabel("Ancienneté moyenne (années)")
    for bar, val in zip(axes[1].patches, tenure_country.values):
        axes[1].text(
            val + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} ans", va="center", fontsize=10,
        )
    axes[1].set_xlim(0, tenure_country.max() * 1.2)
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Pénétration Produit & Ancienneté Client",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "02_product_tenure_analysis.png")


def plot_kpi_heatmap_by_country(df: pd.DataFrame,
                                kpis_country: pd.DataFrame) -> None:
    """
    Normalised heatmap comparing KPI values across countries.
    Each cell shows the raw value; colour reflects relative intensity.

    Parameters
    ----------
    df           : pd.DataFrame  (full dataset, for global reference line)
    kpis_country : pd.DataFrame  (output of compute_kpis_by_group)
    """
    metrics = [
        "churn_rate_%", "balance_mean", "credit_score_mean",
        "tenure_mean", "products_mean", "active_rate_%",
    ]
    labels = [
        "Churn (%)", "Solde Moyen (€)", "Score Crédit",
        "Ancienneté (ans)", "Produits Moy.", "Actifs (%)",
    ]

    kpis_country = kpis_country.set_index("country")
    subset = kpis_country[metrics].copy()

    # Min-max normalisation for colour scale (each metric independently)
    subset_norm = (subset - subset.min()) / (subset.max() - subset.min())

    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(
        subset_norm,
        annot=subset.round(1),
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.8,
        ax=ax,
        cbar_kws={"label": "Intensité relative (0→1)", "shrink": 0.8},
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_title("KPIs par Pays — Valeurs réelles (couleur = intensité relative)",
                 fontsize=12, pad=12)
    plt.tight_layout()
    _save(fig, "02_kpi_heatmap_country.png")


def plot_churn_vs_balance_scatter(df: pd.DataFrame) -> None:
    """
    Scatter plot of balance vs estimated salary, coloured by churn status.
    Reveals whether high-balance clients churn differently.
    A 1% random sample is used to avoid overplotting.
    """
    sample = df.sample(frac=0.2, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group, color in [
        ("Retenu (churn=0)", sample[sample["churn"] == 0], "#3498DB"),
        ("Churner (churn=1)", sample[sample["churn"] == 1], "#E74C3C"),
    ]:
        ax.scatter(
            group["balance"], group["estimated_salary"],
            c=color, alpha=0.4, s=15, label=label, edgecolors="none",
        )

    ax.set_xlabel("Solde Bancaire (€)", fontsize=11)
    ax.set_ylabel("Salaire Estimé (€)", fontsize=11)
    ax.set_title("Solde vs Salaire Estimé — Churners vs Retenus (échantillon 20%)",
                 fontsize=12)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, "02_scatter_balance_salary.png")


def plot_retention_funnel(df: pd.DataFrame) -> None:
    """
    Horizontal funnel chart showing the portfolio breakdown:
    total → high-value → active → retained (non-churners).
    Provides an at-a-glance overview of portfolio health.
    """
    n_total      = len(df)
    n_high_value = df["high_value"].sum()
    n_active     = df["active_member"].sum()
    n_retained   = (df["churn"] == 0).sum()

    stages  = ["Total Clients", "Clients Actifs", "Clients Haute Valeur", "Clients Retenus"]
    values  = [n_total, n_active, n_high_value, n_retained]
    colors  = ["#5DADE2", "#2ECC71", "#F39C12", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        stages[::-1], values[::-1],
        color=colors[::-1], edgecolor="white", height=0.55,
    )
    for bar, val in zip(bars, values[::-1]):
        pct = val / n_total * 100
        ax.text(
            bar.get_width() + n_total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)",
            va="center", fontsize=10, fontweight="bold",
        )
    ax.set_xlim(0, n_total * 1.18)
    ax.set_xlabel("Nombre de clients", fontsize=11)
    ax.set_title("Entonnoir Portefeuille Client", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    plt.tight_layout()
    _save(fig, "02_retention_funnel.png")


# ══════════════════════════════════════════════════════════════════════════════
#  V. EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_outputs(df: pd.DataFrame,
                   kpis: Dict[str, float],
                   kpis_country: pd.DataFrame) -> None:
    """
    Export all outputs needed for Power BI ingestion.

    Files produced
    --------------
    bank_churn_enriched.csv : full enriched dataset with all derived columns
    kpis_summary.csv        : flat KPI table (one row per KPI)
    kpis_by_country.csv     : KPI aggregates by country

    Parameters
    ----------
    df          : pd.DataFrame  (enriched dataset)
    kpis        : dict          (global KPI dictionary)
    kpis_country: pd.DataFrame  (country-level KPI table)
    """
    # 1. Full enriched dataset
    df.to_csv(DATA_OUT, index=False)
    log.info(f"Enriched dataset →  {DATA_OUT.relative_to(ROOT_DIR)}")

    # 2. Flat KPI summary (long format: one row per KPI)
    kpis_df = pd.DataFrame(
        [(k, v) for k, v in kpis.items()],
        columns=["kpi", "value"],
    )
    kpis_df.to_csv(KPI_OUT, index=False)
    log.info(f"KPI summary      →  {KPI_OUT.relative_to(ROOT_DIR)}")

    # 3. Country-level KPIs
    kpis_country.to_csv(COUNTRY_OUT, index=False)
    log.info(f"KPIs by country  →  {COUNTRY_OUT.relative_to(ROOT_DIR)}")

    log.info("All exports complete.")
    log.info("Power BI: import bank_churn_enriched.csv and kpis_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  VI. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Full KPI analysis pipeline:
      1. Load and validate clean dataset
      2. Enrich with derived features
      3. Compute global KPIs
      4. Compute KPIs by group (country, age_group, credit_segment)
      5. Produce all visualisations
      6. Export outputs for Power BI
    """
    log.info("=" * 60)
    log.info("  PIPELINE START — 02_kpis_analysis.py")
    log.info("=" * 60)

    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.family"] = "DejaVu Sans"

    # ── 1. Load ───────────────────────────────────────────────────────────
    df = load_clean_data(DATA_IN)

    # ── 2. Enrich ─────────────────────────────────────────────────────────
    df = enrich_features(df)

    # ── 3. Global KPIs ────────────────────────────────────────────────────
    kpis = compute_global_kpis(df)

    # ── 4. Group KPIs ─────────────────────────────────────────────────────
    kpis_country = compute_kpis_by_group(df, "country")
    kpis_age     = compute_kpis_by_group(df, "age_group")
    kpis_credit  = compute_kpis_by_group(df, "credit_segment")

    log.info("KPIs by country computed")
    log.info("KPIs by age group computed")
    log.info("KPIs by credit segment computed")

    # ── 5. Visualisations ─────────────────────────────────────────────────
    log.info("Generating visualisations …")
    plot_churn_multidimensional(df)
    plot_balance_analysis(df)
    plot_credit_score_analysis(df)
    plot_product_tenure_analysis(df)
    plot_kpi_heatmap_by_country(df, kpis_country)
    plot_churn_vs_balance_scatter(df)
    plot_retention_funnel(df)

    # ── 6. Export ─────────────────────────────────────────────────────────
    export_outputs(df, kpis, kpis_country)

    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
