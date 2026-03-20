# Bank Customer Analysis — Churn, KPIs & Segmentation

Projet d'analyse complète d'un portefeuille bancaire de 10 000 clients.
Pipeline en 3 étapes : nettoyage & EDA → KPIs métier → segmentation K-Means.
Les outputs sont prêts pour ingestion dans Power BI.

---

## Aperçu du projet

La rétention client est un enjeu central en banque de détail. Ce projet construit
un pipeline analytique complet allant de l'exploration brute des données jusqu'à
une segmentation non supervisée, en passant par le calcul d'indicateurs clés de
performance (KPIs) multi-dimensionnels.

> Ce projet ne produit **pas** un modèle de prédiction supervisé.

> L'objectif est l'analyse descriptive, le diagnostic et la segmentation exploratoire.

---

##  Dataset

- **Source :** [Kaggle — Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- **Fichier :** `data/bank_churn.csv`
- **Lignes :** 10 000 clients
- **Variable cible :** `churn` (1 = parti, 0 = retenu)

| Colonne | Type | Description |
|---|---|---|
| `customer_id` | int | Identifiant client (supprimé au nettoyage) |
| `credit_score` | int | Score de crédit (300–900) |
| `country` | str | Pays (France, Spain, Germany) |
| `gender` | str | Genre |
| `age` | int | Âge du client |
| `tenure` | int | Ancienneté en années (0–10) |
| `balance` | float | Solde bancaire (€) |
| `products_number` | int | Nombre de produits détenus (1–4) |
| `credit_card` | int | Possède une carte de crédit (1/0) |
| `active_member` | int | Membre actif (1/0) |
| `estimated_salary` | float | Salaire estimé annuel (€) |
| `churn` | int | A quitté la banque (1/0) — variable cible |

---

## Stack technique

- **Langage :** Python 3.10+
- **Manipulation de données :** Pandas, NumPy
- **Visualisation :** Matplotlib, Seaborn
- **Machine Learning :** Scikit-learn (KMeans, StandardScaler, PCA, Silhouette)
- **Environnement :** Scripts Python autonomes (pas de notebook)
- **Export BI :** CSV structurés compatibles Power BI

---

## Installation & Lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/Malick-Traore/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Télécharger le dataset
Télécharge `bank_churn.csv` depuis Kaggle et place-le dans `data/`.
(Voir lien dans la section Dataset ci-dessus.)

### 4. Lancer le pipeline complet
```bash
python notebooks/01_cleaning_eda.py
python notebooks/02_kpis_analysis.py
python notebooks/03_segmentation_kmeans.py
```

> Les scripts doivent être exécutés **dans cet ordre** : chaque script consomme l'output du précédent.

---

## Structure du projet

```
bank-churn-prediction/
│
├── data/
│   ├── bank_churn.csv                 ← dataset brut (non versionné)
│   ├── bank_churn_clean.csv           ← produit par 01
│   ├── bank_churn_enriched.csv        ← produit par 02
│   ├── bank_churn_segmented.csv       ← produit par 03
│   ├── kpis_summary.csv               ← produit par 02 (format Power BI)
│   ├── kpis_by_country.csv            ← produit par 02
│   └── cluster_profiles.csv           ← produit par 03
│
├── notebooks/
│   ├── 01_cleaning_eda.py             ← Nettoyage & analyse exploratoire
│   ├── 02_kpis_analysis.py            ← KPIs métier & visualisations
│   └── 03_segmentation_kmeans.py      ← Segmentation K-Means
│
├── dashboard/
│   └── screenshots/                   ← Tous les graphiques exportés (.png)
│       ├── 01_outlier_boxplots.png
│       ├── 01_numeric_distributions.png
│       ├── 01_categorical_distributions.png
│       ├── 01_correlation_matrix.png
│       ├── 01_churn_by_derived_features.png
│       ├── 02_churn_multidimensional.png
│       ├── 02_balance_analysis.png
│       ├── 02_credit_score_analysis.png
│       ├── 02_product_tenure_analysis.png
│       ├── 02_kpi_heatmap_country.png
│       ├── 02_scatter_balance_salary.png
│       ├── 02_retention_funnel.png
│       ├── 03_elbow_silhouette.png
│       ├── 03_silhouette_diagram.png
│       ├── 03_cluster_heatmap.png
│       ├── 03_churn_by_cluster.png
│       ├── 03_pca_scatter.png
│       └── 03_radar_profiles.png
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Pipeline détaillé

### Étape 1 — `01_cleaning_eda.py` : Nettoyage & EDA
- Suppression des colonnes identifiantes (`customer_id`)
- Standardisation des noms de colonnes (snake_case)
- Déduplication et coercition des types
- Détection des outliers (méthode IQR de Tukey, k=1.5)
- Feature engineering : `age_group`, `credit_segment`, `balance_group`, `tenure_group`
- 5 visualisations : boxplots, distributions, corrélation, churn par segment

**Output :** `data/bank_churn_clean.csv`

### Étape 2 — `02_kpis_analysis.py` : KPIs & analyse métier
- Calcul de 17 KPIs globaux (churn rate, solde moyen, taux d'activité...)
- KPIs agrégés par pays, tranche d'âge, score de crédit
- Flags dérivés : `high_value` (solde top 25% + actif), `at_risk`
- 7 visualisations : churn multidimensionnel, analyse du solde, entonnoir...

**Output :** `data/bank_churn_enriched.csv`, `data/kpis_summary.csv`, `data/kpis_by_country.csv`

### Étape 3 — `03_segmentation_kmeans.py` : Segmentation K-Means
- Normalisation Z-score (StandardScaler)
- Sélection du K optimal : méthode du coude + score de silhouette (K=2 à 10)
- Entraînement final K-Means (n_init=15, max_iter=500)
- Profilage des clusters + nommage automatique basé sur des règles métier
- 6 visualisations : elbow, silhouette diagram, heatmap profils, PCA 2D, radar

**Output :** `data/bank_churn_segmented.csv`, `data/cluster_profiles.csv`

---

## Résultats clés

| KPI | Valeur |
|---|---|
| Taux de churn global | ~20.4% |
| Solde moyen | ~76 500 € |
| Taux de membres actifs | ~51.5% |
| Taux de détention carte | ~70.5% |
| Clients haute valeur | ~25% |

> Les résultats précis dépendent du dataset téléchargé.
> Consulter `data/kpis_summary.csv` après exécution pour les valeurs exactes.

---

## Intégration Power BI

Importer dans Power BI dans cet ordre :

1. `bank_churn_segmented.csv` → table principale (10 000 lignes, toutes colonnes)
2. `kpis_summary.csv` → visuels "Card" (format long : kpi / value)
3. `kpis_by_country.csv` → carte géographique et comparaisons
4. `cluster_profiles.csv` → visuels de synthèse des segments

---

## Auteur

**Malick Traore** — [@Malick-Traore](https://github.com/Malick-Traore)

**Linkedin** — [@Linkedin](www.linkedin.com/in/traore-malick) 
