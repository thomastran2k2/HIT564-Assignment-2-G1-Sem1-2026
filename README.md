# PRT564 — Assessment 2 (Group 1)

End-to-end pipeline for **Northern Territory crime**, **population**, and **wholesale alcohol supply (PAC)**. The main script **cleans + merges** the source files, writes an analysis-ready dataset, then runs:

- **EDA** for RQ1 (monthly patterns) and RQ2 (assault + predictors)
- **Regression modelling** for RQ2 (feature variants + Ridge/Lasso tuning + diagnostics)

## Requirements

- **Python** 3.8+ recommended  
- **Packages**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`, `scikit-learn`, `scipy`, `statsmodels`

Install example:

```bash
python -m pip install pandas numpy matplotlib seaborn openpyxl scikit-learn scipy statsmodels
```

## Input data

Place these files in the **same directory** as the script (paths are relative to the script folder):

| File | Description |
|------|-------------|
| `nt_crime_statistics_dec_2025.csv` | NT crime statistics |
| `nt-government-regions_1986-to-2025.xlsx` | Population by NT Government region |
| `wholesale-alcohol-supply-by-quarter-2023.xlsx` | Alcohol supply (for imputation context) |
| `wholesale-alcohol-supply-by-quarter-2024.xlsx` | Alcohol supply |
| `wholesale-alcohol-supply-by-quarter-2025.xlsx` | Alcohol supply |

## How to run

From this folder:

```bash
python 01_data_1.py
```

The script runs **Steps 1–6** (load → merge → save CSV), then **EDA**, then **Regression**.

## Outputs

| Output | Description |
|--------|-------------|
| `nt_crime_merged.csv` | Merged, aggregated table: crime counts + population features + PAC columns + region dummies |
| `eda_plots/*.png` | EDA figures listed below |
| `regression_plots/*.png` | Regression figures (tuning, diagnostics, coefficients) |

### Generated plots (`eda_plots/`)

- **Section 1 — Overview:** `1_1_population_by_region.png`, `1_2_offences_by_category.png`, `1_3_alcohol_dv_involvement.png`
- **Section 2 — RQ1 (monthly patterns):** `2_1_offences_by_month.png`, `2_2_crime_rate_per_100k_by_month.png`, `2_3_monthly_trend_2024_vs_2025.png`, `2_4_heatmap_month_category.png`
- **Section 3 — RQ2 (assault):** `3_1_assault_by_region.png` … `3_6_correlation_heatmap.png`


### Generated plots (`regression_plots/`)

The regression section saves multiple figures (exact filenames may change if the script is updated), including:

- **Alpha tuning curves** for Ridge/Lasso (TimeSeriesSplit CV)
- **Assumption checks** (residual plots, Shapiro–Wilk)
- **Cross-validation / test performance comparisons**
- **Model coefficients / feature importance**

## What the pipeline does (summary)

1. **Crime**  
   - Drops `Unknown` reporting region and **year 2023** (only partial year in source).  
   - Remaps rows to **six NT Government statistical regions** (`Greater Darwin`, `Central Australia`, `Big Rivers`, `East Arnhem`, `Barkly`, `Top End`).  
   - For **NT Balance**, uses **SA2** (`Statistical Area 2`) to assign the correct population region where possible; unmatched SA2 defaults to **Top End**.  
   - Encodes **Alcohol involvement** and **DV involvement** as binary **0/1** (`-` → 0).

2. **Population**  
   - Filters to **2024–2025**, builds totals, Aboriginal / non-Aboriginal, sex splits, and **age-group** columns (`Pop_age_*`).

3. **Alcohol**  
   - Loads **2023–2025** quarters.  
   - Maps police-style regions to the same **six** regions; **Darwin + Palmerston → Greater Darwin** and **PAC is summed** for that combined region.

4. **Merge and aggregation**  
   - Left-joins crime to alcohol on **Year, Quarter, Region**, then population on **Year, Region**.  
   - Aggregates **Number of offences** across consistent dimensions.

5. **PAC imputation**  
   - Missing PAC (notably **Q3–Q4 2025** in the source) is filled with the **mean of the same Region × Quarter** across available rows, then rounded to integers.

6. **Modelling helpers**  
   - One-hot encodes **Region**; **Greater Darwin** is dropped as the **reference** category.

7. **Regression (RQ2)**  
   - Builds a monthly assault-rate dataset per region, engineers seasonal + lag features, and uses a **log transform** of assault rate.  
   - Compares 4 feature variants (temporal → + alcohol → + crime context → + region).  
   - Tunes **Ridge** and **Lasso** with **TimeSeriesSplit** cross-validation and runs diagnostics/assumption checks.

## Notes

- If `nt_crime_merged.csv` is open in **Excel** or locked by **OneDrive**, saving may fail with a permission error—close the file or pause sync and rerun.  
- Crime analysis years in the merged file are **2024–2025**; alcohol still uses **2023** where needed for imputation and quarter structure.

## Authors

**PRT564 — Data Analytics and Visualisation**, Assessment 2, **Group 1**.
