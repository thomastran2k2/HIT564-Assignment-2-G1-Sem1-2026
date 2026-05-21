"""
PRT564 - Data Analytics and Visualisation
Assessment 2: Group Project Presentation
Group 1

Script: 01_data_preprocessing.py
Purpose: Full pipeline for Assessment 4 — data cleaning, merging,
         EDA (RQ1), and classification (RQ3, RQ4).

Research Questions:
    RQ3: Classify NT region-months into High/Medium/Low risk for violent crime.
    RQ4: Classify months into High/Medium/Low risk for seasonal resource planning.

Inputs (place all files in the same folder as this script):
    nt_crime_statistics_dec_2025.csv
    nt-government-regions_1986-to-2025.xlsx
    wholesale-alcohol-supply-by-quarter-2023.xlsx
    wholesale-alcohol-supply-by-quarter-2024.xlsx
    wholesale-alcohol-supply-by-quarter-2025.xlsx

Outputs:
    nt_crime_merged.csv
    eda_plots/*.png
    classification_plots/*.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from numpy.polynomial.polynomial import polyfit
from scipy.stats import shapiro, ttest_rel
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import shap
 
warnings.filterwarnings("ignore")

DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
EDA_DIR     = os.path.join(DATA_DIR, "eda_plots")
CLASS_DIR   = os.path.join(DATA_DIR, "classification_plots")
os.makedirs(EDA_DIR,   exist_ok=True)
os.makedirs(CLASS_DIR, exist_ok=True)

def path(filename):
    return os.path.join(DATA_DIR, filename)

def save_eda(fig, name):
    fig.savefig(os.path.join(EDA_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

def save_cls(fig, name):
    fig.savefig(os.path.join(CLASS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")
    
# Shared plot style
sns.set_theme(style="whitegrid", palette="muted", font="DejaVu Sans")
plt.rcParams.update({
    "figure.dpi":       150,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
})
 
REGION_ORDER   = ["Greater Darwin", "Central Australia", "Big Rivers",
                  "East Arnhem", "Barkly", "Top End"]
REGION_PALETTE = sns.color_palette("tab10", n_colors=6)
MONTH_LABELS   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
YEAR_PALETTE   = {2024: "#2196F3", 2025: "#FF9800"}
RISK_ORDER     = ["low", "medium", "high"]
RISK_PALETTE   = {"low": "#42A5F5", "medium": "#FF9800", "high": "#EF5350"}

# == STEP 1: Crime data ========================================================

print("=" * 60)
print("STEP 1: Crime data")
print("=" * 60)

crime = pd.read_csv(path("nt_crime_statistics_dec_2025.csv"))
crime.columns = crime.columns.str.strip()
print(f"Loaded: {crime.shape[0]:,} rows x {crime.shape[1]} cols")

# Drop 'Unknown' reporting region and year 2023 (only 1 month available)
crime = crime[crime["Reporting Region"] != "Unknown"].copy()
crime = crime[crime["Year"] != 2023].copy()
print(f"After dropping Unknown region and year 2023: {crime.shape[0]:,} rows")

# Drop As At column
crime.drop(columns=["As At"], inplace=True)

# Derive Quarter from Month number
crime["Quarter"] = crime["Month number"].apply(lambda m: (m - 1) // 3 + 1)

# Remap all crime rows to NT Government population regions (6 regions).
# This ensures crime rate per 100k uses the correct population denominator.
#
# Logic:
#   - For Darwin, Palmerston, Alice Springs, Katherine, Nhulunbuy, Tennant Creek:
#     map directly to their corresponding population region.
#   - For NT Balance rows: use SA2 value to determine the correct population region.
#     SA2 mapping is based on NT Government boundary definitions (see image ref).
#     Rows with Unknown SA2 remain in Top End (best available assignment).

SA2_TO_REGION = {
    # Barkly
    "Barkly":               "Barkly",
    "Sandover - Plenty":    "Barkly",
    # Big Rivers
    "Elsey":                "Big Rivers",
    "Gulf":                 "Big Rivers",
    "Victoria River":       "Big Rivers",
    # Central Australia
    "Petermann - Simpson":  "Central Australia",
    "Tanami":               "Central Australia",
    "Yuendumu - Anmatjere": "Central Australia",
    # East Arnhem
    "East Arnhem":          "East Arnhem",
    "Anindilyakwa":         "East Arnhem",
    # Top End
    "Alligator":            "Top End",
    "West Arnhem":          "Top End",
    "Thamarrurr":           "Top End",
    "Tiwi Islands":         "Top End",
    "Daly":                 "Top End",
    # Greater Darwin
    "Howard Springs":       "Greater Darwin",
    "Humpty Doo":           "Greater Darwin",
    "Koolpinyah":           "Greater Darwin",
    "Virginia":             "Greater Darwin",
    "Weddell":              "Greater Darwin",
}

REGION_TO_POP = {
    "Darwin":        "Greater Darwin",
    "Palmerston":    "Greater Darwin",
    "Alice Springs": "Central Australia",
    "Katherine":     "Big Rivers",
    "Nhulunbuy":     "East Arnhem",
    "Tennant Creek": "Barkly",
    "NT Balance":    "Top End",   # fallback for unmatched SA2
}

def remap_region(row):
    region = row["Reporting Region"]
    sa2    = row["Statistical Area 2"]
    if region == "NT Balance":
        if pd.notna(sa2) and sa2 in SA2_TO_REGION:
            return SA2_TO_REGION[sa2]
        return "Top End"  # Unknown SA2 → Top End
    return REGION_TO_POP.get(region, region)

crime["Region"] = crime.apply(remap_region, axis=1)
crime.drop(columns=["Reporting Region", "Statistical Area 2"], inplace=True)

# Encode Alcohol involvement and DV involvement as binary (0/1)
# '-' = not applicable for non-assault offences → treated as 0
for col in ["Alcohol involvement", "DV involvement"]:
    crime[col] = crime[col].map({"Yes": 1, "No": 0, "-": 0}).astype(int)

print(f"After region remapping: {crime.shape[0]:,} rows")
print(f"Regions (6): {sorted(crime['Region'].unique())}")
print()


# == STEP 2: Population data ===================================================

print("=" * 60)
print("STEP 2: Population data")
print("=" * 60)

pop = pd.read_excel(path("nt-government-regions_1986-to-2025.xlsx"))
pop = pop[pop["Year"].between(2024, 2025)].copy()
pop.drop(columns=["Status"], inplace=True)
print(f"Loaded and filtered to 2024-2025: {pop.shape[0]} rows")

# Total population per region/year
pop_total = (
    pop.groupby(["Year", "Region"])["Population"]
    .sum().reset_index()
    .rename(columns={"Population": "Total_population"})
)

# Aboriginal / Non-Aboriginal raw counts
pop_abor = (
    pop.groupby(["Year", "Region", "Aboriginal status"])["Population"]
    .sum().unstack(fill_value=0).reset_index()
)
pop_abor.columns.name = None

# Male / Female raw counts
pop_sex = (
    pop.groupby(["Year", "Region", "Sex"])["Population"]
    .sum().unstack(fill_value=0).reset_index()
)
pop_sex.columns.name = None

# Population by age group (18 individual columns)
pop_age = (
    pop.groupby(["Year", "Region", "Age Group"])["Population"]
    .sum().unstack(fill_value=0).reset_index()
)
pop_age.columns.name = None
pop_age.rename(columns={
    col: "Pop_age_" + col.replace("-", "").replace("+", "plus")
    for col in pop_age.columns if col not in ["Year", "Region"]
}, inplace=True)

# Combine all population features
pop_features = (
    pop_total
    .merge(pop_abor, on=["Year", "Region"])
    .merge(pop_sex,  on=["Year", "Region"])
    .merge(pop_age,  on=["Year", "Region"])
)

print(f"Population features: {pop_features.shape[0]} rows x {pop_features.shape[1]} cols")
print(pop_features[["Year", "Region", "Total_population"]].to_string(index=False))
print()


# == STEP 3: Alcohol data ======================================================

print("=" * 60)
print("STEP 3: Alcohol data")
print("=" * 60)

alc_frames = []
for yr in [2023, 2024, 2025]:
    df_alc = pd.read_excel(
        path(f"wholesale-alcohol-supply-by-quarter-{yr}.xlsx"),
        sheet_name="Data"
    )
    alc_frames.append(df_alc)
    print(f"  {yr}: {df_alc.shape[0]} rows")

alc = pd.concat(alc_frames, ignore_index=True)
alc["Quarter Ending"] = pd.to_datetime(alc["Quarter Ending"])
alc["Year"]    = alc["Quarter Ending"].dt.year
alc["Quarter"] = alc["Quarter Ending"].dt.month.apply(
    lambda m: {3: 1, 6: 2, 9: 3, 12: 4}[m]
)
alc.drop(columns=["Quarter Ending"], inplace=True)

# Remap alcohol regions to population regions (same logic as crime)
# Darwin + Palmerston → Greater Darwin (sum PAC)
# Others map 1-to-1
ALC_REGION_MAP = {
    "Darwin":        "Greater Darwin",
    "Palmerston":    "Greater Darwin",
    "Alice Springs": "Central Australia",
    "Katherine":     "Big Rivers",
    "Nhulunbuy":     "East Arnhem",
    "Tennant Creek": "Barkly",
    "NT Balance":    "Top End",
}
alc["Region"] = alc["Region"].map(ALC_REGION_MAP)

pac_cols = [c for c in alc.columns if c not in ["Region", "Year", "Quarter"]]

# Aggregate Darwin + Palmerston into Greater Darwin
alc = (
    alc.groupby(["Year", "Quarter", "Region"])[pac_cols]
    .sum().reset_index()
)

print(f"Combined and remapped: {alc.shape[0]} rows")
print(f"Regions: {sorted(alc['Region'].unique())}")
print("Year x Quarter coverage:")
print(alc.groupby(["Year", "Quarter"]).size().reset_index()
      .rename(columns={0: "n_regions"}).to_string(index=False))
print()


# == STEP 4: Merge =============================================================

print("=" * 60)
print("STEP 4: Merging datasets")
print("=" * 60)

# Crime + Alcohol (left join on Year, Quarter, Region)
merged = pd.merge(crime, alc, on=["Year", "Quarter", "Region"], how="left")
print(f"After crime + alcohol : {merged.shape[0]:,} rows")
print(f"  PAC nulls (Q3-Q4 2025): {merged['Total PAC'].isna().sum():,}")

# + Population (left join on Year, Region)
merged = pd.merge(merged, pop_features, on=["Year", "Region"], how="left")
print(f"After + population    : {merged.shape[0]:,} rows")
print(f"  Population nulls    : {merged['Total_population'].isna().sum():,}")

# Aggregate: sum Number of offences by all meaningful dimensions
pop_cols = (
    ["Total_population", "Aboriginal", "Non-Aboriginal", "Male", "Female"]
    + sorted([c for c in merged.columns if c.startswith("Pop_age_")])
)
alc_cols = pac_cols  # already defined above

group_cols = (
    ["Year", "Quarter", "Month number", "Region",
     "Offence category", "Offence type",
     "Alcohol involvement", "DV involvement"]
    + pop_cols + alc_cols
)

merged = (
    merged.groupby(group_cols, dropna=False)["Number of offences"]
    .sum().reset_index()
)
print(f"After aggregation     : {merged.shape[0]:,} rows")

# Impute missing PAC values (Q3-Q4 2025) using mean of same region + quarter
for col in alc_cols:
    region_quarter_mean = (
        merged.groupby(["Region", "Quarter"])[col].transform("mean")
    )
    merged[col] = merged[col].fillna(region_quarter_mean)

# Round PAC to integer
for col in alc_cols:
    merged[col] = merged[col].round(0).astype(int)

print(f"PAC nulls after imputation: {merged['Total PAC'].isna().sum()}")

# One-hot encode Region (6 regions, drop Greater Darwin as reference category)
region_dummies = pd.get_dummies(merged["Region"], prefix="Region", drop_first=False)
region_dummies.drop(columns=["Region_Greater Darwin"], inplace=True)
merged = pd.concat([merged, region_dummies], axis=1)
print(f"One-hot encoded regions: {[c for c in merged.columns if c.startswith('Region_')]}")
print()


# == STEP 5: Final checks ======================================================

print("=" * 60)
print("STEP 5: Final checks")
print("=" * 60)

print(f"Final shape: {merged.shape[0]:,} rows x {merged.shape[1]} columns")
print("\nNull summary:")
nulls = merged.isnull().sum()
print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "  No nulls -- dataset is complete")
print("\nRow count by Region x Year:")
print(merged.groupby(["Region", "Year"]).size().unstack(fill_value=0).to_string())
print("\nSample rows (3):")
print(merged[["Year", "Quarter", "Month number", "Region",
              "Offence category", "Alcohol involvement", "DV involvement",
              "Number of offences", "Total_population", "Total PAC"]].head(3).to_string())


# == STEP 6: Save ==============================================================

print()
print("=" * 60)
print("STEP 6: Save")
print("=" * 60)

merged.to_csv(os.path.join(DATA_DIR, "nt_crime_merged.csv"), index=False)
print(f"Saved: nt_crime_merged.csv  ({merged.shape[0]:,} rows x {merged.shape[1]} cols)")
print()
print("Starting EDA...")

# ==============================================================================
# STEP 6: FEATURE ENGINEERING FOR CLASSIFICATION
# Build the region-month level dataset used by both RQ3 and RQ4.
# Unit of analysis: 1 row = 1 region × 1 month.
# ==============================================================================
 
print()
print("=" * 60)
print("STEP 6: Feature engineering for classification")
print("=" * 60)
 
# Filter to assault offences only — target variable is assault rate
assault_cls = merged[merged["Offence category"] == "02 Assault"].copy()
 
# Aggregate to region-month level
cls_df = (
    assault_cls
    .groupby(["Year", "Quarter", "Month number", "Region"])
    .agg(
        Assault_offences = ("Number of offences", "sum"),
        Alcohol_involvement = ("Alcohol involvement", "sum"),
        DV_involvement      = ("DV involvement", "sum"),
        Total_PAC           = ("Total PAC", "first"),
        Total_population    = ("Total_population", "first"),
        Aboriginal          = ("Aboriginal", "first"),
        Male                = ("Male", "first"),
        Pop_age_15          = ("Pop_age_15", "first"),
        Pop_age_20          = ("Pop_age_20", "first"),
        Pop_age_65          = ("Pop_age_65", "first"),
        Pop_age_70          = ("Pop_age_70", "first"),
        Pop_age_75          = ("Pop_age_75", "first"),
        Pop_age_80          = ("Pop_age_80", "first"),
        Pop_age_85plus      = ("Pop_age_85plus", "first"),
        # Individual PAC beverage types
        Cask_Wine_PAC           = ("Cask Wine PAC", "first"),
        Bottled_Wine_PAC        = ("Bottled Wine PAC", "first"),
        Fortified_Wine_PAC      = ("Fortified Wine PAC", "first"),
        Full_Strength_Beer_PAC  = ("Full-Strength Beer PAC", "first"),
        Mid_Strength_Beer_PAC   = ("Mid-Strength Beer PAC", "first"),
        Low_Strength_Beer_PAC   = ("Low-Strength Beer PAC", "first"),
    )
    .reset_index()
    .sort_values(["Region", "Year", "Month number"])
    .reset_index(drop=True)
)
 
# Assault rate per 100,000 population — target for labelling
cls_df["assault_rate"] = (
    cls_df["Assault_offences"] / cls_df["Total_population"] * 100000
).round(2)
 
# Demographic proportion features
# Raw counts are divided by population for fair cross-region comparison
cls_df["pct_aboriginal"] = cls_df["Aboriginal"] / cls_df["Total_population"]
cls_df["pct_male"]       = cls_df["Male"]        / cls_df["Total_population"]
# Youth = age 15–24 (two 5-year bands: 15-19 and 20-24)
cls_df["pct_youth"]  = (
    (cls_df["Pop_age_15"] + cls_df["Pop_age_20"]) / cls_df["Total_population"]
)
# Senior = age 65+
cls_df["pct_senior"] = (
    (cls_df["Pop_age_65"] + cls_df["Pop_age_70"] + cls_df["Pop_age_75"]
     + cls_df["Pop_age_80"] + cls_df["Pop_age_85plus"])
    / cls_df["Total_population"]
)
 
# Per-capita alcohol supply features
# Dividing by population controls for region size (Greater Darwin has high
# absolute PAC simply because it has more people)
cls_df["alcohol_per_capita"]                  = cls_df["Total_PAC"]           / cls_df["Total_population"]
cls_df["full_strength_beer_pac_per_capita"]   = cls_df["Full_Strength_Beer_PAC"]  / cls_df["Total_population"]
cls_df["mid_strength_beer_pac_per_capita"]    = cls_df["Mid_Strength_Beer_PAC"]   / cls_df["Total_population"]
cls_df["low_strength_beer_pac_per_capita"]    = cls_df["Low_Strength_Beer_PAC"]   / cls_df["Total_population"]
cls_df["cask_wine_pac_per_capita"]            = cls_df["Cask_Wine_PAC"]           / cls_df["Total_population"]
cls_df["bottled_wine_pac_per_capita"]         = cls_df["Bottled_Wine_PAC"]        / cls_df["Total_population"]
cls_df["fortified_wine_pac_per_capita"]       = cls_df["Fortified_Wine_PAC"]      / cls_df["Total_population"]
 
# Cyclic month encoding
# Integer month 1-12 implies linear relationship (December and January appear
# 11 months apart). Sin/cos pair encodes them as adjacent, which is correct.
cls_df["sin_month"] = np.sin(2 * np.pi * cls_df["Month number"] / 12)
cls_df["cos_month"] = np.cos(2 * np.pi * cls_df["Month number"] / 12)
 
# Risk label: tertile thresholds from 2024 training data ONLY.
# Fixed thresholds applied to 2025 prevents label leakage.
# Lower and upper bounds set to -inf/+inf so that any 2025 observation
# falling outside the 2024 range is still classified rather than dropped.
train_2024    = cls_df[cls_df["Year"] == 2024]
q_edges       = train_2024["assault_rate"].quantile([0, 0.33, 0.66, 1]).to_numpy().copy()
q_edges[0]    = -np.inf   # ensure no observation is below the lower bound
q_edges[-1]   =  np.inf   # ensure no observation is above the upper bound
cls_df["risk_class"] = pd.cut(
    cls_df["assault_rate"],
    bins=q_edges,
    labels=RISK_ORDER,
    include_lowest=True
)
before = len(cls_df)
cls_df = cls_df.dropna(subset=["risk_class"]).copy()
cls_df["risk_class"] = cls_df["risk_class"].astype(str)
print(f"Risk label created. Rows: {len(cls_df)} (dropped {before - len(cls_df)} NaN)")
print(f"Quantile edges from 2024 (33rd/66th percentile): {q_edges[1:-1].round(2)}")
print("\nClass distribution:")
print(cls_df.groupby(["Year", "risk_class"]).size().unstack(fill_value=0))
 
# Sort temporally for TimeSeriesSplit
cls_df = cls_df.sort_values(["Year", "Month number"]).reset_index(drop=True)
 
# Feature lists
#
# RQ3: Classify region-months — full feature set captures BOTH regional
#      characteristics (demographic, population size) AND temporal patterns
#      (sin/cos month). Goal: predict risk for a specific region in a specific month.
#
# RQ4: Classify months for seasonal resource planning — uses only temporal
#      and alcohol supply features. Demographic features (pct_youth, pct_aboriginal,
#      pct_male, pct_senior, Total_population) are deliberately excluded because
#      they are stable regional characteristics that do not vary by month.
#      Including them would answer "which region is risky" rather than
#      "which month of the year is risky" — which is RQ3's question, not RQ4's.
 
FEATURES_RQ3 = [
    "Total_population",
    "Alcohol_involvement",
    "DV_involvement",
    "pct_aboriginal",
    "pct_male",
    "pct_youth",
    "pct_senior",
    "alcohol_per_capita",
    "full_strength_beer_pac_per_capita",
    "mid_strength_beer_pac_per_capita",
    "low_strength_beer_pac_per_capita",
    "cask_wine_pac_per_capita",
    "bottled_wine_pac_per_capita",
    "fortified_wine_pac_per_capita",
    "sin_month",
    "cos_month",
]
 
# RQ4 uses only seasonal and alcohol supply features.
# sin_month + cos_month: capture cyclic seasonal pattern.
# Alcohol features: supply varies by quarter/season and drives assault risk.
# Alcohol_involvement + DV_involvement: crime context signals that vary month to month.
# Demographic features excluded — they are region-level constants, not monthly signals.
FEATURES_RQ4 = [
    "sin_month",
    "cos_month",
    "alcohol_per_capita",
    "full_strength_beer_pac_per_capita",
    "mid_strength_beer_pac_per_capita",
    "low_strength_beer_pac_per_capita",
    "cask_wine_pac_per_capita",
    "bottled_wine_pac_per_capita",
    "fortified_wine_pac_per_capita",
    "Alcohol_involvement",
    "DV_involvement",
]
 
print(f"\nRQ3 features ({len(FEATURES_RQ3)}): demographic + alcohol + temporal")
print(f"RQ4 features ({len(FEATURES_RQ4)}): temporal + alcohol only (no demographics)")
 
 
# ==============================================================================
# EDA SECTION 1: DATASET OVERVIEW
# ==============================================================================
 
print()
print("=" * 60)
print("EDA SECTION 1: Dataset Overview")
print("=" * 60)
 
df = merged.copy()
df["Month_label"] = df["Month number"].apply(lambda m: MONTH_LABELS[m - 1])
assault_eda = df[df["Offence category"] == "02 Assault"].copy()
pop_plot    = df.groupby(["Region", "Year"])["Total_population"].first().reset_index()
 
# 1.1 Population by region: 2024 vs 2025
pop_pivot = (
    pop_plot.pivot(index="Region", columns="Year", values="Total_population")
    .loc[REGION_ORDER]
)
x = np.arange(len(REGION_ORDER))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.barh(x + w/2, pop_pivot[2024] / 1000, w, label="2024",
             color="#2196F3", edgecolor="white")
b2 = ax.barh(x - w/2, pop_pivot[2025] / 1000, w, label="2025",
             color="#FF9800", edgecolor="white")
ax.set_yticks(x)
ax.set_yticklabels(REGION_ORDER)
ax.set_xlabel("Population (thousands)")
ax.set_title("Population by NT Government Region: 2024 vs 2025")
ax.legend(title="Year")
ax.set_xlim(0, pop_pivot.max().max() / 1000 * 1.2)
ax.invert_yaxis()
for bar, val in zip(b1, pop_pivot[2024]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8)
for bar, val in zip(b2, pop_pivot[2025]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8)
fig.tight_layout()
save_eda(fig, "1_1_population_by_region.png")
 
# 1.2 Offences by category: 2024 vs 2025
cat_by_year = (
    df.groupby(["Offence category", "Year"])["Number of offences"]
    .sum().unstack(fill_value=0)
)
cat_by_year.index = [c.split(" ", 1)[1] for c in cat_by_year.index]
cat_by_year = cat_by_year.sort_values(2024, ascending=True)
 
x = np.arange(len(cat_by_year))
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.barh(x + w/2, cat_by_year[2024], w, label="2024",
             color="#2196F3", edgecolor="white")
b2 = ax.barh(x - w/2, cat_by_year[2025], w, label="2025",
             color="#FF9800", edgecolor="white")
ax.set_yticks(x)
ax.set_yticklabels(cat_by_year.index)
ax.set_xlabel("Number of Offences")
ax.set_title("Offences by Category: 2024 vs 2025")
ax.legend(title="Year")
ax.set_xlim(0, cat_by_year.max().max() * 1.18)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.invert_yaxis()
for bar, val in zip(b1, cat_by_year[2024]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8)
for bar, val in zip(b2, cat_by_year[2025]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8)
fig.tight_layout()
save_eda(fig, "1_2_offences_by_category.png")
 
# 1.3 Alcohol and DV involvement
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, title, colors in zip(
    axes,
    ["Alcohol involvement", "DV involvement"],
    ["Alcohol Involvement in Offences", "DV Involvement in Offences"],
    [["#EF5350", "#42A5F5"], ["#AB47BC", "#66BB6A"]]
):
    involved     = df[df[col] == 1]["Number of offences"].sum()
    not_involved = df[df[col] == 0]["Number of offences"].sum()
    total        = involved + not_involved
    labels = [
        f"Yes\n{involved:,}\n({involved/total*100:.1f}%)",
        f"No / N/A\n{not_involved:,}\n({not_involved/total*100:.1f}%)"
    ]
    ax.bar(labels, [involved, not_involved], color=colors, edgecolor="white", width=0.5)
    ax.set_title(title)
    ax.set_ylabel("Number of Offences")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.suptitle("Alcohol and DV Involvement Across All Offences (2024-2025)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save_eda(fig, "1_3_alcohol_dv_involvement.png")
 
 
# ==============================================================================
# EDA SECTION 2: RQ1 — MONTHLY CRIME PATTERNS
# ==============================================================================
 
print()
print("=" * 60)
print("EDA SECTION 2: RQ1 - Monthly Crime Patterns")
print("=" * 60)
 
monthly_all = (
    df.groupby("Month number")["Number of offences"]
    .sum().reset_index()
)
monthly_all["Month_label"] = monthly_all["Month number"].apply(
    lambda m: MONTH_LABELS[m - 1])
 
pop_year           = df.groupby(["Year", "Region"])["Total_population"].first().reset_index()
total_pop_per_year = pop_year.groupby("Year")["Total_population"].sum()
monthly_yr = df.groupby(["Year", "Month number"])["Number of offences"].sum().reset_index()
monthly_yr["Total_pop"]     = monthly_yr["Year"].map(total_pop_per_year)
monthly_yr["Rate_per_100k"] = (
    monthly_yr["Number of offences"] / monthly_yr["Total_pop"] * 100000
).round(1)
monthly_yr["Month_label"] = monthly_yr["Month number"].apply(
    lambda m: MONTH_LABELS[m - 1])
 
# 2.1 Total offences by month
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ["#EF5350" if v == monthly_all["Number of offences"].max()
              else "#42A5F5" for v in monthly_all["Number of offences"]]
bars = ax.bar(monthly_all["Month_label"], monthly_all["Number of offences"],
              color=bar_colors, edgecolor="white")
ax.set_xlabel("Month")
ax.set_ylabel("Total Number of Offences")
ax.set_title("Total Offences by Month — All Categories (2024-2025)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars, monthly_all["Number of offences"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:,}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(0, monthly_all["Number of offences"].max() * 1.12)
peak_month = monthly_all.loc[monthly_all["Number of offences"].idxmax(), "Month_label"]
ax.annotate(f"Peak: {peak_month}",
            xy=(peak_month, monthly_all["Number of offences"].max()),
            xytext=(peak_month, monthly_all["Number of offences"].max() * 1.07),
            ha="center", color="#EF5350", fontsize=10, fontweight="bold")
fig.tight_layout()
save_eda(fig, "2_1_offences_by_month.png")
 
# 2.2 Crime rate per 100k by month
monthly_rate = (
    monthly_yr.groupby("Month number")
    .agg(Rate_per_100k=("Rate_per_100k", "mean"),
         Month_label=("Month_label", "first"))
    .reset_index()
    .sort_values("Month number")
)
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ["#EF5350" if v == monthly_rate["Rate_per_100k"].max()
              else "#66BB6A" for v in monthly_rate["Rate_per_100k"]]
bars = ax.bar(monthly_rate["Month_label"], monthly_rate["Rate_per_100k"],
              color=bar_colors, edgecolor="white")
ax.set_xlabel("Month")
ax.set_ylabel("Offences per 100,000 Population")
ax.set_title("Average Monthly Crime Rate per 100,000 Population (2024-2025)")
for bar, val in zip(bars, monthly_rate["Rate_per_100k"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(0, monthly_rate["Rate_per_100k"].max() * 1.15)
fig.tight_layout()
save_eda(fig, "2_2_crime_rate_per_100k_by_month.png")
 
# 2.3 Monthly trend: 2024 vs 2025
fig, ax = plt.subplots(figsize=(10, 5))
for yr, grp in monthly_yr.groupby("Year"):
    grp = grp.sort_values("Month number")
    ax.plot(grp["Month_label"], grp["Number of offences"],
            marker="o", linewidth=2, label=str(yr), color=YEAR_PALETTE[yr])
    for _, row in grp.iterrows():
        ax.text(row["Month_label"], row["Number of offences"] + 20,
                f"{int(row['Number of offences']):,}",
                ha="center", fontsize=7.5, color=YEAR_PALETTE[yr])
ax.set_xlabel("Month")
ax.set_ylabel("Total Number of Offences")
ax.set_title("Monthly Crime Trend: 2024 vs 2025")
ax.legend(title="Year")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, monthly_yr["Number of offences"].max() * 1.15)
fig.tight_layout()
save_eda(fig, "2_3_monthly_trend_2024_vs_2025.png")
 
# 2.4 Heatmap: Month x Offence category
heat_data = (
    df.groupby(["Offence category", "Month number"])["Number of offences"]
    .sum().unstack(fill_value=0)
)
heat_data.index   = [c.split(" ", 1)[1] for c in heat_data.index]
heat_data.columns = MONTH_LABELS
fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(heat_data, annot=True, fmt=",d", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Number of Offences"})
ax.set_title("Offences by Month and Category — Heatmap (2024-2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Offence Category")
fig.tight_layout()
save_eda(fig, "2_4_heatmap_month_category.png")
 
peak_m = monthly_all.loc[monthly_all["Number of offences"].idxmax(), "Month_label"]
print(f"RQ1: Peak crime month = {peak_m}")
 
 
# ==============================================================================
# EDA SECTION 3: ASSAULT ANALYSIS (context for classification)
# ==============================================================================
 
print()
print("=" * 60)
print("EDA SECTION 3: Assault Analysis")
print("=" * 60)
 
assault_agg = (
    assault_eda
    .groupby(["Year", "Quarter", "Month number", "Region"])
    .agg(
        Assault_offences = ("Number of offences", "sum"),
        Alcohol_involved = ("Alcohol involvement", "sum"),
        DV_involved      = ("DV involvement", "sum"),
        Total_PAC        = ("Total PAC", "first"),
        Total_population = ("Total_population", "first"),
        Aboriginal       = ("Aboriginal", "first"),
    )
    .reset_index()
)
assault_agg["Assault_rate_100k"] = (
    assault_agg["Assault_offences"] / assault_agg["Total_population"] * 100000
).round(1)
assault_agg["Month_label"] = assault_agg["Month number"].apply(
    lambda m: MONTH_LABELS[m - 1])
 
# 3.1 Raw counts and quarterly rate per region: 2024 vs 2025
assault_region_yr = (
    assault_agg.groupby(["Region", "Year"])
    .agg(Total_assaults=("Assault_offences", "sum"))
    .reset_index()
)
counts_pivot = assault_region_yr.pivot(
    index="Region", columns="Year", values="Total_assaults"
).loc[REGION_ORDER]
rate_pivot = (
    assault_agg.groupby(["Region", "Year"])["Assault_rate_100k"]
    .mean().round(1).unstack()
).loc[REGION_ORDER]
 
x = np.arange(len(REGION_ORDER))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
b1 = axes[0].barh(x + w/2, counts_pivot[2024], w, label="2024",
                  color="#2196F3", edgecolor="white")
b2 = axes[0].barh(x - w/2, counts_pivot[2025], w, label="2025",
                  color="#FF9800", edgecolor="white")
axes[0].set_yticks(x); axes[0].set_yticklabels(REGION_ORDER)
axes[0].set_xlabel("Total Assault Offences")
axes[0].set_title("Total Assault Offences by Region: 2024 vs 2025")
axes[0].legend(title="Year")
axes[0].set_xlim(0, counts_pivot.max().max() * 1.2)
axes[0].invert_yaxis()
for bar, val in zip(b1, counts_pivot[2024]):
    axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f"{val:,}", va="center", fontsize=8)
for bar, val in zip(b2, counts_pivot[2025]):
    axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f"{val:,}", va="center", fontsize=8)
 
b3 = axes[1].barh(x + w/2, rate_pivot[2024], w, label="2024",
                  color="#2196F3", edgecolor="white")
b4 = axes[1].barh(x - w/2, rate_pivot[2025], w, label="2025",
                  color="#FF9800", edgecolor="white")
axes[1].set_yticks(x); axes[1].set_yticklabels(REGION_ORDER)
axes[1].set_xlabel("Avg Quarterly Assault Rate per 100,000 Population")
axes[1].set_title("Avg Quarterly Assault Rate per 100k: 2024 vs 2025")
axes[1].legend(title="Year")
axes[1].set_xlim(0, rate_pivot.max().max() * 1.2)
axes[1].invert_yaxis()
for bar, val in zip(b3, rate_pivot[2024]):
    axes[1].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                 f"{val:,.0f}", va="center", fontsize=8)
for bar, val in zip(b4, rate_pivot[2025]):
    axes[1].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                 f"{val:,.0f}", va="center", fontsize=8)
fig.suptitle("Assault Offences by Region: Raw Counts vs Quarterly Rate per 100k",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save_eda(fig, "3_1_assault_by_region.png")
 
# 3.2 Average monthly assault trend by region
assault_monthly_region = (
    assault_agg.groupby(["Month number", "Region"])["Assault_offences"]
    .mean().reset_index()
)
assault_monthly_region["Month_label"] = assault_monthly_region["Month number"].apply(
    lambda m: MONTH_LABELS[m - 1])
fig, ax = plt.subplots(figsize=(12, 6))
for region, color in zip(REGION_ORDER, REGION_PALETTE):
    grp = (assault_monthly_region[assault_monthly_region["Region"] == region]
           .sort_values("Month number"))
    ax.plot(grp["Month_label"], grp["Assault_offences"],
            marker="o", linewidth=2, label=region, color=color)
ax.set_xlabel("Month")
ax.set_ylabel("Average Assault Offences")
ax.set_title("Average Monthly Assault Offences by Region (2024-2025)")
ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
save_eda(fig, "3_2_assault_trend_by_region.png")
 
# 3.3 PAC vs Assault scatter
fig, ax = plt.subplots(figsize=(9, 6))
for region, color in zip(REGION_ORDER, REGION_PALETTE):
    grp = assault_agg[assault_agg["Region"] == region]
    ax.scatter(grp["Total_PAC"] / 1000, grp["Assault_offences"],
               label=region, color=color, alpha=0.75, s=60, edgecolors="white")
x_vals = assault_agg["Total_PAC"] / 1000
y_vals = assault_agg["Assault_offences"]
c0, m0 = polyfit(x_vals, y_vals, 1)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
ax.plot(x_line, m0 * x_line + c0, color="black", linewidth=1.5,
        linestyle="--", label="Overall trend")
corr_pac = assault_agg[["Total_PAC", "Assault_offences"]].corr().iloc[0, 1]
ax.set_xlabel("Total PAC (thousands of litres)")
ax.set_ylabel("Assault Offences")
ax.set_title(f"Alcohol Supply (PAC) vs Assault Offences\n(Pearson r = {corr_pac:.3f})")
ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
save_eda(fig, "3_3_scatter_pac_vs_assault.png")
 
# 3.4 Correlation heatmap of assault predictors
corr_cols = assault_agg[["Assault_offences", "Total_PAC", "Total_population",
                          "Aboriginal", "Alcohol_involved", "DV_involved"]].copy()
corr_cols.rename(columns={
    "Assault_offences": "Assault Offences",
    "Total_PAC":        "Total PAC",
    "Total_population": "Total Population",
    "Aboriginal":       "Aboriginal Pop.",
    "Alcohol_involved": "Alcohol Involved",
    "DV_involved":      "DV Involved",
}, inplace=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Pearson Correlation"})
ax.set_title("Correlation Heatmap — Assault Predictors")
fig.tight_layout()
save_eda(fig, "3_4_correlation_heatmap.png")
 
 
# ==============================================================================
# EDA SECTION 4: CLASSIFICATION-SPECIFIC EDA
# These plots directly inform model design choices for RQ3 and RQ4.
# ==============================================================================
 
print()
print("=" * 60)
print("EDA SECTION 4: Classification EDA")
print("=" * 60)
 
# 4.1 Risk class distribution by region
region_risk = (
    cls_df.groupby(["Region", "risk_class"])
    .size().reset_index(name="count")
)
x = np.arange(len(REGION_ORDER))
fig, ax = plt.subplots(figsize=(11, 5))
for i, cls in enumerate(RISK_ORDER):
    vals = [
        region_risk[
            (region_risk["Region"] == r) & (region_risk["risk_class"] == cls)
        ]["count"].sum()
        for r in REGION_ORDER
    ]
    ax.bar(x + i * 0.25, vals, 0.25, label=cls.capitalize(),
           color=RISK_PALETTE[cls], edgecolor="white")
ax.set_xticks(x + 0.25)
ax.set_xticklabels(REGION_ORDER, rotation=15, ha="right")
ax.set_ylabel("Number of Region-Months")
ax.set_title("Risk Class Distribution by Region (2024–2025)")
ax.legend(title="Risk Class")
fig.tight_layout()
save_cls(fig, "CLS1_risk_by_region.png")
 
# 4.2 Risk class distribution by month — seasonal pattern
# High-risk months cluster Nov-Jan, justifying cyclic month encoding
month_risk = (
    cls_df.groupby(["Month number", "risk_class"])
    .size().reset_index(name="count")
)
fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(12)
for cls in RISK_ORDER:
    vals = np.array([
        month_risk[
            (month_risk["Month number"] == m) & (month_risk["risk_class"] == cls)
        ]["count"].sum()
        for m in range(1, 13)
    ])
    ax.bar(MONTH_LABELS, vals, bottom=bottom, label=cls.capitalize(),
           color=RISK_PALETTE[cls], edgecolor="white")
    bottom += vals
ax.set_ylabel("Number of Region-Months")
ax.set_title("Risk Class Distribution by Month (2024–2025)\n"
             "High-risk observations cluster Nov–Jan, justifying cyclic month encoding")
ax.legend(title="Risk Class")
fig.tight_layout()
save_cls(fig, "CLS2_risk_by_month.png")
 
# 4.3 GaussianNB assumption check: Shapiro-Wilk per feature per class
# GNB assumes each feature is normally distributed within each class.
# W close to 1 and p > 0.05 indicates normality is not rejected.
print("\n-- GaussianNB Assumption Check: Shapiro-Wilk (features vs class) --")
print(f"  {'Feature':<38} {'Class':<10} {'W':>8} {'p':>10} {'Normal?'}")
print("  " + "-" * 78)
for feat in FEATURES_RQ3:
    for cls in RISK_ORDER:
        vals = cls_df[cls_df["risk_class"] == cls][feat].dropna().values
        if len(vals) >= 3:
            w_stat, p_val = shapiro(vals)
            normal = "Yes" if p_val > 0.05 else "Mild violation"
            print(f"  {feat:<38} {cls:<10} {w_stat:>8.4f} {p_val:>10.4f}  {normal}")
 
# 4.4 Feature correlation heatmap
fig, ax = plt.subplots(figsize=(11, 9))
corr_cls = cls_df[FEATURES_RQ4].corr()
mask = np.triu(np.ones_like(corr_cls, dtype=bool))
sns.heatmap(corr_cls, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, linewidths=0.4, ax=ax,
            cbar_kws={"label": "Pearson r"})
ax.set_title("Correlation Heatmap — Classification Features")
fig.tight_layout()
save_cls(fig, "CLS3_feature_correlation.png")
 
print("\nClassification EDA complete.")
 
 
# ==============================================================================
# CLASSIFICATION PIPELINE (shared for RQ3 and RQ4)
# Steps per RQ:
#   1. Hyperparameter tuning: RF and SVM via GridSearchCV + TimeSeriesSplit
#   2. TimeSeriesSplit CV on 2024 training data (3 folds)
#   3. Paired t-test: GNB vs RF on CV scores
#   4. Final test set evaluation (2025)
#   5. GNB parameters (class priors)
#   6. RF feature importance
#   7. SHAP explanation for class 'high'
#   8. Risk prediction table
# ==============================================================================
 
def run_classification(cls_df, features, rq_label, n_splits=3):
    """
    Full classification pipeline for one research question.
 
    Parameters
    ----------
    cls_df   : DataFrame with risk_class column, sorted by [Year, Month number]
    features : list of feature column names
    rq_label : string label e.g. "RQ3" used for printing and plot filenames
    """
 
    print()
    print("=" * 60)
    print(f"{rq_label}: Classification Pipeline")
    print("=" * 60)
 
    # ── Train / test split ────────────────────────────────────────────────────
    train = cls_df[cls_df["Year"] == 2024].copy()
    test  = cls_df[cls_df["Year"] == 2025].copy()
 
    X_train = train[features]
    y_train = train["risk_class"]
    X_test  = test[features]
    y_test  = test["risk_class"]
 
    print(f"Train: {len(train)} obs | Test: {len(test)} obs")
    print(f"Train class dist: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test  class dist: {y_test.value_counts().sort_index().to_dict()}")
 
    tscv = TimeSeriesSplit(n_splits=n_splits)
 
    # ── GridSearchCV: Random Forest ───────────────────────────────────────────
    # scoring='f1_weighted' is appropriate for imbalanced classes (16/14/28 split)
    # TimeSeriesSplit ensures tuning always trains on past and validates on future
    print(f"\n-- {rq_label} GridSearchCV: Random Forest (scoring=f1_weighted) --")
    rf_param_grid = {
        "n_estimators":      [100, 300, 500],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }
    rf_gs = GridSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        param_grid=rf_param_grid,
        cv=tscv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    rf_gs.fit(X_train, y_train)
    rf_best = rf_gs.best_estimator_
    print(f"  Best params : {rf_gs.best_params_}")
    print(f"  Best CV F1  : {rf_gs.best_score_:.4f}")
 
    # ── GridSearchCV: SVM ─────────────────────────────────────────────────────
    print(f"\n-- {rq_label} GridSearchCV: SVM (scoring=f1_weighted) --")
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(class_weight="balanced", random_state=42)),
    ])
    svm_param_grid = {
        "model__kernel": ["rbf", "poly"],
        "model__C":      [0.1, 1, 10, 100],
        "model__gamma":  ["scale", "auto"],
    }
    svm_gs = GridSearchCV(
        estimator=svm_pipe,
        param_grid=svm_param_grid,
        cv=tscv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    svm_gs.fit(X_train, y_train)
    svm_best = svm_gs.best_estimator_
    print(f"  Best params : {svm_gs.best_params_}")
    print(f"  Best CV F1  : {svm_gs.best_score_:.4f}")
 
    # ── GaussianNB (no hyperparameters; StandardScaler applied via Pipeline) ──
    gnb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GaussianNB()),
    ])
    gnb_pipe.fit(X_train, y_train)
 
    # ── TimeSeriesSplit CV on training data only (2024) ───────────────────────
    # CV runs on 2024 data only to avoid contamination from 2025 test set.
    print(f"\n-- {rq_label} TimeSeriesSplit CV (n_splits=3, 2024 only) --")
    gnb_cv, svm_cv, rf_cv = [], [], []
    fold_rows = []
 
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # Skip fold if training set has fewer than 2 classes — models
        # cannot train on a single class (particularly SVM and RF).
        # This can occur with small datasets like RQ4 (12 training obs).
        if len(ytr.unique()) < 2:
            print(f"  Fold {fold+1}: skipped — only 1 class in training set "
                  f"({ytr.unique()[0]}), {len(Xtr)} obs")
            continue

        try:
            # GNB
            g = Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())])
            g.fit(Xtr, ytr)
            g_acc = accuracy_score(yval, g.predict(Xval))
            gnb_cv.append(g_acc)

            # SVM (best params from grid search)
            best_svm_params = {
                k.replace("model__", ""): v
                for k, v in svm_gs.best_params_.items()
            }
            s = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  SVC(**best_svm_params,
                               class_weight="balanced", random_state=42)),
            ])
            s.fit(Xtr, ytr)
            s_acc = accuracy_score(yval, s.predict(Xval))
            svm_cv.append(s_acc)

            # RF (best params from grid search)
            r = RandomForestClassifier(
                **rf_gs.best_params_,
                class_weight="balanced", random_state=42
            )
            r.fit(Xtr, ytr)
            r_acc = accuracy_score(yval, r.predict(Xval))
            rf_cv.append(r_acc)

            fold_rows.append({
                "Fold": fold + 1,
                "Train": len(Xtr),
                "Val":   len(Xval),
                "GNB":   round(g_acc, 4),
                "SVM":   round(s_acc, 4),
                "RF":    round(r_acc, 4),
            })

        except ValueError as e:
            print(f"  Fold {fold+1}: skipped — {e}")
            continue
 
    if fold_rows:
        print(pd.DataFrame(fold_rows).to_string(index=False))
        print(f"\n  Mean CV Accuracy - "
              f"GNB: {np.mean(gnb_cv):.4f} | "
              f"SVM: {np.mean(svm_cv):.4f} | "
              f"RF : {np.mean(rf_cv):.4f}")
    else:
        print("  All folds skipped — dataset too small for CV.")
        # Use dummy scores so downstream code does not crash
        gnb_cv.extend([0.5, 0.5])
        svm_cv.extend([0.5, 0.5])
        rf_cv.extend([0.5, 0.5])
 
    # ── Paired t-test: GNB vs RF ──────────────────────────────────────────────
    # Tests whether the difference in CV accuracy between GNB and RF is
    # statistically significant. With only 3 folds, power is limited.
    print(f"\n-- {rq_label} Paired t-test: GNB vs RF --")
    print("  H0: Mean CV accuracy of GNB = Mean CV accuracy of RF")
    print("  H1: The two models have significantly different mean CV accuracy")
    t_stat, p_val = ttest_rel(gnb_cv, rf_cv)
    decision = "Reject H0" if p_val < 0.05 else "Fail to reject H0 (not significant)"
    print(f"  t = {t_stat:.4f}, p = {p_val:.4f}  ->  {decision}")
    print("  Note: 3 CV folds provides limited statistical power.")
 
    # ── Test set predictions ──────────────────────────────────────────────────
    y_pred_gnb = gnb_pipe.predict(X_test)
    y_pred_svm = svm_best.predict(X_test)
    y_pred_rf  = rf_best.predict(X_test)
 
    print(f"\n-- {rq_label} Test Set Results (2025) --")
    for name, y_pred in [("GaussianNB",   y_pred_gnb),
                          ("SVM",          y_pred_svm),
                          ("RandomForest", y_pred_rf)]:
        print(f"\n  === {name} ===")
        print(confusion_matrix(y_test, y_pred, labels=RISK_ORDER))
        print(classification_report(y_test, y_pred, labels=RISK_ORDER,
                                    zero_division=0))
 
    # ── GNB class priors ──────────────────────────────────────────────────────
    gnb_model = gnb_pipe.named_steps["model"]
    print(f"\n-- {rq_label} GNB Class Priors (learned from 2024 training data) --")
    for i, cls in enumerate(gnb_model.classes_):
        print(f"  Class '{cls}': prior = {gnb_model.class_prior_[i]:.4f}")
 
    # ── Metrics summary table ─────────────────────────────────────────────────
    metrics = {}
    for name, y_pred in [("GaussianNB",   y_pred_gnb),
                          ("SVM",          y_pred_svm),
                          ("RandomForest", y_pred_rf)]:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        metrics[name] = {
            "Accuracy":       round(accuracy_score(y_test, y_pred), 4),
            "Precision (W)":  round(prec, 4),
            "Recall (W)":     round(rec,  4),
            "F1 (W)":         round(f1,   4),
        }
    compare_df = pd.DataFrame(metrics).T
    print(f"\n-- {rq_label} Model Comparison Table --")
    print(compare_df.to_string())
 
    # ── Plot: model comparison bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(compare_df))
    bw = 0.18
    colors_bar = ["#2196F3", "#66BB6A", "#FF9800", "#EF5350"]
    for i, col in enumerate(compare_df.columns):
        ax.bar(x_pos + i * bw, compare_df[col], bw,
               label=col, color=colors_bar[i], edgecolor="white")
    ax.set_xticks(x_pos + 1.5 * bw)
    ax.set_xticklabels(compare_df.index)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"{rq_label}: Model Comparison — Test Set Metrics (2025)")
    ax.legend(title="Metric")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    save_cls(fig, f"{rq_label}_model_comparison.png")
 
    # ── Plot: confusion matrices ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name, y_pred in zip(
        axes,
        ["GaussianNB", "SVM", "RandomForest"],
        [y_pred_gnb, y_pred_svm, y_pred_rf],
    ):
        cm = confusion_matrix(y_test, y_pred, labels=RISK_ORDER)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=RISK_ORDER, yticklabels=RISK_ORDER,
                    ax=ax, cbar=False)
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name}\nAccuracy = {acc:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(f"{rq_label}: Confusion Matrices — 2025 Test Set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_cls(fig, f"{rq_label}_confusion_matrices.png")
 
    # ── RF feature importance ─────────────────────────────────────────────────
    rf_imp = pd.DataFrame({
        "Feature":    features,
        "Importance": rf_best.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
 
    print(f"\n-- {rq_label} Random Forest Feature Importance --")
    print(rf_imp.to_string(index=False))
 
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_colors = ["#EF5350" if i == 0 else "#90CAF9"
                  for i in range(len(rf_imp))]
    ax.barh(rf_imp["Feature"][::-1], rf_imp["Importance"][::-1],
            color=imp_colors[::-1], edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(f"{rq_label}: Random Forest Feature Importance\n"
                 "(Red = most important)")
    fig.tight_layout()
    save_cls(fig, f"{rq_label}_rf_feature_importance.png")
 
    # ── SHAP explanation: RF, class='high' ────────────────────────────────────
    # SHAP (SHapley Additive exPlanations) shows each feature's contribution
    # to predicting the 'high' risk class for each test observation.
    explainer   = shap.TreeExplainer(rf_best)
    shap_values = explainer.shap_values(X_test.to_numpy())
    high_idx    = list(rf_best.classes_).index("high")
 
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # New SHAP API: shape (n_samples, n_features, n_classes)
        shap_vals_high = shap_values[:, :, high_idx]
    else:
        # Old SHAP API: list of arrays per class
        shap_vals_high = shap_values[high_idx]
 
    shap.summary_plot(
        shap_vals_high,
        X_test.to_numpy(),
        feature_names=features,
        show=False,
    )
    plt.title(f"{rq_label}: SHAP Summary — 'High' Risk Class (RandomForest)")
    plt.tight_layout()
    plt.savefig(os.path.join(CLASS_DIR, f"{rq_label}_shap_high.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {rq_label}_shap_high.png")
 
    # ── Risk prediction table ─────────────────────────────────────────────────
    # RQ3 has Region column; RQ4 (month-level) does not
    base_cols = ["Region", "Month number"] if "Region" in test.columns else ["Month number"]
    pred_df = test[base_cols].copy().reset_index(drop=True)
    pred_df["Month"]  = pred_df["Month number"].apply(lambda m: MONTH_LABELS[m - 1])
    pred_df["Actual"] = y_test.values
    pred_df["GNB"]    = y_pred_gnb
    pred_df["SVM"]    = y_pred_svm
    pred_df["RF"]     = y_pred_rf

    print(f"\n-- {rq_label} Risk Predictions (2025) --")
    if "Region" in pred_df.columns:
        print(pred_df.sort_values(["Region", "Month number"])
              [["Region", "Month", "Actual", "GNB", "SVM", "RF"]]
              .to_string(index=False))
    else:
        print(pred_df.sort_values("Month number")
              [["Month", "Actual", "GNB", "SVM", "RF"]]
              .to_string(index=False))

    # ── Plot: Predicted vs Actual risk class ──────────────────────────────────
    from matplotlib.patches import Patch
    color_map  = {"low": "#42A5F5", "medium": "#FF9800", "high": "#EF5350"}
    legend_elements = [
        Patch(facecolor="#42A5F5", label="Low"),
        Patch(facecolor="#FF9800", label="Medium"),
        Patch(facecolor="#EF5350", label="High"),
    ]

    if rq_label == "RQ3":
        # Heatmap: region x month, side-by-side Actual vs RF Predicted
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        risk_num    = {"low": 0, "medium": 1, "high": 2}

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, col, title in zip(
            axes,
            ["Actual", "RF"],
            ["Actual Risk Class (2025)", "Predicted Risk Class — RF (2025)"]
        ):
            pivot_str = pred_df.pivot_table(
                index="Region", columns="Month", values=col, aggfunc="first"
            )
            ordered_months = [m for m in month_order if m in pivot_str.columns]
            pivot_str = pivot_str[ordered_months]
            pivot_num = pivot_str.applymap(lambda x: risk_num.get(x, np.nan))
            sns.heatmap(
                pivot_num, ax=ax,
                cmap=["#42A5F5", "#FF9800", "#EF5350"],
                vmin=0, vmax=2,
                linewidths=0.5, linecolor="white",
                annot=pivot_str.values, fmt="",
                cbar=False, annot_kws={"size": 8}
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Month")
            ax.set_ylabel("Region")
            ax.tick_params(axis="x", rotation=45)

        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=10)
        fig.suptitle("RQ3: Predicted vs Actual Risk Class by Region (2025)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        save_cls(fig, "RQ3_predicted_vs_actual_heatmap.png")

    else:
        # RQ4: bar chart — one bar per month, Actual vs RF side by side
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        # rq4_df is already at month level — pred_df has one row per month
        # Sort by Month number to ensure calendar order
        pred_monthly = (
            pred_df.sort_values("Month number")
            .reset_index(drop=True)
        )
        # Ensure month labels are in calendar order
        pred_monthly = pred_monthly.set_index("Month").reindex(month_order).reset_index()

        x  = np.arange(len(pred_monthly))
        w  = 0.35
        actual_colors = [color_map.get(v, "grey") for v in pred_monthly["Actual"]]
        pred_colors   = [color_map.get(v, "grey") for v in pred_monthly["RF"]]

        fig, ax = plt.subplots(figsize=(12, 4))
        bars1 = ax.bar(x - w/2, [1] * len(pred_monthly), w,
                       color=actual_colors, edgecolor="white", label="Actual")
        bars2 = ax.bar(x + w/2, [1] * len(pred_monthly), w,
                       color=pred_colors,  edgecolor="white", label="Predicted (RF)")

        for bar, val in zip(bars1, pred_monthly["Actual"]):
            ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                    val.capitalize(), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        for bar, val in zip(bars2, pred_monthly["RF"]):
            ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                    val.capitalize(), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(pred_monthly["Month"])
        ax.set_yticks([])
        ax.set_ylim(0, 1.3)
        ax.set_title("RQ4: Monthly Risk Predictions vs Actual (2025)\n"
                     "Left bar = Actual | Right bar = Predicted (RF)",
                     fontsize=12, fontweight="bold")
        ax.legend(handles=legend_elements, title="Risk Class",
                  loc="upper right", fontsize=9)
        fig.tight_layout()
        save_cls(fig, "RQ4_monthly_predictions.png")

    return {
        "gnb_pipe":   gnb_pipe,
        "svm_best":   svm_best,
        "rf_best":    rf_best,
        "compare_df": compare_df,
        "pred_df":    pred_df,
        "gnb_cv":     gnb_cv,
        "svm_cv":     svm_cv,
        "rf_cv":      rf_cv,
    }
 
 
# ==============================================================================
# RQ3: CLASSIFY REGION-MONTHS INTO RISK CLASSES
# Features: demographic + per-capita alcohol (14 features, no sin/cos)
# Region signal is captured implicitly through demographic and alcohol features,
# not through explicit region dummies — encourages model to learn drivers.
# ==============================================================================
 
rq3_results = run_classification(cls_df, FEATURES_RQ3, "RQ3")
 
# ==============================================================================
# RQ4: CLASSIFY MONTHS INTO RISK CLASSES (SEASONAL RESOURCE PLANNING)
#
# RQ4 operates at a different granularity from RQ3.
# Instead of region-month observations (72 obs), RQ4 aggregates to
# month level (24 obs: 12 months x 2 years) by averaging assault rate
# and alcohol supply features across all 6 regions.
#
# This ensures the risk label reflects seasonal timing rather than
# regional characteristics. A separate risk label is created from the
# month-level assault rate so that Barkly (persistently High at region level)
# does not dominate the seasonal classification.
#
# Limitation: 24 observations is a small dataset. Results should be
# interpreted with caution and treated as indicative rather than definitive.
# ==============================================================================

print()
print("=" * 60)
print("STEP: Build RQ4 month-level dataset")
print("=" * 60)

# Aggregate cls_df from region-month to month level
rq4_df = (
    cls_df.groupby(["Year", "Month number"])
    .agg(
        assault_rate                       = ("assault_rate",                      "mean"),
        Alcohol_involvement                = ("Alcohol_involvement",               "mean"),
        DV_involvement                     = ("DV_involvement",                    "mean"),
        alcohol_per_capita                 = ("alcohol_per_capita",                "mean"),
        full_strength_beer_pac_per_capita  = ("full_strength_beer_pac_per_capita", "mean"),
        mid_strength_beer_pac_per_capita   = ("mid_strength_beer_pac_per_capita",  "mean"),
        low_strength_beer_pac_per_capita   = ("low_strength_beer_pac_per_capita",  "mean"),
        cask_wine_pac_per_capita           = ("cask_wine_pac_per_capita",          "mean"),
        bottled_wine_pac_per_capita        = ("bottled_wine_pac_per_capita",       "mean"),
        fortified_wine_pac_per_capita      = ("fortified_wine_pac_per_capita",     "mean"),
        sin_month                          = ("sin_month",                         "first"),
        cos_month                          = ("cos_month",                         "first"),
    )
    .reset_index()
)

# Create risk label from month-level assault rate (NOT region-month level)
# Tertile thresholds computed from 2024 training months only — no leakage
train_rq4    = rq4_df[rq4_df["Year"] == 2024]
q_edges_rq4  = train_rq4["assault_rate"].quantile([0, 0.33, 0.66, 1]).to_numpy().copy()
q_edges_rq4[0]  = -np.inf   # ensure no observation falls outside range
q_edges_rq4[-1] =  np.inf

rq4_df["risk_class"] = pd.cut(
    rq4_df["assault_rate"],
    bins=q_edges_rq4,
    labels=RISK_ORDER,
    include_lowest=True
)
rq4_df = rq4_df.dropna(subset=["risk_class"]).copy()
rq4_df["risk_class"] = rq4_df["risk_class"].astype(str)
rq4_df = rq4_df.sort_values(["Year", "Month number"]).reset_index(drop=True)

print(f"RQ4 dataset: {len(rq4_df)} rows (12 months x 2 years)")
print(f"Quantile edges from 2024 months (33rd/66th percentile): {q_edges_rq4[1:-1].round(2)}")
print("\nRQ4 class distribution:")
print(rq4_df.groupby(["Year", "risk_class"]).size().unstack(fill_value=0))

rq4_results = run_classification(rq4_df, FEATURES_RQ4, "RQ4", n_splits=2)
 
# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
 
print()
print("=" * 60)
print("PIPELINE COMPLETE - SUMMARY")
print("=" * 60)
 
print("\nRQ3 - Model Comparison (Test Set 2025):")
print(rq3_results["compare_df"].to_string())
 
print("\nRQ4 - Model Comparison (Test Set 2025):")
print(rq4_results["compare_df"].to_string())
 
print(f"\nEDA plots saved to       : {EDA_DIR}/")
print(f"Classification plots saved: {CLASS_DIR}/")