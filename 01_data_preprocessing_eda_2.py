"""
PRT564 - Data Analytics and Visualisation
Assessment 2: Group Project Presentation
Group 1

Script: 01_data_preprocessing.py
Purpose: Load, clean, merge all source datasets, then run EDA for RQ1 and RQ2.

Key design decision:
    All crime rows are remapped to the 6 NT Government population regions
    (Greater Darwin, Central Australia, Big Rivers, East Arnhem, Barkly, Top End)
    using Reporting Region and SA2 values. This ensures crime rate per 100k
    is calculated against the correct population denominator for each region.

Inputs (place all files in the same folder as this script):
    nt_crime_statistics_dec_2025.csv
    nt-government-regions_1986-to-2025.xlsx
    wholesale-alcohol-supply-by-quarter-2023.xlsx
    wholesale-alcohol-supply-by-quarter-2024.xlsx
    wholesale-alcohol-supply-by-quarter-2025.xlsx

Outputs:
    nt_crime_merged.csv
    eda_plots/*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from numpy.polynomial.polynomial import polyfit

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = DATA_DIR
PLOT_DIR = os.path.join(OUT_DIR, "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def path(filename):
    return os.path.join(DATA_DIR, filename)

def save(fig, name):
    fig.savefig(os.path.join(PLOT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


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

merged.to_csv(os.path.join(OUT_DIR, "nt_crime_merged.csv"), index=False)
print(f"Saved: nt_crime_merged.csv  ({merged.shape[0]:,} rows x {merged.shape[1]} cols)")
print()
print("Starting EDA...")


# == EDA =======================================================================
# Section 1 — Dataset Overview
# Section 2 — RQ1: Monthly Crime Patterns
# Section 3 — RQ2: Assault Analysis & Predictors
# ==============================================================================

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
MONTH_LABELS   = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
YEAR_PALETTE   = {2024: "#2196F3", 2025: "#FF9800"}

df = merged.copy()
df["Month_label"] = df["Month number"].apply(lambda m: MONTH_LABELS[m - 1])
assault = df[df["Offence category"] == "02 Assault"].copy()
pop_plot = df.groupby(["Region", "Year"])["Total_population"].first().reset_index()


# ── Section 1: Dataset Overview ───────────────────────────────────────────────
print()
print("=" * 60)
print("EDA SECTION 1: Dataset Overview")
print("=" * 60)

# 1.1 Population by region — grouped bar 2024 vs 2025
pop_pivot = pop_plot.pivot(index="Region", columns="Year", values="Total_population").loc[REGION_ORDER]

x = np.arange(len(REGION_ORDER))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.barh(x + w/2, pop_pivot[2024] / 1000, w, label="2024", color="#2196F3", edgecolor="white")
b2 = ax.barh(x - w/2, pop_pivot[2025] / 1000, w, label="2025", color="#FF9800", edgecolor="white")
ax.set_yticks(x)
ax.set_yticklabels(REGION_ORDER)
ax.set_xlabel("Population (thousands)")
ax.set_title("Population by NT Government Region: 2024 vs 2025")
ax.legend(title="Year")
ax.set_xlim(0, pop_pivot.max().max() / 1000 * 1.2)
ax.invert_yaxis()
for bar, val in zip(b1, pop_pivot[2024]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8)
for bar, val in zip(b2, pop_pivot[2025]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8)
fig.tight_layout()
save(fig, "1_1_population_by_region.png")

# 1.2 Offences by category — side-by-side 2024 vs 2025
cat_by_year = (df.groupby(["Offence category", "Year"])["Number of offences"]
               .sum().unstack(fill_value=0))
cat_by_year.index = [c.split(" ", 1)[1] for c in cat_by_year.index]
cat_by_year = cat_by_year.sort_values(2024, ascending=True)

x = np.arange(len(cat_by_year))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.barh(x + w/2, cat_by_year[2024], w, label="2024", color="#2196F3", edgecolor="white")
b2 = ax.barh(x - w/2, cat_by_year[2025], w, label="2025", color="#FF9800", edgecolor="white")
ax.set_yticks(x)
ax.set_yticklabels(cat_by_year.index)
ax.set_xlabel("Number of Offences")
ax.set_title("Offences by Category: 2024 vs 2025")
ax.legend(title="Year")
ax.set_xlim(0, cat_by_year.max().max() * 1.18)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.invert_yaxis()
for bar, val in zip(b1, cat_by_year[2024]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8)
for bar, val in zip(b2, cat_by_year[2025]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8)
fig.tight_layout()
save(fig, "1_2_offences_by_category.png")

# 1.3 Alcohol & DV involvement
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
    labels = [f"Yes\n{involved:,}\n({involved/total*100:.1f}%)",
              f"No / N/A\n{not_involved:,}\n({not_involved/total*100:.1f}%)"]
    ax.bar(labels, [involved, not_involved], color=colors, edgecolor="white", width=0.5)
    ax.set_title(title)
    ax.set_ylabel("Number of Offences")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.suptitle("Alcohol and DV Involvement Across All Offences (2024-2025)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "1_3_alcohol_dv_involvement.png")


# ── Section 2: RQ1 — Monthly Crime Patterns ───────────────────────────────────
print()
print("=" * 60)
print("EDA SECTION 2: RQ1 - Monthly Crime Patterns")
print("=" * 60)

monthly_all = (df.groupby("Month number")["Number of offences"]
               .sum().reset_index())
monthly_all["Month_label"] = monthly_all["Month number"].apply(
    lambda m: MONTH_LABELS[m - 1])

pop_year = df.groupby(["Year", "Region"])["Total_population"].first().reset_index()
total_pop_per_year = pop_year.groupby("Year")["Total_population"].sum()
monthly_yr = (df.groupby(["Year", "Month number"])["Number of offences"]
              .sum().reset_index())
monthly_yr["Total_pop"]     = monthly_yr["Year"].map(total_pop_per_year)
monthly_yr["Rate_per_100k"] = (monthly_yr["Number of offences"]
                                / monthly_yr["Total_pop"] * 100000).round(1)
monthly_yr["Month_label"]   = monthly_yr["Month number"].apply(
    lambda m: MONTH_LABELS[m - 1])

# 2.1 Total offences by month
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#EF5350" if v == monthly_all["Number of offences"].max()
          else "#42A5F5" for v in monthly_all["Number of offences"]]
bars = ax.bar(monthly_all["Month_label"], monthly_all["Number of offences"],
              color=colors, edgecolor="white")
ax.set_xlabel("Month")
ax.set_ylabel("Total Number of Offences")
ax.set_title("Total Offences by Month - All Categories (2024-2025)")
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
save(fig, "2_1_offences_by_month.png")

# 2.2 Crime rate per 100k by month
monthly_rate = (monthly_yr.groupby("Month number")
                .agg(Rate_per_100k=("Rate_per_100k", "mean"),
                     Month_label=("Month_label", "first"))
                .reset_index().sort_values("Month number"))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#EF5350" if v == monthly_rate["Rate_per_100k"].max()
          else "#66BB6A" for v in monthly_rate["Rate_per_100k"]]
bars = ax.bar(monthly_rate["Month_label"], monthly_rate["Rate_per_100k"],
              color=colors, edgecolor="white")
ax.set_xlabel("Month")
ax.set_ylabel("Offences per 100,000 Population")
ax.set_title("Average Monthly Crime Rate per 100,000 Population (2024-2025)")
for bar, val in zip(bars, monthly_rate["Rate_per_100k"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(0, monthly_rate["Rate_per_100k"].max() * 1.15)
fig.tight_layout()
save(fig, "2_2_crime_rate_per_100k_by_month.png")

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
save(fig, "2_3_monthly_trend_2024_vs_2025.png")

# 2.4 Heatmap: Month x Offence category
heat_data = (df.groupby(["Offence category", "Month number"])["Number of offences"]
             .sum().unstack(fill_value=0))
heat_data.index   = [c.split(" ", 1)[1] for c in heat_data.index]
heat_data.columns = MONTH_LABELS

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(heat_data, annot=True, fmt=",d", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Number of Offences"})
ax.set_title("Offences by Month and Category - Heatmap (2024-2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Offence Category")
fig.tight_layout()
save(fig, "2_4_heatmap_month_category.png")


# ── Section 3: RQ2 — Assault Analysis & Predictors ────────────────────────────
print()
print("=" * 60)
print("EDA SECTION 3: RQ2 - Assault Analysis & Predictors")
print("=" * 60)

assault_agg = (
    assault.groupby(["Year", "Quarter", "Month number", "Region"])
    .agg(
        Assault_offences =("Number of offences", "sum"),
        Alcohol_involved =("Alcohol involvement", "sum"),
        DV_involved      =("DV involvement", "sum"),
        Total_PAC        =("Total PAC", "first"),
        Total_population =("Total_population", "first"),
        Aboriginal       =("Aboriginal", "first"),
    )
    .reset_index()
)
assault_agg["Assault_rate_100k"] = (
    assault_agg["Assault_offences"] / assault_agg["Total_population"] * 100000
).round(1)
assault_agg["Month_label"] = assault_agg["Month number"].apply(
    lambda m: MONTH_LABELS[m - 1])

# 3.1 Assault by region: side-by-side 2024 vs 2025
# Raw counts
assault_region_yr = (assault_agg.groupby(["Region", "Year"])
                     .agg(Total_assaults=("Assault_offences", "sum"))
                     .reset_index())
counts_pivot = assault_region_yr.pivot(index="Region", columns="Year",
                                       values="Total_assaults").loc[REGION_ORDER]

# Quarterly rate averaged per year per region
rate_pivot = (assault_agg.groupby(["Region", "Year"])["Assault_rate_100k"]
              .mean().round(1).unstack()).loc[REGION_ORDER]

x = np.arange(len(REGION_ORDER))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: raw counts
b1 = axes[0].barh(x + w/2, counts_pivot[2024], w, label="2024", color="#2196F3", edgecolor="white")
b2 = axes[0].barh(x - w/2, counts_pivot[2025], w, label="2025", color="#FF9800", edgecolor="white")
axes[0].set_yticks(x)
axes[0].set_yticklabels(REGION_ORDER)
axes[0].set_xlabel("Total Assault Offences")
axes[0].set_title("Total Assault Offences by Region: 2024 vs 2025")
axes[0].legend(title="Year")
axes[0].set_xlim(0, counts_pivot.max().max() * 1.2)
axes[0].invert_yaxis()
for bar, val in zip(b1, counts_pivot[2024]):
    axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8)
for bar, val in zip(b2, counts_pivot[2025]):
    axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8)

# Right: quarterly rate averaged per year
b3 = axes[1].barh(x + w/2, rate_pivot[2024], w, label="2024", color="#2196F3", edgecolor="white")
b4 = axes[1].barh(x - w/2, rate_pivot[2025], w, label="2025", color="#FF9800", edgecolor="white")
axes[1].set_yticks(x)
axes[1].set_yticklabels(REGION_ORDER)
axes[1].set_xlabel("Avg Quarterly Assault Rate per 100,000 Population")
axes[1].set_title("Avg Quarterly Assault Rate per 100k: 2024 vs 2025")
axes[1].legend(title="Year")
axes[1].set_xlim(0, rate_pivot.max().max() * 1.2)
axes[1].invert_yaxis()
for bar, val in zip(b3, rate_pivot[2024]):
    axes[1].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, f"{val:,.0f}", va="center", fontsize=8)
for bar, val in zip(b4, rate_pivot[2025]):
    axes[1].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, f"{val:,.0f}", va="center", fontsize=8)

fig.suptitle("Assault Offences by Region: Raw Counts vs Quarterly Rate per 100k",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "3_1_assault_by_region.png")

# Store for summary
region_assault = pd.DataFrame({
    "Total_assaults": counts_pivot.sum(axis=1),
    "Avg_rate": rate_pivot.mean(axis=1)
})

# 3.2 Assault trend by month x region
assault_monthly_region = (assault_agg.groupby(["Month number", "Region"])
                          ["Assault_offences"].mean().reset_index())
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
save(fig, "3_2_assault_trend_by_region.png")

# 3.3 Distribution + log transform
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(assault_agg["Assault_offences"], bins=30, kde=True,
             ax=axes[0], color="#42A5F5", edgecolor="white")
axes[0].set_title(
    f"Distribution of Assault Offences\n(Skewness = {assault_agg['Assault_offences'].skew():.2f})")
axes[0].set_xlabel("Assault Offences")
axes[0].set_ylabel("Frequency")

log_vals = np.log1p(assault_agg["Assault_offences"])
sns.histplot(log_vals, bins=30, kde=True,
             ax=axes[1], color="#66BB6A", edgecolor="white")
axes[1].set_title(f"Log-Transformed Assault Offences\n(Skewness = {log_vals.skew():.2f})")
axes[1].set_xlabel("log(1 + Assault Offences)")
axes[1].set_ylabel("Frequency")

fig.suptitle("Assault Offences Distribution: Original vs Log Transform",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "3_3_assault_distribution.png")

# 3.4 PAC trend by quarter x region
pac_trend = (assault_agg.groupby(["Year", "Quarter", "Region"])
             .agg(Total_PAC=("Total_PAC", "first")).reset_index())
pac_trend["YQ"] = (pac_trend["Year"].astype(str) + "-Q"
                   + pac_trend["Quarter"].astype(str))

fig, ax = plt.subplots(figsize=(12, 6))
for region, color in zip(REGION_ORDER, REGION_PALETTE):
    grp = (pac_trend[pac_trend["Region"] == region]
           .sort_values(["Year", "Quarter"]))
    ax.plot(grp["YQ"], grp["Total_PAC"] / 1000,
            marker="o", linewidth=2, label=region, color=color)
ax.set_xlabel("Year-Quarter")
ax.set_ylabel("Total PAC (thousands of litres)")
ax.set_title("Wholesale Alcohol Supply (Total PAC) by Region and Quarter (2024-2025)")
ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.tick_params(axis="x", rotation=30)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}k"))
fig.tight_layout()
save(fig, "3_4_pac_trend_by_region.png")

# 3.5 Scatter: PAC vs Assault
fig, ax = plt.subplots(figsize=(9, 6))
for region, color in zip(REGION_ORDER, REGION_PALETTE):
    grp = assault_agg[assault_agg["Region"] == region]
    ax.scatter(grp["Total_PAC"] / 1000, grp["Assault_offences"],
               label=region, color=color, alpha=0.75, s=60, edgecolors="white")
x = assault_agg["Total_PAC"] / 1000
y = assault_agg["Assault_offences"]
c, m = polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, m * x_line + c, color="black", linewidth=1.5,
        linestyle="--", label="Overall trend")
corr = assault_agg[["Total_PAC", "Assault_offences"]].corr().iloc[0, 1]
ax.set_xlabel("Total PAC (thousands of litres)")
ax.set_ylabel("Assault Offences")
ax.set_title(f"Alcohol Supply (PAC) vs Assault Offences\n(Pearson r = {corr:.3f})")
ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
save(fig, "3_5_scatter_pac_vs_assault.png")

# 3.6 Correlation heatmap
corr_df = assault_agg[["Assault_offences", "Total_PAC", "Total_population",
                        "Aboriginal", "Alcohol_involved", "DV_involved"]].copy()
corr_df.rename(columns={
    "Assault_offences": "Assault Offences",
    "Total_PAC":        "Total PAC",
    "Total_population": "Total Population",
    "Aboriginal":       "Aboriginal Pop.",
    "Alcohol_involved": "Alcohol Involved",
    "DV_involved":      "DV Involved",
}, inplace=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Pearson Correlation"})
ax.set_title("Correlation Heatmap - Assault Predictors (RQ2)")
fig.tight_layout()
save(fig, "3_6_correlation_heatmap.png")


# ── EDA Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)
peak_m     = monthly_all.loc[monthly_all["Number of offences"].idxmax(), "Month_label"]
top_region = region_assault["Avg_rate"].idxmax()
skew_orig  = assault_agg["Assault_offences"].skew()
skew_log   = np.log1p(assault_agg["Assault_offences"]).skew()
print(f"RQ1: Peak crime month          = {peak_m}")
print(f"RQ2: Highest assault rate/100k = {top_region} "
      f"({region_assault.loc[top_region, 'Avg_rate']:,.0f})")
print(f"RQ2: PAC vs Assault r          = {corr:.3f}")
print(f"RQ2: Skewness original={skew_orig:.2f}, log={skew_log:.2f}")
print(f"     -> Log transform recommended for regression")
print(f"\nPlots saved to: {PLOT_DIR}")
