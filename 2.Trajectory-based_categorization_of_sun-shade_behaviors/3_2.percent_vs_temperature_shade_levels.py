import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###########################################
# CONFIG
###########################################
ped_folder = "./pedestrian_category"
weather_file = "../data/3.weather_info/fig3_daily-info_update.csv"
output_excel = "./category_percent_vs_temperature_with_shade_levels.csv"
figure_folder = "./figure"
os.makedirs(figure_folder, exist_ok=True)

# shadow level bins
bins = [0.00, 0.10, 0.2, 0.3, 1.0]
labels = ["0-10%", "10-20%", "20-30%", "30-100%"]

# pedestrian categories (base)
categories = ["Heliophile", "Photophobic", "Sun-chaser", "Shade-chaser"]

# merged categories
merged_defs = {
    "Sun-preferring": ["Heliophile", "Sun-chaser"],
    "Shade-preferring": ["Photophobic", "Shade-chaser"]
}

###########################################
# LOAD WEATHER FILE (first 3 columns)
###########################################
df_weather = pd.read_csv(weather_file)
df_weather_subset = df_weather.iloc[:, :3].copy()
df_weather_subset.columns = [col.strip() for col in df_weather_subset.columns]

###########################################
# PROCESS EACH DAILY FILE
###########################################
all_data = []

ped_files = sorted([f for f in os.listdir(ped_folder) if f.endswith(".csv")])
for idx, fname in enumerate(ped_files):
    df = pd.read_csv(os.path.join(ped_folder, fname))

    # shadow proportion classification
    df["shadow_class"] = pd.cut(
        df["avg_shadow_proportion"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    row_data = {}

    for shadow_label in labels:
        subset = df[df["shadow_class"] == shadow_label]
        valid_subset = subset[subset["category"].isin(categories)]
        total_count = len(valid_subset)

        if total_count == 0:
            for cat in categories:
                row_data[f"{shadow_label}_{cat}"] = np.nan
            for mcat in merged_defs:
                row_data[f"{shadow_label}_{mcat}"] = np.nan
        else:
            counts = valid_subset["category"].value_counts()

            # base categories — multiply by 100
            for cat in categories:
                row_data[f"{shadow_label}_{cat}"] = (
                    counts.get(cat, 0) / total_count * 100
                )

            # merged categories — multiply by 100
            for mcat, subcats in merged_defs.items():
                merged_count = sum(counts.get(sc, 0) for sc in subcats)
                row_data[f"{shadow_label}_{mcat}"] = (
                    merged_count / total_count * 100
                )

    weather_values = df_weather_subset.iloc[idx].to_dict()
    weather_values.update(row_data)
    all_data.append(weather_values)

###########################################
# SAVE TO CSV (0–100)
###########################################
df_all = pd.DataFrame(all_data)

shadow_cols = (
    [f"{label}_{cat}" for label in labels for cat in categories] +
    [f"{label}_{mcat}" for label in labels for mcat in merged_defs]
)

cols = list(df_weather_subset.columns) + shadow_cols
df_all = df_all[cols]

df_all.to_csv(output_excel, index=False, encoding="utf-8-sig")
print(f"✅ Summary saved to: {output_excel}")

###########################################
# PLOTTING (not multiplied by 100)
###########################################
df = df_all.copy()
x = df.iloc[:, 2].values  # Air_Temp_C
colors = ["red", "blue", "green", "purple"]

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

###########################################
# Original four categories
###########################################
for ax_idx, cat in enumerate(categories):
    ax = axes[ax_idx]
    line_handles = []
    line_labels = []

    for label, color in zip(labels, colors):
        y = df[f"{label}_{cat}"].values  # already in %

        ax.scatter(x, y, color=color, s=80, alpha=0.8)

        valid = ~np.isnan(y)
        if valid.sum() > 1:
            coeffs = np.polyfit(x[valid], y[valid], 1)
            xs = np.sort(x[valid])
            ys = np.polyval(coeffs, xs)
            line, = ax.plot(xs, ys, color=color, linewidth=3)
            line_handles.append(line)
            line_labels.append(f"{label} (slope={coeffs[0]:.2f})")

    ax.set_title(cat)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(line_handles, line_labels, title="Shadow Level")

###########################################
# Merged two categories
###########################################
for i, (merged_name, _) in enumerate(merged_defs.items(), start=4):
    ax = axes[i]
    line_handles = []
    line_labels = []

    for label, color in zip(labels, colors):
        y = df[f"{label}_{merged_name}"].values

        ax.scatter(x, y, color=color, s=80, alpha=0.8)

        valid = ~np.isnan(y)
        if valid.sum() > 1:
            coeffs = np.polyfit(x[valid], y[valid], 1)
            xs = np.sort(x[valid])
            ys = np.polyval(coeffs, xs)
            line, = ax.plot(xs, ys, color=color, linewidth=3)
            line_handles.append(line)
            line_labels.append(f"{label} (slope={coeffs[0]:.2f})")

    ax.set_title(merged_name)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(line_handles, line_labels, title="Shadow Level")

plt.tight_layout()
fig_path = os.path.join(figure_folder, "Category_vs_Temperature_with_shade_levels.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {fig_path}")
plt.show()

###########################################
# Append fitted points to original CSV (still 0–100)
###########################################
df_fit = df_all.copy()

x_fit = [24, 48]
x_col = df_fit.columns[2]  # Air_Temp_C

df_fit["x_fit_AirTemp"] = np.nan
df_fit.loc[0, "x_fit_AirTemp"] = x_fit[0]
df_fit.loc[1, "x_fit_AirTemp"] = x_fit[1]

x_data = df_fit[x_col].values.astype(float)

fit_y_cols = (
    [f"{label}_{cat}" for label in labels for cat in categories] +
    [f"{label}_{mcat}" for label in labels for mcat in merged_defs]
)

for y_col in fit_y_cols:
    fit_col = f"fit_{y_col}"
    df_fit[fit_col] = np.nan

    y_data = df_fit[y_col].values.astype(float)
    valid = ~np.isnan(y_data)

    if valid.sum() > 1:
        coeffs = np.polyfit(x_data[valid], y_data[valid], 1)
        y_fit = np.polyval(coeffs, x_fit)
        df_fit.loc[0, fit_col] = y_fit[0]
        df_fit.loc[1, fit_col] = y_fit[1]

df_fit.to_csv(
    output_excel,
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

print("All data in CSV are 0–100, fitted columns appended")