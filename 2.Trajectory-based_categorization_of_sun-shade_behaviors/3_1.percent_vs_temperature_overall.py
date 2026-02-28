# ==========================================
# File: plot_daily_category_ratio.py
# Function: Calculate the daily proportion of 4 categories and plot (scatter + linear fit)
#           Legend shows scatter + slope
#           Additionally merge two categories (sun/shade-preferring) into the CSV
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###########################################
# Path configuration
###########################################
output_folder = "./pedestrian_category"
weather_file = "../data/3.weather_info/fig3_daily-info_update.csv"
output_ratio_file = "category_percent_vs_temperature_overall.csv"

# Figure save folder
figure_folder = "./figure"
os.makedirs(figure_folder, exist_ok=True)

###########################################
# Calculate daily category proportions (excluding Remaining Others)
###########################################
daily_results = []

for fname in sorted(os.listdir(output_folder)):
    if not fname.endswith(".csv"):
        continue

    date_str = fname.replace(".csv", "")
    file_path = os.path.join(output_folder, fname)
    df = pd.read_csv(file_path)

    # Remove Others
    df_filtered = df[df['category'] != 'Others']
    if len(df_filtered) == 0:
        continue

    counts = df_filtered['category'].value_counts()
    total = counts.sum()

    heliophile = counts.get('Heliophile', 0) / total * 100
    photophobic = counts.get('Photophobic', 0) / total * 100
    sun_chaser = counts.get('Sun-chaser', 0) / total * 100
    shade_chaser = counts.get('Shade-chaser', 0) / total * 100

    # Merge two categories
    sun_preferring = heliophile + sun_chaser
    shade_preferring = photophobic + shade_chaser

    daily_results.append({
        'file': date_str,
        'Heliophile(%)': heliophile,
        'Photophobic(%)': photophobic,
        'Sun-chaser(%)': sun_chaser,
        'Shade-chaser(%)': shade_chaser,
        'Sun-preferring(%)': sun_preferring,
        'Shade-preferring(%)': shade_preferring
    })

###########################################
# Merge the first 5 columns of weather file
###########################################
weather_df = pd.read_csv(weather_file).iloc[:, :5]
ratio_df = pd.DataFrame(daily_results)

merged_df = pd.concat([weather_df, ratio_df.iloc[:, 1:]], axis=1)
merged_df.to_csv(output_ratio_file, index=False, encoding='utf-8-sig', float_format="%.4f")

print("\n✅ Daily category proportions saved to:", output_ratio_file)

###########################################
# Plot Temperature vs Proportion (scatter + linear fit)
###########################################
sns.set(style="whitegrid")
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13
})

x = merged_df['Apparent_Temp_C'].values

# === Four-category proportion plot ===
plt.figure(figsize=(9, 6))

categories = {
    'Heliophile(%)': ('gold', 'Heliophile'),
    'Photophobic(%)': ('gray', 'Photophobic'),
    'Sun-chaser(%)': ('orange', 'Sun-chaser'),
    'Shade-chaser(%)': ('teal', 'Shade-chaser')
}

scatter_handles = []

for col, (color, label) in categories.items():
    y = merged_df[col].values

    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        fit_line = np.polyval(coeffs, np.sort(x))
    else:
        slope = np.nan

    scatter = plt.scatter(
        x, y,
        color=color,
        s=80,
        alpha=0.8,
        label=f"{label} (slope = {slope:.2f})"
    )
    scatter_handles.append(scatter)

    if len(x) > 1:
        plt.plot(np.sort(x), fit_line, color=color, linewidth=3)

plt.xlabel("Temperature (°C)")
plt.ylabel("Proportion (%)")
plt.title("Pedestrian Category Proportion vs Temperature")
plt.legend(handles=scatter_handles, loc='upper right')
plt.tight_layout()

fig_path1 = os.path.join(figure_folder, "Category_vs_Temperature.png")
plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
print(f"Figure saved: {fig_path1}")
plt.show()

# === Merged two-category proportion plot ===
sun_like = merged_df['Sun-preferring(%)'].values
shade_like = merged_df['Shade-preferring(%)'].values

plt.figure(figsize=(9, 6))

coeff_sun = np.polyfit(x, sun_like, 1)
coeff_shade = np.polyfit(x, shade_like, 1)

scatter_sun = plt.scatter(
    x, sun_like,
    color='gold',
    s=80,
    alpha=0.8,
    label=f"Sun-preferring (slope = {coeff_sun[0]:.2f})"
)

scatter_shade = plt.scatter(
    x, shade_like,
    color='gray',
    s=80,
    alpha=0.8,
    label=f"Shade-preferring (slope = {coeff_shade[0]:.2f})"
)

plt.plot(np.sort(x), np.polyval(coeff_sun, np.sort(x)), color='gold', linewidth=3)
plt.plot(np.sort(x), np.polyval(coeff_shade, np.sort(x)), color='gray', linewidth=3)

plt.xlabel("Temperature (°C)")
plt.ylabel("Proportion (%)")
plt.title("Pedestrian Preference vs Temperature")
plt.legend(loc='upper right')
plt.tight_layout()

fig_path2 = os.path.join(figure_folder, "Sun_vs_Shade_Proportion.png")
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
print(f"Figure saved: {fig_path2}")
plt.show()

###########################################
# Append fitted points to the original CSV (only 2 rows)
###########################################
df = merged_df.copy()

x_fit = [24, 48]
x_col = "Apparent_Temp_C"

fit_y_cols = [
    'Heliophile(%)',
    'Photophobic(%)',
    'Sun-chaser(%)',
    'Shade-chaser(%)',
    'Sun-preferring(%)',
    'Shade-preferring(%)'
]

df["x_fit_ApparentTemp"] = np.nan
df.loc[0, "x_fit_ApparentTemp"] = float(x_fit[0])
df.loc[1, "x_fit_ApparentTemp"] = float(x_fit[1])

x_data = df[x_col].values.astype(float)

for y_col in fit_y_cols:
    fit_col_name = f"fit_{y_col}"
    df[fit_col_name] = np.nan

    y_data = df[y_col].values.astype(float)
    coeffs = np.polyfit(x_data, y_data, 1)
    y_fit = np.polyval(coeffs, x_fit)

    df.loc[0, fit_col_name] = float(y_fit[0])
    df.loc[1, fit_col_name] = float(y_fit[1])

df.to_csv(
    output_ratio_file,
    index=False,
    encoding='utf-8-sig',
    float_format="%.4f"
)

print("Appended 1 x column + 6 fitted y columns to the original CSV")