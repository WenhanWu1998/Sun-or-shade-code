import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ====================== NEW: create figure folder ======================
figure_folder = "./figure"
os.makedirs(figure_folder, exist_ok=True)

# ====================== Load data ======================
data_file = "shade_occupancy_by_weather_conditions.csv"
df = pd.read_csv(data_file)

# ====================== Rename columns 2–5 ======================
df.rename(columns={
    df.columns[1]: "Temperature",
    df.columns[2]: "Feel Like Temp",
    df.columns[3]: "Rain Probability",
    df.columns[4]: "Wind Speed"
}, inplace=True)

# Candidate columns for X-axis
x_columns = ["Temperature", "Feel Like Temp", "Rain Probability", "Wind Speed"]
y_columns = ['spatial_percent_bias', 'temporal_percent_bias']

# Global style with enlarged fonts
sns.set(style="whitegrid")
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# ====================== Figure 1: spatial_percent_bias ======================
plt.figure(figsize=(16, 12))
for i, x_col in enumerate(x_columns):
    plt.subplot(2, 2, i+1)
    x = df[x_col].values
    y = df[y_columns[0]].values  # spatial_percent_bias

    sns.scatterplot(x=x, y=y, s=70)

    # Fit line & compute Pearson r
    coeffs = np.polyfit(x, y, 1)
    fit_line = np.polyval(coeffs, x)
    r = np.corrcoef(x, y)[0, 1]

    plt.plot(x, fit_line, color='red', linestyle='--',
             label=f'r = {r:.3f}')

    plt.xlabel(x_col, fontsize=16)
    plt.ylabel(y_columns[0], fontsize=16)
    plt.title(f'{y_columns[0]} vs {x_col}', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.suptitle('Spatial Percent vs Weather Info', fontsize=20, y=1.02)
plt.savefig(os.path.join(figure_folder, "spatial_percent_bias_vs_weather.png"),
            dpi=300, bbox_inches='tight')
plt.show()

# ====================== Figure 2: temporal_percent_bias ======================
plt.figure(figsize=(16, 12))
for i, x_col in enumerate(x_columns):
    plt.subplot(2, 2, i + 1)
    x = df[x_col].values
    y = df[y_columns[1]].values  # temporal_percent_bias

    sns.scatterplot(x=x, y=y, s=70)

    # Fit line & compute Pearson r
    coeffs = np.polyfit(x, y, 1)
    fit_line = np.polyval(coeffs, x)
    r = np.corrcoef(x, y)[0, 1]

    plt.plot(x, fit_line, color='red', linestyle='--',
             label=f'r = {r:.3f}')

    plt.xlabel(x_col, fontsize=16)
    plt.ylabel(y_columns[1], fontsize=16)
    plt.title(f'{y_columns[1]} vs {x_col}', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.suptitle('Temporal Percent vs Weather Info', fontsize=20, y=1.02)
plt.savefig(os.path.join(figure_folder, "temporal_percent_bias_vs_weather.png"),
            dpi=300, bbox_inches='tight')
plt.show()


# ====================== Append fitted-point columns to original CSV (only 2 rows) ======================

fit_x_dict = {
    "Temperature": [22, 38],
    "Feel Like Temp": [24, 48],
    "Rain Probability": [0, 100],
    "Wind Speed": [1, 9]
}

n_rows = len(df)

for x_col, x_fit in fit_x_dict.items():
    # First create empty columns (NaN)
    df[f"x_fit_{x_col}"] = np.nan
    df[f"spatial_fit_{x_col}"] = np.nan
    df[f"temporal_fit_{x_col}"] = np.nan

    x_data = df[x_col].values.astype(float)

    # Fit
    coeffs_spatial = np.polyfit(x_data, df['spatial_percent_bias'].values.astype(float), 1)
    coeffs_temporal = np.polyfit(x_data, df['temporal_percent_bias'].values.astype(float), 1)

    y_spatial_fit = np.polyval(coeffs_spatial, x_fit)
    y_temporal_fit = np.polyval(coeffs_temporal, x_fit)

    # Write fitted points only in the first two rows; keep others as NaN
    df.loc[0, f"x_fit_{x_col}"] = float(x_fit[0])
    df.loc[1, f"x_fit_{x_col}"] = float(x_fit[1])

    df.loc[0, f"spatial_fit_{x_col}"] = float(y_spatial_fit[0])
    df.loc[1, f"spatial_fit_{x_col}"] = float(y_spatial_fit[1])

    df.loc[0, f"temporal_fit_{x_col}"] = float(y_temporal_fit[0])
    df.loc[1, f"temporal_fit_{x_col}"] = float(y_temporal_fit[1])

# Overwrite the same CSV file (data is preserved)
df.to_csv(data_file, index=False)

print("✅ 12 fitted-point columns have been appended to the original CSV (values only in the first two rows)")