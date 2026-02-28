import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###########################################
# Path configuration
###########################################
ped_folder = "./pedestrian_characteristics"
weather_file = "../data/3.weather_info/fig3_daily-info_update.csv"

output_lingering_file = "lingering_vs_temperature_sun_shade.csv"

figure_folder = "./figure"
os.makedirs(figure_folder, exist_ok=True)

keep_categories = ["Sun-chaser", "Shade-chaser"]

###########################################
# Daily mean lingering (Sun / Shade, temperature)
###########################################
daily_results = []

for fname in sorted(os.listdir(ped_folder)):
    if not fname.endswith("_pedestrian_trajectory.csv"):
        continue

    date_str = fname.replace("_pedestrian_trajectory.csv", "")
    df = pd.read_csv(os.path.join(ped_folder, fname))
    df = df[df["category"].isin(keep_categories)]
    if len(df) == 0:
        continue

    result = {"file": date_str}

    for cat in keep_categories:
        sub = df[df["category"] == cat]
        result[f"{cat}_sun"] = sub["lingering_sun"].mean()
        result[f"{cat}_shade"] = sub["lingering_shade"].mean()

    daily_results.append(result)

weather_df = pd.read_csv(weather_file).iloc[:, :5]
lingering_df = pd.DataFrame(daily_results)
merged_df = pd.concat([weather_df, lingering_df.iloc[:, 1:]], axis=1)

###########################################
# Append fitted points to temperature-lingering CSV
###########################################
df_fit = merged_df.copy()

x_fit = [24, 48]
x_col = "Apparent_Temp_C"

df_fit["x_fit_ApparentTemp"] = np.nan
df_fit.loc[0, "x_fit_ApparentTemp"] = float(x_fit[0])
df_fit.loc[1, "x_fit_ApparentTemp"] = float(x_fit[1])

fit_cols = [
    "Sun-chaser_sun",
    "Sun-chaser_shade",
    "Shade-chaser_sun",
    "Shade-chaser_shade"
]

x_data = df_fit[x_col].values.astype(float)

for col in fit_cols:
    fit_col_name = f"fit_{col}"
    df_fit[fit_col_name] = np.nan

    y_data = df_fit[col].values.astype(float)
    mask = ~np.isnan(x_data) & ~np.isnan(y_data)

    if mask.sum() > 1:
        k, b = np.polyfit(x_data[mask], y_data[mask], 1)
        y_fit = k * np.array(x_fit) + b

        df_fit.loc[0, fit_col_name] = float(y_fit[0])
        df_fit.loc[1, fit_col_name] = float(y_fit[1])

df_fit.to_csv(
    output_lingering_file,
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

###########################################
# Plot settings (unchanged)
###########################################
sns.set(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11
})

###########################################
# Figure: Temperature vs Mean Lingering (4 lines)
###########################################
plt.figure(figsize=(7, 6))

x = merged_df["Apparent_Temp_C"]

line_colors = {
    "Sun-chaser_sun": "#D55E00",
    "Sun-chaser_shade": "#0072B2",
    "Shade-chaser_sun": "#CC79A7",
    "Shade-chaser_shade": "#009E73"
}

for cat in keep_categories:
    for env in ["sun", "shade"]:
        y = merged_df[f"{cat}_{env}"]
        mask = ~np.isnan(x) & ~np.isnan(y)
        c = line_colors[f"{cat}_{env}"]

        if mask.sum() > 1:
            k, b = np.polyfit(x[mask], y[mask], 1)
            xs = np.sort(x[mask])
            plt.plot(xs, k * xs + b, lw=2.8, color=c,
                     label=f"{cat} ({env}, slope={k:.3f})")

        plt.scatter(x[mask], y[mask], s=60, alpha=0.8, color=c)

plt.xlabel("Temperature (°C)")
plt.ylabel("Mean Lingering Time (s)")
plt.title("Mean Lingering Time vs Temperature")
plt.legend(frameon=True)

plt.tight_layout()

save_path = os.path.join(
    figure_folder,
    "Sun_Shade_Chaser_Lingering_vs_Temperature.png"
)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Temperature-lingering CSV (including fitted points) saved: {output_lingering_file}")
print(f"Figure saved: {save_path}")