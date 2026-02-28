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

output_speed_file = "speed_vs_temperature_category.csv"

figure_folder = "./figure"
os.makedirs(figure_folder, exist_ok=True)

keep_categories = ["Heliophile", "Photophobic", "Sun-chaser", "Shade-chaser"]

###########################################
# Calculate daily mean speed by category (for temperature plot, m/s)
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
        result[f"{cat}_speed"] = sub["speed_total"].mean() if len(sub) > 0 else np.nan

    daily_results.append(result)

weather_df = pd.read_csv(weather_file).iloc[:, :5]
speed_df = pd.DataFrame(daily_results)
merged_df = pd.concat([weather_df, speed_df.iloc[:, 1:]], axis=1)

###########################################
# Append fit points to CSV (x = 24, 48 °C)
###########################################
df_fit = merged_df.copy()

x_fit = [24, 48]
x_col = "Apparent_Temp_C"

df_fit["x_fit_ApparentTemp"] = np.nan
df_fit.loc[0, "x_fit_ApparentTemp"] = float(x_fit[0])
df_fit.loc[1, "x_fit_ApparentTemp"] = float(x_fit[1])

x_data = df_fit[x_col].values.astype(float)

for cat in keep_categories:
    y_col = f"{cat}_speed"
    fit_col_name = f"fit_{cat}_speed"

    df_fit[fit_col_name] = np.nan
    y_data = df_fit[y_col].values.astype(float)

    mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    if mask.sum() < 2:
        continue

    coeffs = np.polyfit(x_data[mask], y_data[mask], 1)
    y_fit = np.polyval(coeffs, x_fit)

    df_fit.loc[0, fit_col_name] = float(y_fit[0])
    df_fit.loc[1, fit_col_name] = float(y_fit[1])

df_fit.to_csv(
    output_speed_file,
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

###########################################
# Plot settings
###########################################
sns.set(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

###########################################
# Temperature vs mean speed (m/s)
###########################################
plt.figure(figsize=(6, 6))

x = merged_df["Apparent_Temp_C"]

colors = {
    "Heliophile": "gold",
    "Photophobic": "gray",
    "Sun-chaser": "orange",
    "Shade-chaser": "teal"
}

handles = []

for cat in keep_categories:
    y = merged_df[f"{cat}_speed"]
    mask = ~np.isnan(x) & ~np.isnan(y)

    sc = plt.scatter(
        x[mask], y[mask],
        s=70, alpha=0.8, color=colors[cat]
    )

    if mask.sum() > 1:
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        xs = np.sort(x[mask])
        plt.plot(xs, slope * xs + intercept,
                 color=colors[cat], lw=3)
        label = f"{cat} (slope={slope:.3f})"
    else:
        label = f"{cat} (slope=NA)"

    sc.set_label(label)
    handles.append(sc)

plt.xlabel("Temperature (°C)")
plt.ylabel("Mean Speed (m/s)")
plt.title("Mean Speed vs Temperature")
plt.legend(handles=handles, loc="upper right")

###########################################
# Save figure
###########################################
plt.tight_layout()
save_path = os.path.join(figure_folder, "MeanSpeed_vs_Temperature.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved: {save_path}")
print(f"Temperature-Speed CSV (with fit points) saved: {output_speed_file}")