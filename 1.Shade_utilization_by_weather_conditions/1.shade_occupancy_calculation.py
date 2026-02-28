import os
import pandas as pd
import numpy as np
import ast

# ====================== Configuration ======================
combine_folder = "../data/1.combine_data"
ped_folder = "../data/2.pedestrian_trajectory"
weather_file = "../data/3.weather_info/fig3_daily-info_update.csv"
output_file = "shade_occupancy_by_weather_conditions.csv"
daily_results = []

# ====================== Main loop ======================
for fname in sorted(os.listdir(combine_folder)):
    if not fname.endswith("_pedestrian_shadow_information.csv"):
        continue

    # Extract date (e.g., "2022-06-04")
    date_str = fname.split("_pedestrian_shadow_information.csv")[0]

    combine_path = os.path.join(combine_folder, fname)
    ped_path = os.path.join(ped_folder, f"{date_str}_pedestrian_trajectory.csv")

    # ===== Read combine_data file =====
    df = pd.read_csv(combine_path)

    frame_stats = (
        df.groupby('frame')['in_shadow']
        .agg(total_peds='count', shadow_peds='sum')
        .reset_index()
    )
    frame_stats['percent'] = frame_stats['shadow_peds'] / frame_stats['total_peds']
    frame_stats['shadow_fraction'] = df.groupby('frame')['shadow_proportion'].mean().values
    frame_stats['shadow_bias'] = frame_stats['percent'] - frame_stats['shadow_fraction']

    spatial_percent = frame_stats['percent'].mean() * 100
    spatial_percent_bias = frame_stats['shadow_bias'].mean() * 100

    # ===== Read trajectory file, compute shadow time ratio and bias =====
    ped_df = pd.read_csv(ped_path)
    ratios = []
    biases = []
    shadow_fraction_per_frame = df.groupby('frame')['shadow_proportion'].mean()

    for _, row in ped_df.iterrows():
        # Shadow time ratio of the trajectory
        segments = ast.literal_eval(row['shadow_segments'])
        total_time = sum(t for _, t in segments)
        shadow_time = sum(t for s, t in segments if s)
        ratio = shadow_time / total_time
        ratios.append(ratio)

        # Frame range of the trajectory
        traj_frames = [p[0] for p in ast.literal_eval(row['trajectory'])]
        start_frame, end_frame = int(min(traj_frames)), int(max(traj_frames))

        # Average shadow_proportion over the corresponding frame range
        avg_shadow_fraction = shadow_fraction_per_frame.loc[
            (shadow_fraction_per_frame.index >= start_frame) &
            (shadow_fraction_per_frame.index <= end_frame)
        ].mean()

        # Trajectory-level bias
        biases.append(ratio - avg_shadow_fraction)

    temporal_percent = np.mean(ratios) * 100
    temporal_percent_bias = np.mean(biases) * 100

    # ===== Save daily results =====
    daily_results.append({
        'file': date_str,
        'spatial_percent_bias': spatial_percent_bias,
        'temporal_percent_bias': temporal_percent_bias
    })

# ====================== Merge weather data ======================
weather_df = pd.read_csv(weather_file).iloc[:, :5]
shadow_df = pd.DataFrame(daily_results)
merged_df = pd.concat([weather_df, shadow_df.iloc[:, 1:]], axis=1)

# ====================== Save results ======================
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig', float_format="%.4f")

print("\n✅ Saved to:", output_file)