import os
import pandas as pd
import ast
import numpy as np

folder = "./pedestrian_characteristics"
FPS = 6
FRAME_PER_SAMPLE = 2
DT = FRAME_PER_SAMPLE / FPS  # Time interval per sample (seconds)
THRESHOLD = 0.3  # m/s

# ======================
# Loop through all files
# ======================
for fname in os.listdir(folder):
    if not fname.endswith("_pedestrian_trajectory.csv"):
        continue

    path = os.path.join(folder, fname)
    df = pd.read_csv(path)

    lingering_total_list = []
    lingering_sun_list = []
    lingering_shade_list = []

    for _, row in df.iterrows():

        # ======================
        # speed_total_seg
        # ======================
        if pd.isna(row["speed_total_seg"]):
            lingering_total_list.append(np.nan)
        else:
            try:
                speeds_total = ast.literal_eval(row["speed_total_seg"])
                count_total = sum(1 for v in speeds_total if v < THRESHOLD)
                if count_total == 0:
                    lingering_total_list.append(np.nan)
                else:
                    lingering_total_list.append(count_total * DT)
            except:
                lingering_total_list.append(np.nan)

        # ======================
        # speed_sun_seg
        # ======================
        if pd.isna(row["speed_sun_seg"]):
            lingering_sun_list.append(np.nan)
        else:
            try:
                speeds_sun = ast.literal_eval(row["speed_sun_seg"])
                count_sun = sum(1 for v in speeds_sun if v < THRESHOLD)
                if count_sun == 0:
                    lingering_sun_list.append(np.nan)
                else:
                    lingering_sun_list.append(count_sun * DT)
            except:
                lingering_sun_list.append(np.nan)

        # ======================
        # speed_shade_seg
        # ======================
        if pd.isna(row["speed_shade_seg"]):
            lingering_shade_list.append(np.nan)
        else:
            try:
                speeds_shade = ast.literal_eval(row["speed_shade_seg"])
                count_shade = sum(1 for v in speeds_shade if v < THRESHOLD)
                if count_shade == 0:
                    lingering_shade_list.append(np.nan)
                else:
                    lingering_shade_list.append(count_shade * DT)
            except:
                lingering_shade_list.append(np.nan)

    # ======================
    # Write to DataFrame
    # ======================
    df["lingering_total"] = lingering_total_list
    df["lingering_sun"] = lingering_sun_list
    df["lingering_shade"] = lingering_shade_list

    df.to_csv(path, index=False)
    print(f" Processed: {fname}")

print("\n All files processed successfully!")