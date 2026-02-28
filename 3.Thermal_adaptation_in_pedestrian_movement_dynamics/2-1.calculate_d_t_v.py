import os
import pandas as pd
import numpy as np
import ast

folder = "./pedestrian_characteristics"
FPS = 6
FRAME_PER_SAMPLE = 2
DT = FRAME_PER_SAMPLE / FPS  # Time interval per sample (seconds)

def compute_distance(traj_list):
    total = 0.0
    for i in range(len(traj_list) - 1):
        x1, y1 = traj_list[i][1], traj_list[i][2]
        x2, y2 = traj_list[i + 1][1], traj_list[i + 1][2]
        total += float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    return total

# ======================
# Loop through all files
# ======================
for fname in os.listdir(folder):
    if not fname.endswith("_pedestrian_trajectory.csv"):
        continue

    path = os.path.join(folder, fname)
    df = pd.read_csv(path)

    # ---------- total ----------
    dist_total_seg_list = []
    dist_total_list = []
    dur_total_list = []
    speed_total_seg_list = []
    speed_total_list = []

    # ---------- sun ----------
    dist_sun_seg_list = []
    dist_sun_list = []
    dur_sun_list = []
    speed_sun_seg_list = []
    speed_sun_list = []

    # ---------- shade ----------
    dist_shade_seg_list = []
    dist_shade_list = []
    dur_shade_list = []
    speed_shade_seg_list = []
    speed_shade_list = []

    for _, row in df.iterrows():
        category = row["category"]

        # ======================
        # Others
        # ======================
        if category == "Others":
            for L in [
                dist_total_seg_list, dist_total_list, dur_total_list,
                speed_total_seg_list, speed_total_list,
                dist_sun_seg_list, dist_sun_list, dur_sun_list,
                speed_sun_seg_list, speed_sun_list,
                dist_shade_seg_list, dist_shade_list, dur_shade_list,
                speed_shade_seg_list, speed_shade_list
            ]:
                L.append(np.nan)
            continue

        traj = ast.literal_eval(row["trajectory"])

        # ======================
        # total segment
        # ======================
        dist_total_seg = []
        speed_total_seg = []

        for i in range(len(traj) - 1):
            x1, y1 = traj[i][1], traj[i][2]
            x2, y2 = traj[i + 1][1], traj[i + 1][2]
            d = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            dist_total_seg.append(d)
            speed_total_seg.append(float(d / DT))

        dist_total_seg_list.append(dist_total_seg if dist_total_seg else np.nan)
        speed_total_seg_list.append(speed_total_seg if speed_total_seg else np.nan)

        # total scalar
        dist_total = compute_distance(traj)
        duration_total = row["duration_frames"] / FPS
        speed_total = dist_total / duration_total if duration_total > 0 else np.nan

        dist_total_list.append(dist_total)
        dur_total_list.append(duration_total)
        speed_total_list.append(speed_total)

        # ======================
        # sun / shade
        # ======================
        # parse new_shadow_segments
        try:
            new_shadow_segments = ast.literal_eval(row["new_shadow_segments"])
        except:
            new_shadow_segments = []

        # expand to boolean list of same length as traj
        new_shadow_flags = []
        for state, length in new_shadow_segments:
            new_shadow_flags.extend([state] * length)
        if len(new_shadow_flags) < len(traj):
            new_shadow_flags.extend([False] * (len(traj) - len(new_shadow_flags)))

        if category == "Heliophile":
            dist_sun_seg_list.append(dist_total_seg)
            dist_sun_list.append(dist_total)
            dur_sun_list.append(duration_total)
            speed_sun_seg_list.append(speed_total_seg)
            speed_sun_list.append(speed_total)

            for L in [
                dist_shade_seg_list, dist_shade_list, dur_shade_list,
                speed_shade_seg_list, speed_shade_list
            ]:
                L.append(np.nan)

        elif category == "Photophobic":
            for L in [
                dist_sun_seg_list, dist_sun_list, dur_sun_list,
                speed_sun_seg_list, speed_sun_list
            ]:
                L.append(np.nan)

            dist_shade_seg_list.append(dist_total_seg)
            dist_shade_list.append(dist_total)
            dur_shade_list.append(duration_total)
            speed_shade_seg_list.append(speed_total_seg)
            speed_shade_list.append(speed_total)

        elif category in ["Sun-chaser", "Shade-chaser"]:
            dist_sun = dist_shade = 0.0
            sun_intervals = shade_intervals = 0
            dist_sun_seg, speed_sun_seg = [], []
            dist_shade_seg, speed_shade_seg = [], []

            for i in range(len(traj) - 1):
                x1, y1 = traj[i][1], traj[i][2]
                x2, y2 = traj[i + 1][1], traj[i + 1][2]
                d = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

                if new_shadow_flags[i + 1]:  # <-- use new_shadow_segments
                    dist_shade += d
                    shade_intervals += 1
                    dist_shade_seg.append(d)
                    speed_shade_seg.append(float(d / DT))
                else:
                    dist_sun += d
                    sun_intervals += 1
                    dist_sun_seg.append(d)
                    speed_sun_seg.append(float(d / DT))

            # sun
            dist_sun_seg_list.append(dist_sun_seg if sun_intervals else np.nan)
            dist_sun_list.append(dist_sun if sun_intervals else np.nan)
            dur_sun_list.append(sun_intervals * DT if sun_intervals else np.nan)
            speed_sun_seg_list.append(speed_sun_seg if sun_intervals else np.nan)
            speed_sun_list.append(dist_sun / (sun_intervals * DT) if sun_intervals else np.nan)

            # shade
            dist_shade_seg_list.append(dist_shade_seg if shade_intervals else np.nan)
            dist_shade_list.append(dist_shade if shade_intervals else np.nan)
            dur_shade_list.append(shade_intervals * DT if shade_intervals else np.nan)
            speed_shade_seg_list.append(speed_shade_seg if shade_intervals else np.nan)
            speed_shade_list.append(dist_shade / (shade_intervals * DT) if shade_intervals else np.nan)

    # ======================
    # Write to DataFrame (strict order)
    # ======================
    df["distance_total_seg"] = dist_total_seg_list
    df["distance_total"] = dist_total_list
    df["duration_total"] = dur_total_list
    df["speed_total_seg"] = speed_total_seg_list
    df["speed_total"] = speed_total_list

    df["distance_sun_seg"] = dist_sun_seg_list
    df["distance_sun"] = dist_sun_list
    df["duration_sun"] = dur_sun_list
    df["speed_sun_seg"] = speed_sun_seg_list
    df["speed_sun"] = speed_sun_list

    df["distance_shade_seg"] = dist_shade_seg_list
    df["distance_shade"] = dist_shade_list
    df["duration_shade"] = dur_shade_list
    df["speed_shade_seg"] = speed_shade_seg_list
    df["speed_shade"] = speed_shade_list

    df.to_csv(path, index=False)
    print(f" Processed: {fname}")

print("\n All files processed successfully!")