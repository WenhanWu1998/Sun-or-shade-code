# %% Aggregate transition_0, transition_1, transition_2 into category, and print overall statistics
import os
import pandas as pd
import ast
from collections import Counter

########################################### CONFIG ###########################################
trajectory_folder = "./pedestrian_category"  # Folder containing the original complete trajectories
ped_folder = "./pedestrian_shadow_data/pedestrian"  # Frame-level trajectory file (one row per frame)

# Global counter
total_counts = Counter()

########################################### PROCESS TRAJECTORY FILES ###########################################
for filename in sorted(os.listdir(trajectory_folder)):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(trajectory_folder, filename)
    df = pd.read_csv(file_path)

    # Aggregate transition_0,1,2 into category (take the first non-empty value)
    def merge_transitions(row):
        for col in ['transition_0', 'transition_1', 'transition_2']:
            if pd.notna(row.get(col, None)) and row.get(col) != "":
                return row[col]
        return "Others"  # If all are empty, label as Others

    df['category'] = df.apply(merge_transitions, axis=1)

    # Update global counts
    total_counts.update(df['category'])

    # Save back to the complete trajectory file (overwrite original file)
    df.to_csv(file_path, index=False)

    ########################################### MAP TO FRAME-LEVEL FILE ###########################################
    date_str = filename.split("_")[0]
    ped_file_path = os.path.join(ped_folder, f"{date_str}_pedestrian_information_filter_rotation_cut.csv")
    if not os.path.exists(ped_file_path):
        print(f"No frame-level file for {date_str}, skip mapping.")
        continue

    ped_data = pd.read_csv(ped_file_path)
    ped_data['category'] = ""  # Add new category column

    # Iterate over each complete trajectory and assign category to corresponding frames
    for idx, row in df.iterrows():
        pid = row['person_id']
        cat = row.get('category', "Others")

        traj_list = ast.literal_eval(row['trajectory'])
        frames = [t for t, _, _, _ in traj_list]

        mask = (ped_data['person_id'] == pid) & (ped_data['frame'].isin(frames))
        ped_data.loc[mask, 'category'] = cat

    # Overwrite original CSV file
    ped_data.to_csv(ped_file_path, index=False)
    print(f"Updated {ped_file_path}")

########################################### PRINT OVERALL STATISTICS ###########################################
print("\n===== Overall category statistics =====")
total = sum(total_counts.values())
for cat in ["Heliophile", "Photophobic", "Sun-chaser", "Shade-chaser", "Others"]:
    cnt = total_counts.get(cat, 0)
    print(f"{cat}: {cnt} ({cnt/total*100:.2f}%)")