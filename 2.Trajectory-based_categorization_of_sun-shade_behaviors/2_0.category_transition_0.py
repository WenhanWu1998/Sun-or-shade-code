# %% Full version: directly generate transition_0 in ./pedestrian_category and map to frame-level trajectories
import os
import pandas as pd
import ast

########################################### CONFIG ###########################################
trajectory_folder = "./pedestrian_category"  # Folder where the original complete trajectories already exist
ped_folder = "./pedestrian_shadow_data/pedestrian"  # Frame-level trajectory file (one row per frame)

# Classification counters
count_heliophile = 0
count_photophobic = 0
count_others = 0

########################################### PROCESS TRAJECTORY FILES ###########################################
for filename in sorted(os.listdir(trajectory_folder)):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(trajectory_folder, filename)
    df = pd.read_csv(file_path)

    final_categories = []

    for segments_str in df['shadow_segments']:
        try:
            segments = eval(segments_str)
        except Exception:
            final_categories.append("")
            count_others += 1
            continue

        # Only process the case where transition = 0
        if len(segments) == 1:
            first_state = segments[0][0]  # True means shadow, False means sunlight
            if first_state:
                cat = "Photophobic"
                count_photophobic += 1
            else:
                cat = "Heliophile"
                count_heliophile += 1
        else:
            cat = ""
            count_others += 1

        final_categories.append(cat)

    # Update transition_0 column (overwrite original file)
    df['transition_0'] = final_categories
    df.to_csv(file_path, index=False)

    ########################################### MAP TO FRAME-LEVEL FILE ###########################################
    date_str = filename.split("_")[0]
    ped_file_path = os.path.join(ped_folder, f"{date_str}_pedestrian_information_filter_rotation_cut.csv")
    if not os.path.exists(ped_file_path):
        print(f"No frame-level file for {date_str}, skip mapping.")
        continue

    ped_data = pd.read_csv(ped_file_path)

    # Add a new column transition_0
    ped_data['transition_0'] = ""

    # Iterate over each complete trajectory and assign transition_0 to corresponding frames
    for idx, row in df.iterrows():
        pid = row['person_id']
        cat = row.get('transition_0', "")

        traj_list = ast.literal_eval(row['trajectory'])
        frames = [t for t, _, _, _ in traj_list]

        mask = (ped_data['person_id'] == pid) & (ped_data['frame'].isin(frames))
        ped_data.loc[mask, 'transition_0'] = cat

    # Overwrite original CSV file
    ped_data.to_csv(ped_file_path, index=False)
    print(f"Updated {ped_file_path}")

########################################### PRINT STATISTICS ###########################################
total = count_heliophile + count_photophobic + count_others
print("===== Classification for transition = 0 =====")
print(f"Heliophile: {count_heliophile} ({count_heliophile/total*100:.2f}%)")
print(f"Photophobic: {count_photophobic} ({count_photophobic/total*100:.2f}%)")
