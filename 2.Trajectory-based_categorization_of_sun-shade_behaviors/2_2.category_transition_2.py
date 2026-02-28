# %% Full version: classification and mapping for transition_2 (ignore ≤3-frame segments, merge adjacent same states, only process trajectories with transition>=2)
import os
import pandas as pd
import ast

########################################### CONFIG ###########################################
trajectory_folder = "./pedestrian_category"  # Folder where the original complete trajectories already exist
ped_folder = "./pedestrian_shadow_data/pedestrian"  # Frame-level trajectory file (one row per frame)
MIN_DURATION = 6  # Ignore short states with duration < 6 frames (2s)

# Classification counters
count_heliophile = 0
count_photophobic = 0
count_sun_chaser = 0
count_shade_chaser = 0
count_others = 0
total_transition2 = 0  # Total number of trajectories with transition>=2

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
            continue

        # Only process trajectories with transition >= 2 (original number of segments >= 3)
        if len(segments) < 3:
            final_categories.append("")
            continue

        # Accumulate total number of transition>=2 trajectories
        total_transition2 += 1

        # Ignore short-duration state segments
        filtered_segments = [seg for seg in segments if (len(seg) > 1 and seg[1] >= MIN_DURATION)]

        # Merge adjacent segments with the same state (tuple → list fix)
        merged_segments = []
        for seg in filtered_segments:
            seg = list(seg)  # Convert to list for modification
            if not merged_segments:
                merged_segments.append(seg)
            else:
                if seg[0] == merged_segments[-1][0]:
                    # Same state → merge durations
                    merged_segments[-1][1] += seg[1]
                else:
                    merged_segments.append(seg)

        # Classification based on merged segment count
        if len(merged_segments) == 1:
            # Single-segment trajectory → heliophile / photophobic
            state = merged_segments[0][0]
            if state:  # Shadow
                cat = "Photophobic"
                count_photophobic += 1
            else:      # Sunlight
                cat = "Heliophile"
                count_heliophile += 1
        elif len(merged_segments) == 2:
            # Two-segment trajectory → sun-chaser / shade-chaser
            first_state = merged_segments[0][0]
            second_state = merged_segments[1][0]

            if first_state and not second_state:
                cat = "Sun-chaser"  # Shadow → Sunlight
                count_sun_chaser += 1
            elif not first_state and second_state:
                cat = "Shade-chaser"  # Sunlight → Shadow
                count_shade_chaser += 1
            else:
                cat = ""
        else:
            # Remaining cases with 3 or more segments → classified as Others
            cat = "Others"
            count_others += 1

        final_categories.append(cat)

    # Update transition_2 column (overwrite original file)
    df['transition_2'] = final_categories
    df.to_csv(file_path, index=False)

    ########################################### MAP TO FRAME-LEVEL FILE ###########################################
    date_str = filename.split("_")[0]
    ped_file_path = os.path.join(ped_folder, f"{date_str}_pedestrian_information_filter_rotation_cut.csv")
    if not os.path.exists(ped_file_path):
        print(f"No frame-level file for {date_str}, skip mapping.")
        continue

    ped_data = pd.read_csv(ped_file_path)

    # Add a new column transition_2
    ped_data['transition_2'] = ""

    # Iterate over each complete trajectory and assign transition_2 to corresponding frames
    for idx, row in df.iterrows():
        pid = row['person_id']
        cat = row.get('transition_2', "")

        traj_list = ast.literal_eval(row['trajectory'])
        frames = [t for t, _, _, _ in traj_list]

        mask = (ped_data['person_id'] == pid) & (ped_data['frame'].isin(frames))
        ped_data.loc[mask, 'transition_2'] = cat

    # Overwrite original CSV file
    ped_data.to_csv(ped_file_path, index=False)
    print(f"Updated {ped_file_path}")

########################################### PRINT STATISTICS ###########################################
print("===== Classification for transition_2 (original transition>=2) after ignoring short segments and merging =====")
print(f"Total trajectories with transition >=2: {total_transition2}")
print(f"Heliophile: {count_heliophile} ({count_heliophile/total_transition2*100:.2f}%)")
print(f"Photophobic: {count_photophobic} ({count_photophobic/total_transition2*100:.2f}%)")
print(f"Sun-chaser: {count_sun_chaser} ({count_sun_chaser/total_transition2*100:.2f}%)")
print(f"Shade-chaser: {count_shade_chaser} ({count_shade_chaser/total_transition2*100:.2f}%)")
print(f"Others: {count_others} ({count_others/total_transition2*100:.2f}%)")