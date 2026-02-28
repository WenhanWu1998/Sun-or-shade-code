import os
import pandas as pd
import numpy as np
import ast

folder = "./pedestrian_characteristics"
MIN_COUNT = 6


def process_shadow_segments_chaser(segments):
    """
    Only for Sun-chaser / Shade-chaser
    Rules:
    - Find the "main segments where state changes" (count >= MIN_COUNT and state differs from previous main segment)
    - All short segments (< MIN_COUNT) before this main segment → assign previous main segment state
    - All short segments (< MIN_COUNT) after this main segment → assign current main segment state
    - Merge consecutive segments with same state, keep only two segments
    - Output each segment as tuple (state, count)
    """
    # Find all main segment indices
    main_indices = [i for i, (_, c) in enumerate(segments) if c >= MIN_COUNT]

    if len(main_indices) < 2:
        return segments  # abnormal case, return directly

    # Find first main segment where state changes
    prev_state = segments[main_indices[0]][0]
    change_idx = None
    for idx in main_indices[1:]:
        state, _ = segments[idx]
        if state != prev_state:
            change_idx = idx
            break
    if change_idx is None:
        return segments  # no state-changing main segment, return

    change_state, _ = segments[change_idx]

    reassigned = []
    for i, (state, count) in enumerate(segments):
        if count < MIN_COUNT:
            if i < change_idx:
                reassigned.append((prev_state, count))   # short segment before → previous main segment state
            else:
                reassigned.append((change_state, count)) # short segment after → current main segment state
        else:
            reassigned.append((state, count))

    # Merge consecutive segments with same state
    merged = []
    for state, count in reassigned:
        if not merged:
            merged.append([state, count])
        else:
            if merged[-1][0] == state:
                merged[-1][1] += count
            else:
                merged.append([state, count])

    # Keep only two segments
    if len(merged) > 2:
        merged = [merged[0], merged[-1]]

    # ✅ Convert list to tuples
    merged = [tuple(seg) for seg in merged]

    return merged


# ======================
# Batch process files
# ======================
for fname in os.listdir(folder):
    if not fname.endswith(".csv"):
        continue

    path = os.path.join(folder, fname)
    df = pd.read_csv(path)

    new_shadow_segments = []

    for _, row in df.iterrows():

        shadow_segments = row.get("shadow_segments", np.nan)
        transition_2 = row.get("transition_2", np.nan)
        category = row.get("category", "")

        # ---------- Case 1: transition_2 missing ----------
        if pd.isna(transition_2):
            new_shadow_segments.append(shadow_segments)
            continue

        # ---------- Case 2: Others ----------
        if category == "Others":
            new_shadow_segments.append(np.nan)  # Excel output empty
            continue

        try:
            segments = ast.literal_eval(shadow_segments)

            # ---------- Heliophile ----------
            if category == "Heliophile":
                total = sum(c for _, c in segments)
                new_shadow_segments.append([(False, total)])

            # ---------- Photophobic ----------
            elif category == "Photophobic":
                total = sum(c for _, c in segments)
                new_shadow_segments.append([(True, total)])

            # ---------- Sun-chaser / Shade-chaser ----------
            elif category in ["Sun-chaser", "Shade-chaser"]:
                processed = process_shadow_segments_chaser(segments)
                new_shadow_segments.append(processed)

            else:
                new_shadow_segments.append(shadow_segments)

        except Exception:
            new_shadow_segments.append(shadow_segments)

    df["new_shadow_segments"] = new_shadow_segments
    df.to_csv(path, index=False)
    print(f"✅ Processed: {fname}")

print("\n All files updated with new_shadow_segments")