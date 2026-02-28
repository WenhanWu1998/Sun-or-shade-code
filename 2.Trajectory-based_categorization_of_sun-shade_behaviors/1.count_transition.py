import os
import pandas as pd
import matplotlib.pyplot as plt

########################################### CONFIG ###########################################
trajectory_folder = "../data/2.pedestrian_trajectory"
figure_folder = "./figure"   
os.makedirs(figure_folder, exist_ok=True)

transition_counts = []  # Store transition counts for all individuals

########################################### PROCESS ###########################################
for filename in os.listdir(trajectory_folder):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(trajectory_folder, filename)
    df = pd.read_csv(file_path)

    for segments_str in df['shadow_segments']:
        try:
            segments = eval(segments_str)
            n_transitions = max(len(segments) - 1, 0)
            transition_counts.append(n_transitions)
        except Exception:
            continue  # Skip rows with format errors

########################################### PLOT ###########################################
plt.figure(figsize=(9, 6))

# Count frequency of each transition number
counts = pd.Series(transition_counts).value_counts().sort_index()

plt.bar(counts.index, counts.values, color='lightseagreen', edgecolor='black')
plt.xlabel("Number of Transitions", fontsize=14)
plt.ylabel("Number of Trajectories", fontsize=14)
plt.title("Distribution of Transition Counts", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(counts.index, fontsize=12)
plt.yticks(fontsize=12)

# Display counts above each bar
for x, y in zip(counts.index, counts.values):
    plt.text(x, y + 5, str(y), ha='center', va='bottom', fontsize=11)

plt.tight_layout()

# Save figure to ./figure folder
fig_path = os.path.join(figure_folder, "Count_transition_distribution.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {fig_path}")

plt.show()