import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--lerobot_base_path", type=str, required=True)
parser.add_argument("--replay_base_path", type=str, required=True)
parser.add_argument("--output_base_path", type=str, required=True)

args = parser.parse_args()


# --- 1. Define file paths ---
# LeRobot dataset source paths
lerobot_base_path = Path(args.lerobot_base_path)
lerobot_data_path = lerobot_base_path / "data" / "chunk-000"
lerobot_meta_path = lerobot_base_path / "meta"
# (New) LeRobot video path
lerobot_video_path = lerobot_base_path / "videos" / "chunk-000" / "observation.images.ego_view"


# Replay dataset source path
replay_base_path = Path(args.replay_base_path)

# Define output paths for the new dataset
output_base_path = Path(args.output_base_path)
output_data_path = output_base_path / "data" / "chunk-000"
output_meta_path = output_base_path / "meta"
# (New) Video output path for the new dataset
output_video_path = output_base_path / "videos" / "chunk-000" / "observation.images.ego_view"


# --- 2. Create all output directories ---
print("Creating output directories...")
output_data_path.mkdir(parents=True, exist_ok=True)
output_meta_path.mkdir(parents=True, exist_ok=True)
output_video_path.mkdir(parents=True, exist_ok=True)


# --- 3. Preparation ---
# Build mapping from episode_index to trajectory_id
episode_to_trajectory = {}
with open(lerobot_meta_path / "episodes.jsonl", "r") as f:
    for line in f:
        meta_data = json.loads(line)
        episode_to_trajectory[meta_data["episode_index"]] = meta_data["trajectory_id"]

# Define columns to concatenate from the Replay dataset
replay_cols = [
    'wrist_l_pos', 'wrist_l_rot6d', 'thumb_l_pos', 'thumb_l_rot6d', 
    'index_l_pos', 'index_l_rot6d', 'middle_l_pos', 'middle_l_rot6d', 
    'ring_l_pos', 'ring_l_rot6d', 'pinky_l_pos', 'pinky_l_rot6d', 
    'wrist_r_pos', 'wrist_r_rot6d', 'thumb_r_pos', 'thumb_r_rot6d', 
    'index_r_pos', 'index_r_rot6d', 'middle_r_pos', 'middle_r_rot6d', 
    'ring_r_pos', 'ring_r_rot6d', 'pinky_r_pos', 'pinky_r_rot6d', 
    'camera_egoview_pos', 'camera_egoview_rot6d'
]

# Lists for collecting newly added data to compute global statistics
all_added_states_data = []
all_added_actions_data = []

# --- 4. Iterate, augment data and copy videos ---
print("Starting data augmentation and video copying...")
# Use sorted() to ensure consistent processing order
for lerobot_file in tqdm(sorted(lerobot_data_path.glob("episode_*.parquet"))):
    episode_index = int(lerobot_file.stem.split("_")[-1])
    
    if episode_index not in episode_to_trajectory:
        continue
        
    trajectory_id = episode_to_trajectory[episode_index]
    demo_id = int(trajectory_id.split('-')[-1])
    replay_file = replay_base_path / f"demo_{demo_id}.parquet"

    if not replay_file.exists():
        print(f"Warning: Cannot find corresponding Replay file {replay_file.name}, skipping episode {episode_index}")
        continue

    df_lerobot = pd.read_parquet(lerobot_file)
    df_replay = pd.read_parquet(replay_file)
    
    # --- Data alignment ---
    assert len(df_lerobot) == len(df_replay)
    min_len = min(len(df_lerobot), len(df_replay))
    if min_len == 0:
        continue
    df_lerobot = df_lerobot.iloc[:min_len].copy()
    df_replay_subset = df_replay[replay_cols].iloc[:min_len]

    # --- State and Action augmentation ---
    new_states = []
    new_actions = []
    added_states_for_stats = []
    added_actions_for_stats = []

    for i in range(min_len):
        replay_state_flat = np.concatenate([np.array(x).flatten() for x in df_replay_subset.iloc[i].values])
        
        if i + 1 < min_len:
            replay_action_flat = np.concatenate([np.array(x).flatten() for x in df_replay_subset.iloc[i+1].values])
        else: # Padding
            replay_action_flat = np.concatenate([np.array(x).flatten() for x in df_replay_subset.iloc[-1].values])
            
        new_states.append(np.concatenate([df_lerobot['observation.state'].iloc[i], replay_state_flat]))
        new_actions.append(np.concatenate([df_lerobot['action'].iloc[i], replay_action_flat]))
        
        added_states_for_stats.append(replay_state_flat)
        added_actions_for_stats.append(replay_action_flat)

    df_lerobot['observation.state'] = new_states
    df_lerobot['action'] = new_actions
    
    all_added_states_data.extend(added_states_for_stats)
    all_added_actions_data.extend(added_actions_for_stats)

    # --- Save Parquet results ---
    df_lerobot.to_parquet(output_data_path / lerobot_file.name)

    # --- (New) Copy the corresponding video file ---
    video_filename = lerobot_file.with_suffix(".mp4").name
    source_video_file = lerobot_video_path / video_filename
    
    if source_video_file.exists():
        shutil.copy(source_video_file, output_video_path / video_filename)
    else:
        print(f"Warning: Cannot find corresponding video file {source_video_file.name}, skipping copy")

print(f"Data augmentation and video copying complete!")
print(f"  - Parquet files saved to: {output_data_path}")
print(f"  - Video files saved to: {output_video_path}")


# --- 5. Metadata update ---
print("\nStarting metadata file updates...")

# Convert collected data to Numpy arrays for efficient computation
states_np = np.array(all_added_states_data)
actions_np = np.array(all_added_actions_data)

# Get a sample Replay DataFrame to determine the dimension of each new field
sample_df_replay = pd.read_parquet(next(replay_base_path.glob("*.parquet")))
col_dims = {col: len(np.array(sample_df_replay[col].iloc[0]).flatten()) for col in replay_cols}
added_dim = sum(col_dims.values()) # Total newly added dimensions

# --- 5.1 Update info.json ---
info_file = lerobot_meta_path / "info.json"
with open(info_file, 'r') as f:
    info_data = json.load(f)

info_data['features']['observation.state']['shape'][0] += added_dim
info_data['features']['action']['shape'][0] += added_dim

with open(output_meta_path / "info.json", 'w') as f:
    json.dump(info_data, f, indent=4)
print(f"Updated and saved: {output_meta_path / 'info.json'}")


# --- 5.2 Update modality.json ---
modality_file = lerobot_meta_path / "modality.json"
with open(modality_file, 'r') as f:
    modality_data = json.load(f)

last_state_end = max(v['end'] for v in modality_data['state'].values())
last_action_end = max(v['end'] for v in modality_data['action'].values())

for col_name, dim in col_dims.items():
    modality_data['state'][col_name] = {'original_key': 'observation.state', 'start': last_state_end, 'end': last_state_end + dim}
    last_state_end += dim
    modality_data['action'][col_name] = {'original_key': 'action', 'start': last_action_end, 'end': last_action_end + dim}
    last_action_end += dim

with open(output_meta_path / "modality.json", 'w') as f:
    json.dump(modality_data, f, indent=4)
print(f"Updated and saved: {output_meta_path / 'modality.json'}")


# --- 5.3 Update stats.json and metadata.json ---
stats_file = lerobot_meta_path / "stats.json"
metadata_file = lerobot_meta_path / "metadata.json"
with open(stats_file, 'r') as f:
    stats_data = json.load(f)
with open(metadata_file, 'r') as f:
    metadata_data = json.load(f)

stats_to_compute = {
    "mean": np.mean, "std": np.std, "min": np.min, "max": np.max,
    "q01": lambda x, axis: np.quantile(x, 0.01, axis=axis),
    "q99": lambda x, axis: np.quantile(x, 0.99, axis=axis)
}

data_map = {'observation.state': states_np, 'action': actions_np}
metadata_map = {'observation.state': 'state', 'action': 'action'}

current_col_idx = 0
for col_name, dim in col_dims.items():
    col_slice = slice(current_col_idx, current_col_idx + dim)
    
    for key, data_array in data_map.items():
        meta_key = metadata_map[key]
        col_stats = {}
        for stat_name, func in stats_to_compute.items():
            col_stats[stat_name] = func(data_array[:, col_slice], axis=0).tolist()
            
        metadata_data['dataset_statistics'][meta_key][col_name] = col_stats
        
        metadata_data['modalities'][meta_key][col_name] = {
            "absolute": True, "rotation_type": "rotation_6d" if "rot6d" in col_name else None,
            "shape": [dim], "continuous": True
        }
    current_col_idx += dim
    
for key, data_array in data_map.items():
    for stat_name, func in stats_to_compute.items():
        if stat_name in stats_data[key]:
            full_stats = func(data_array, axis=0).tolist()
            stats_data[key][stat_name].extend(full_stats)

with open(output_meta_path / "stats.json", 'w') as f:
    json.dump(stats_data, f, indent=4)
print(f"Updated and saved: {output_meta_path / 'stats.json'}")

with open(output_meta_path / "metadata.json", 'w') as f:
    json.dump(metadata_data, f, indent=4)
print(f"Updated and saved: {output_meta_path / 'metadata.json'}")


# --- 6. Copy remaining metadata files ---
print("\nCopying remaining metadata files...")
handled_meta_files = ["info.json", "modality.json", "metadata.json", "stats.json"]
for meta_file in lerobot_meta_path.iterdir():
    if meta_file.name not in handled_meta_files:
        shutil.copy(meta_file, output_meta_path / meta_file.name)
print("\n--------------------")
print("All tasks complete! The new dataset has been generated at:")
print(output_base_path.absolute())
print("--------------------")