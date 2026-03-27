
import argparse

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, write_info, write_stats
from lerobot.datasets.v21.convert_dataset_v20_to_v21 import V20, V21
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import json


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    
    print(parquet_paths)
    print(all_low_dim_data_list)

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        print(f"Computing statistics for {le_modality}...")
        # check if the data is the modality is actually a list of numbers
        # skip if it is a string
        if isinstance(all_low_dim_data[le_modality].iloc[0], str):
            print(f"Skipping {le_modality} because it is a string")
            continue

        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics



def convert_dataset(
    repo_id: str,
    root: str | None = None,
    push_to_hub: bool = False,
    delete_old_stats: bool = False,
    branch: str | None = None,
):
    if root is not None:
        dataset = LeRobotDataset(repo_id, root, revision=V21)
    else:
        dataset = LeRobotDataset(repo_id, revision=V21, force_cache_sync=True)

    if (dataset.root / STATS_PATH).is_file():
        (dataset.root / STATS_PATH).unlink()

    parquet_files = list((Path(root)).glob("data/*/*.parquet"))

    print(f"root: {root}")
    print(f"parquet_files: {parquet_files}")

    le_statistics = calculate_dataset_statistics(parquet_files)
    stats_path = os.path.join(root, "meta/stats.json")
    with open(stats_path, "w") as f:
        json.dump(le_statistics, f, indent=4)

    dataset.meta.info["codebase_version"] = V20
    write_info(dataset.meta.info, dataset.root)

    if push_to_hub:
        dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")

    # delete old stats.json file
    if delete_old_stats and (dataset.root / EPISODES_STATS_PATH).is_file:
        (dataset.root / EPISODES_STATS_PATH).unlink()

    hub_api = HfApi()
    if delete_old_stats and hub_api.file_exists(
        repo_id=dataset.repo_id, filename=EPISODES_STATS_PATH, revision=branch, repo_type="dataset"
    ):
        hub_api.delete_file(
            path_in_repo=EPISODES_STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset"
        )
    if push_to_hub:
        hub_api.create_tag(repo_id, tag=V20, revision=branch, repo_type="dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to the local dataset root directory. If not provided, the script will use the dataset from local.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to the hub after conversion. Defaults to False.",
    )
    parser.add_argument(
        "--delete-old-stats",
        action="store_true",
        help="Delete the old stats.json file after conversion. Defaults to False.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))


"""
Example:
    python preprocessing/convert_dataset_v21_to_v20_gr00t.py \
        --repo-id=part2/basic_pick_place \
        --root=/path/to/egodex_lerobot_gr00t/part2/basic_pick_place/
"""