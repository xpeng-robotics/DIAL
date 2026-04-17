# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from gr00t.utils.video import get_all_frames, get_frames_by_timestamps

from .embodiment_tags import EmbodimentTag
from .schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
    StateActionMetadata,
)
from .transform import ComposedModalityTransform
import random
import sys
import torch.distributed as dist

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_RELATIVE_STATS_FILENAME = "meta/relative_stats.json"


def load_metadata(model_path):
    """Load the transforms for the model."""
    # Load metadata for normalization stats
    model_path = Path(model_path)
    exp_cfg_dir = model_path / "experiment_cfg"
    metadata_path = exp_cfg_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadatas = json.load(f)

    for tag, metadata_dict in metadatas.items():
        metadatas[tag] = DatasetMetadata.model_validate(metadata_dict)

    return metadatas


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


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""

    random_indices_start: int = sys.maxsize

    vlm_gap: Optional[int] = None
            
    action_gap_range: Optional[list] = None


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        relative_stats_filename: str = LE_ROBOT_RELATIVE_STATS_FILENAME,  # New: support explicit selection of stats file
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        split: str = None,
        temporal_split: float = None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.relative_stats_filename = relative_stats_filename

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._trajectory_ids, self._trajectory_lengths, self._trajectory_tasks = self._get_trajectories(split=split, temporal_split=temporal_split)
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._max_delta_index = self._get_max_delta_index()

        # NOTE(YL): method to predict the task progress
        if "action.task_progress" in self._modality_keys["action"]:
            print("action.task_progress is in the action modality, task progress will be label")
            self._modality_keys["action"].append("action.task_progress")
            self._metadata.modalities.action["task_progress"] = StateActionMetadata(
                absolute=True, rotation_type=None, shape=(1,), continuous=True
            )
            # assume the task progress is uniformly distributed between 0 and 1
            self._metadata.statistics.action["task_progress"] = DatasetStatisticalValues(
                max=[1.0], min=[0.0], mean=[0.5], std=[0.2887], q01=[0.01], q99=[0.99]
            )

        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths
    
    @property
    def trajectory_tasks(self) -> list[str]:
        return self._trajectory_tasks

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    def _get_max_delta_index(self) -> int:
        """Calculate the maximum delta index across all modalities.
        Returns:
            int: The maximum delta index value.
        """
        max_delta_index = 0
        for delta_index in self.delta_indices.values():
            max_delta_index = max(max_delta_index, delta_index.max())
        return max_delta_index

    @property
    def max_delta_index(self) -> int:
        """The maximum delta index across all modalities."""
        return self._max_delta_index

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e} {stats_path}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            le_statistics = calculate_dataset_statistics(parquet_files)
            with open(stats_path, "w") as f:
                json.dump(le_statistics, f, indent=4)
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Dataset relative statistics
        relative_stats_path = self.dataset_path / self.relative_stats_filename
        if relative_stats_path.exists():
            print(f"Load dataset relative statistics from {relative_stats_path} ...")
            with open(relative_stats_path, "r") as f:
                relative_le_statistics = json.load(f)

            relative_dataset_statistics = {"state": {}, "action": {}}
            for our_modality in ["state", "action"]:
                for subkey in simplified_modality_meta[our_modality]:
                    if not (subkey.endswith("pos") or subkey.endswith("rot") or subkey.endswith("rot6d")):
                        continue

                    prefix, suffix = subkey.rsplit("_", 1)
                    if our_modality == "state":
                        candidates = [f"observation.state.relative.{prefix}_in_world_{suffix}", f"observation.state.relative.{prefix}_in_wrist_{suffix}"]
                    else:
                        candidates = [f"action.relative.{prefix}_in_world_{suffix}", f"action.relative.{prefix}_in_wrist_{suffix}"]
                        
                    for candidate in candidates:
                        if candidate in relative_le_statistics:
                            relative_dataset_statistics[our_modality][f"relative.{subkey}"] = relative_le_statistics[candidate]
            
            dataset_statistics["state"].update(relative_dataset_statistics["state"])
            dataset_statistics["action"].update(relative_dataset_statistics["action"])
        # print(f"dataset_statistics.keys: {dataset_statistics.keys()}")



        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self, split=None, temporal_split=None) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            if temporal_split is None:
                trajectory_lengths.append(episode["length"])
            else:
                trajectory_lengths.append(int(episode["length"] * temporal_split))
                # print(f"trajectory_length: {episode['length']}\t{int(episode['length'] * temporal_split)}")
        # print(f"trajectory_ids: {trajectory_ids}")
        # print(f"trajectory_lengths: {trajectory_lengths}")

        # NOTE: Only for PhysicalAI-Robotics-GR00T-Teleop-Sim Data
        if len(episode_metadata[0]["tasks"][0].split()) == 1:
            trajectory_tasks = []
            for episode in episode_metadata:
                trajectory_tasks.append(["unlocked_waist: " + episode["remarks"]])
            print(f"Using unified task instructions, e.g., {episode_metadata[0]['tasks']} --> {trajectory_tasks[0]}")
        else:
            trajectory_tasks = None

        if split is not None:
            trajectory_ids = eval(f"trajectory_ids{split}")
            trajectory_lengths = eval(f"trajectory_lengths{split}")
            if trajectory_tasks is not None:
                trajectory_tasks = eval(f"trajectory_tasks{split}")
            print(f"len({self.dataset_path}): {len(trajectory_ids)}")
            
        return np.array(trajectory_ids), np.array(trajectory_lengths), trajectory_tasks

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        all_steps: list[tuple[int, int]] = []
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))
        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        # print(f"self.lerobot_modality_meta: {self.lerobot_modality_meta}")

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                if key == "action.task_progress":
                    continue

                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def reset_metadata(self, model_path):
        metadatas = load_metadata(model_path)
        metadata = metadatas.get(self.tag)
        self.transforms.set_metadata(metadata)
        self._metadata = metadata

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        # print(f"self.all_steps: {self.all_steps}")
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        return self.transforms(self.get_step_data(trajectory_id, base_index))

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]

        # this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)

        if self.trajectory_tasks is not None:
            # print(self.trajectory_tasks[trajectory_index])
            return self.trajectory_tasks[trajectory_index]


        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
            if original_key not in self.curr_traj_data.keys():
                original_key = subkey
        for i in range(len(step_indices)):
            task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
        
        # print(self.tasks.loc[task_indices]["task"].tolist())
        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")



class LeRobotSingleDatasetWithGoalImage(LeRobotSingleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm_gap = self.modality_configs["action"].vlm_gap
        self.action_gap_range = self.modality_configs["action"].action_gap_range # e.g. [16, 48]
        if self.vlm_gap is not None:
            assert self.action_gap_range is not None

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        # If new parameters are not configured, fall back to the old logic
        if self.vlm_gap is None:
            data = {}
            # Get the data for all modalities
            self.curr_traj_data = self.get_trajectory_data(trajectory_id)
            for modality in self.modality_keys:
                # Get the data corresponding to each key in the modality
                for key in self.modality_keys[modality]:
                    data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
            return data
        
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        traj_idx = self.get_trajectory_index(trajectory_id)
        max_len = self.trajectory_lengths[traj_idx]

        # 1. Compute indices
        vlm_curr_idx = base_index
        goal_idx = min(vlm_curr_idx + self.vlm_gap, max_len - 1)
        
        # Action Head gap is randomly sampled within the range
        low, high = self.action_gap_range if self.action_gap_range else (self.vlm_gap, self.vlm_gap)
        actual_action_gap = random.randint(low, high)
        action_curr_idx = max(0, goal_idx - actual_action_gap)

        # 2. Get data for each modality
        for modality in self.modality_keys:
            for key in self.modality_keys[modality]:
                if modality == "video":
                    # Special handling for video: return 3 frames [VLM current, goal, Action current]
                    indices = np.array([vlm_curr_idx, goal_idx, action_curr_idx])
                    # print(f"indices: {indices}")
                    data[key] = self.get_video_by_indices(trajectory_id, key, indices)
                elif modality in ["state", "action"]:
                    # Action and State use action_curr_idx as the base index
                    data[key] = self.get_state_or_action(trajectory_id, modality, key, action_curr_idx)
                else:
                    data[key] = self.get_data_by_modality(trajectory_id, modality, key, vlm_curr_idx)
        return data

    def get_video_by_indices(self, trajectory_id: int, key: str, indices: np.ndarray) -> np.ndarray:
        # Helper function: read video frames directly by indices
        indices = np.clip(indices, 0, self.trajectory_lengths[self.get_trajectory_index(trajectory_id)] - 1)
        key_clean = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key_clean)
        timestamp = self.curr_traj_data["timestamp"].to_numpy()
        video_timestamp = timestamp[indices]
        return get_frames_by_timestamps(video_path.as_posix(), video_timestamp, 
                                      video_backend=self.video_backend, **self.video_backend_kwargs)

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices

        # print(self.modality_configs['action'])
        # print(type(self.delta_indices[key]), self.delta_indices[key])
        action_horizon = len(self.modality_configs['action'].delta_indices)
        step_indices = self.delta_indices[key] + base_index

        if self.modality_configs[key.split('.')[0]].random_indices_start == sys.maxsize:
            step_indices = np.concatenate([step_indices, step_indices[-1:]+action_horizon])
        else:
            random_indices_start = self.modality_configs[key.split('.')[0]].random_indices_start
            
            vlm_goal_idx = step_indices[-1:] + action_horizon + random_indices_start//2
            act_goal_idx = step_indices[-1:] + action_horizon + np.random.randint(random_indices_start, 1)

            step_indices = np.concatenate([step_indices, vlm_goal_idx, act_goal_idx])
        # print(f"step_indices: {step_indices}")


        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    




class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for dataset in self.datasets:
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths
            trajectory_sampling_weights /= trajectory_sampling_weights.sum()
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            raise ValueError(
                "No primary dataset found, please at least set one dataset's weight to 1.0"
            )

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = rng.choice(
            len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        # dataset, trajectory_name, step = self.sample_step(index)
        
        # return dataset.transforms(dataset.get_step_data(trajectory_name, step))

        
        max_retries = 10  # Maximum number of retries
        current_index = index
        
        for attempt in range(max_retries):
            try:
                # 1. Call the sampling function to get metadata
                dataset, trajectory_name, step = self.sample_step(index)
                
                # 2. Try to get the actual data (this is where errors may occur)
                return dataset.transforms(dataset.get_step_data(trajectory_name, step))
                
            except Exception as e:
                # Log the error for tracking which file is corrupted
                import logging
                logging.warning(f"Error loading data at index {index} (attempt {attempt+1}): {e}")
                
                # Change the index to resample a different sample
                # Use a random offset or simple increment
                index = (index + 1) % len(self)
                
        raise RuntimeError(f"Failed to load a valid sample after {max_retries} retries.")

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # print(f"self.dataset_lengths: {self.dataset_lengths} !!!")
        # print(f"self.dataset_sampling_weights: {self.dataset_sampling_weights} !!!")
        return int(
            (self.dataset_lengths / self.dataset_sampling_weights)[
                self.primary_dataset_indices
            ].max()
        )

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # # Number of dimensions (assuming consistent across tasks)
            # num_dims = len(per_task_stats[0][modality]["mean"])

            # # Initialize accumulators for means and variances
            # weighted_means = np.zeros(num_dims)
            # weighted_squares = np.zeros(num_dims)

            # --- Key fix: get the full shape instead of just the length ---
            sample_mean = np.array(per_task_stats[0][modality]["mean"])
            data_shape = sample_mean.shape # e.g. (3,) or (16, 3)

            # Initialize accumulators with the correct shape
            weighted_means = np.zeros(data_shape)
            weighted_squares = np.zeros(data_shape)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            # Prevent tiny negative values due to floating-point precision
            overall_variance = np.maximum(overall_variance, 0.0)
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict) -> None:
        """Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def reset_metadata(self, model_path):
        metadatas = load_metadata(model_path)
        for dataset in self.datasets:
            metadata = metadatas.get(dataset.tag)
            print("Reset!!!", metadata)
            dataset.set_transforms_metadata(metadata)
        # Sync update merged_metadata to ensure correct values are written on save
        self.merged_metadata = metadatas




class MultiEmbodimentLeRobotMixtureDataset(LeRobotMixtureDataset):
    # @staticmethod
    # def compute_overall_statistics(
    #     per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    #     dataset_sampling_weights: list[float] | np.ndarray,
    #     percentile_mixing_method: str = "weighted_average",
    # ) -> dict[str, dict[str, list[float]]]:
    #     """
    #     Computes overall statistics from per-task statistics using dataset sample weights.
    #     Only considers tasks that contain the specific modality when computing statistics for that modality.
    #     """
    #     # Convert weights to numpy array for easier indexing, but DO NOT normalize globally yet.
    #     # We will normalize locally for each modality based on available tasks.
    #     raw_dataset_sampling_weights = np.array(dataset_sampling_weights)

    #     # Initialize overall statistics dict
    #     overall_stats: dict[str, dict[str, list[float]]] = {}

    #     # Get the set of all unique modality keys across all tasks
    #     modality_keys = set()
    #     for per_task_stat in per_task_stats:
    #         modality_keys.update(per_task_stat.keys())

    #     for modality in modality_keys:
    #         # 1. Collect data only from tasks that have this modality
    #         valid_means = []
    #         valid_stds = []
    #         valid_mins = []
    #         valid_maxs = []
    #         valid_q01s = []
    #         valid_q99s = []
    #         valid_weights = []

    #         # Iterate through all tasks and their corresponding weights
    #         for task_stats, raw_weight in zip(per_task_stats, raw_dataset_sampling_weights):
    #             if modality in task_stats:
    #                 stats = task_stats[modality]
                    
    #                 valid_means.append(stats["mean"])
    #                 valid_stds.append(stats["std"])
    #                 valid_mins.append(stats["min"])
    #                 valid_maxs.append(stats["max"])
    #                 valid_q01s.append(stats["q01"])
    #                 valid_q99s.append(stats["q99"])
    #                 valid_weights.append(raw_weight)

    #         # Convert to numpy arrays for vectorized calculation
    #         if not valid_weights:
    #             continue # Skip if no task has this modality (should not happen given logic above)

    #         valid_means = np.array(valid_means)
    #         valid_stds = np.array(valid_stds)
    #         valid_mins = np.array(valid_mins)
    #         valid_maxs = np.array(valid_maxs)
    #         valid_q01s = np.array(valid_q01s)
    #         valid_q99s = np.array(valid_q99s)
    #         valid_weights = np.array(valid_weights)

    #         # 2. Re-normalize weights so they sum to 1 for this specific subset of tasks
    #         weight_sum = valid_weights.sum()
    #         if weight_sum == 0:
    #             # Handle edge case where weights might be 0, though unlikely in sampling config
    #             normalized_weights = np.ones_like(valid_weights) / len(valid_weights)
    #         else:
    #             normalized_weights = valid_weights / weight_sum

    #         # 3. Compute Overall Statistics

    #         # Weighted Mean
    #         # shape: (num_valid_tasks, num_dims) -> (num_dims,)
    #         overall_mean = np.average(valid_means, axis=0, weights=normalized_weights)

    #         # Weighted Variance / Std
    #         # Formula: E[X^2] - (E[X])^2
    #         # First compute E[X^2] for each task: variance + mean^2
    #         valid_second_moments = valid_stds**2 + valid_means**2
    #         # Weighted average of second moments
    #         overall_second_moment = np.average(valid_second_moments, axis=0, weights=normalized_weights)
    #         # Overall variance
    #         overall_variance = overall_second_moment - overall_mean**2
    #         # Avoid negative variance due to floating point errors
    #         overall_variance = np.maximum(overall_variance, 0)
    #         overall_std = np.sqrt(overall_variance)

    #         # Min / Max (Global min/max across valid tasks)
    #         overall_min = np.min(valid_mins, axis=0)
    #         overall_max = np.max(valid_maxs, axis=0)

    #         # Quantiles (q01, q99)
    #         if percentile_mixing_method == "weighted_average":
    #             weighted_q01 = np.average(valid_q01s, axis=0, weights=normalized_weights)
    #             weighted_q99 = np.average(valid_q99s, axis=0, weights=normalized_weights)
    #         elif percentile_mixing_method == "min_max":
    #             weighted_q01 = np.min(valid_q01s, axis=0)
    #             weighted_q99 = np.max(valid_q99s, axis=0)
    #         else:
    #             raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

    #         # Store results (convert numpy arrays back to lists)
    #         overall_stats[modality] = {
    #             "min": overall_min.tolist(),
    #             "max": overall_max.tolist(),
    #             "mean": overall_mean.tolist(),
    #             "std": overall_std.tolist(),
    #             "q01": weighted_q01.tolist(),
    #             "q99": weighted_q99.tolist(),
    #         }

    #     return overall_stats

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # print(f"len(per_task_stats): {len(per_task_stats)} len(normalized_weights): {len(normalized_weights)}")

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = set()
        for per_task_stat in per_task_stats:
            modality_keys.update(per_task_stat.keys())

        for modality in modality_keys:
            # # Number of dimensions (assuming consistent across tasks)
            # num_dims = len(per_task_stats[0][modality]["mean"])

            # # Initialize accumulators for means and variances
            # weighted_means = np.zeros(num_dims)
            # weighted_squares = np.zeros(num_dims)

            # --- Key fix: get the full shape instead of just the length ---
            sample_mean = np.array(per_task_stats[0][modality]["mean"])
            data_shape = sample_mean.shape # e.g. (3,) or (16, 3)

            # Initialize accumulators with the correct shape
            weighted_means = np.zeros(data_shape)
            weighted_squares = np.zeros(data_shape)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            # Prevent tiny negative values due to floating-point precision
            overall_variance = np.maximum(overall_variance, 0.0)
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats



    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = MultiEmbodimentLeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = MultiEmbodimentLeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)


    def update_metadata(self, metadata_config: dict) -> None:
        """Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)

        start_idx = 0
        for tag, metadatas in all_metadatas.items():
            # print(f"start_idx: {start_idx} tag: {tag} len(metadatas): {len(metadatas)} len(self.dataset_sampling_weights.tolist()): {len(self.dataset_sampling_weights.tolist())}")
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist()[start_idx:start_idx+len(metadatas)],
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
            start_idx = start_idx+len(metadatas)
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])