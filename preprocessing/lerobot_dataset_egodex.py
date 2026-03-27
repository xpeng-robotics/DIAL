import argparse
import gc
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import validate_episode_buffer, validate_frame, write_info, get_episode_data_index, check_timestamps_sync
from lerobot.datasets.compute_stats import get_feature_stats


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        dtype = features[key]["dtype"]

        if dtype == "string":
            continue  # skip strings
        elif dtype in ["image", "video"]:
            ep_stats[key] = {
                "mean": np.zeros((3, 1, 1)),     # default mean
                "std": np.ones((3, 1, 1)),       # default std
                "min": np.zeros((3, 1, 1)),      # default min
                "max": np.ones((3, 1, 1)),       # default max
                "count": np.array([1]),          # default count
            }
            continue
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1

            ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

    return ep_stats



class EgoDexLeRobotDataset(LeRobotDataset):
    def encode_episode_videos(self, episode_index: int, orig_video_paths: dict) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.

        This method handles video encoding steps:
        - Video encoding via ffmpeg
        - Video info updating in metadata
        - Raw image cleanup

        Args:
            episode_index (int): Index of the episode to encode.
        """
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue

            orig_video_path = orig_video_paths[key]
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(orig_video_path, video_path)
            print(f"copy {orig_video_path} to {video_path}")

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if len(self.meta.video_keys) > 0 and episode_index == 0:
            self.meta.update_video_info()
            write_info(self.meta.info, self.meta.root)  # ensure video info always written properly

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        try:
            validate_frame(frame, self.features)
        except Exception as e:
            print(e)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None, orig_video_paths: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer
        else:
            episode_buffer = episode_data

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        if has_video_keys and not use_batched_encoding:
            self.encode_episode_videos(episode_index, orig_video_paths)

        # `meta.save_episode` should be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        # Check if we should trigger batch encoding
        if has_video_keys and use_batched_encoding:
            raise NotImplementedError

        # Episode data index and timestamp checking
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        # Verify that we have one parquet file per episode and the number of video files matches the number of encoded episodes
        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == (self.num_episodes - self.episodes_since_last_encoding) * len(
            self.meta.video_keys
        )

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()
            

