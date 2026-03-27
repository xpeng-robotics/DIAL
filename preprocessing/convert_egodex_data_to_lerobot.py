"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot_dataset_egodex import EgoDexLeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import os
from pprint import pprint
import json


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> EgoDexLeRobotDataset:
    motors = [
        'camera',
        'hip',
        'leftArm',
        'leftForearm',
        'leftHand',
        'leftIndexFingerIntermediateBase',
        'leftIndexFingerIntermediateTip',
        'leftIndexFingerKnuckle',
        'leftIndexFingerMetacarpal',
        'leftIndexFingerTip',
        'leftLittleFingerIntermediateBase',
        'leftLittleFingerIntermediateTip',
        'leftLittleFingerKnuckle',
        'leftLittleFingerMetacarpal',
        'leftLittleFingerTip',
        'leftMiddleFingerIntermediateBase',
        'leftMiddleFingerIntermediateTip',
        'leftMiddleFingerKnuckle',
        'leftMiddleFingerMetacarpal',
        'leftMiddleFingerTip',
        'leftRingFingerIntermediateBase',
        'leftRingFingerIntermediateTip',
        'leftRingFingerKnuckle',
        'leftRingFingerMetacarpal',
        'leftRingFingerTip',
        'leftShoulder',
        'leftThumbIntermediateBase',
        'leftThumbIntermediateTip',
        'leftThumbKnuckle',
        'leftThumbTip',
        'neck1',
        'neck2',
        'neck3',
        'neck4',
        'rightArm',
        'rightForearm',
        'rightHand',
        'rightIndexFingerIntermediateBase',
        'rightIndexFingerIntermediateTip',
        'rightIndexFingerKnuckle',
        'rightIndexFingerMetacarpal',
        'rightIndexFingerTip',
        'rightLittleFingerIntermediateBase',
        'rightLittleFingerIntermediateTip',
        'rightLittleFingerKnuckle',
        'rightLittleFingerMetacarpal',
        'rightLittleFingerTip',
        'rightMiddleFingerIntermediateBase',
        'rightMiddleFingerIntermediateTip',
        'rightMiddleFingerKnuckle',
        'rightMiddleFingerMetacarpal',
        'rightMiddleFingerTip',
        'rightRingFingerIntermediateBase',
        'rightRingFingerIntermediateTip',
        'rightRingFingerKnuckle',
        'rightRingFingerMetacarpal',
        'rightRingFingerTip',
        'rightShoulder',
        'rightThumbIntermediateBase',
        'rightThumbIntermediateTip',
        'rightThumbKnuckle',
        'rightThumbTip',
        'spine1',
        'spine2',
        'spine3',
        'spine4',
        'spine5',
        'spine6',
        'spine7'
    ]
    cameras = [
        "ego_view",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors)*9,),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors)*9,),
        },
        "annotation.human.coarse_action": {
            "dtype": "int64",
            "shape": (1,),
        }
    }


    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3,1080,1920),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    # if Path(LEROBOT_HOME / repo_id).exists():
    #     shutil.rmtree(LEROBOT_HOME / repo_id)

    return EgoDexLeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    # grab info from HDF5
    mp4_file = ep_path.with_suffix('.mp4')

    with h5py.File(ep_path, "r") as root:
        tfdtype = root['/transforms/camera'][0].dtype
        num_frames = root['/transforms/camera'].shape[0]

        # get SE(3) transforms. Note: all transforms (including camera extrinsics) are expressed in the ARKit origin frame, 
        # which is a stationary frame on the ground that is set at the beginning of a recording session. 
        # the exact position and orientation of the origin frame depends on how the Vision Pro is initialized. 
        # you may want to instead express the transforms in the camera frame (see utils.data_utils.convert_to_camera_frame). 
        # you may also want to grab a "chunk" of N transforms with root['/transforms/'+tf_name][frame_id:frame_id+N] instead of just one. 
        query_tfs = sorted(list(root['/transforms'].keys()))
        assert len(query_tfs) == 69
        tfs = np.zeros([num_frames, len(query_tfs), 9], dtype=tfdtype) # [num_frames, 69, 9]
        for i, tf_name in enumerate(query_tfs):
            transform_matrix = root['/transforms/' + tf_name]
            position = transform_matrix[:, :3, 3] # [num_frames, 3]
            rotation_matrix = transform_matrix[:, :3, :3] # [num_frames, 3, 3]
            rotation_6d = np.concatenate([rotation_matrix[:, :, 0], rotation_matrix[:, :, 1]], axis=-1) # [num_frames, 6]
            pos_rot6d = np.concatenate([position, rotation_6d], axis=-1) # [num_frames, 9]
            tfs[:, i] = pos_rot6d
        tfs = tfs.reshape(num_frames, -1) # [num_frames, 69*9]

        # cam_int = root['/camera/intrinsic'][:] # intrinsics [3, 3]

        # natural language description of task
        if root.attrs['llm_type'] == 'reversible':
            direction = root.attrs['which_llm_description']
            lang_instruct = root.attrs['llm_description' if direction == '1' else 'llm_description2'] 
        else:
            lang_instruct = root.attrs['llm_description']

        lang_instruct = lang_instruct.lower().strip('.') 

        # add joint prediction confidences, if present in this HDF5
        # confs = None
        # if 'confidences' in root.keys():
        #     confs = np.zeros([len(self.query_tfs)], dtype=tfdtype)
        #     for i, tf_name in enumerate(self.query_tfs):
        #         confs[i] = root['/confidences/' + tf_name][frame_id]

    return query_tfs, tfs, mp4_file, lang_instruct


def populate_dataset(
    dataset: EgoDexLeRobotDataset,
    hdf5_files: list[Path],
    # task: str,
    episodes: list[int] | None = None,
) -> EgoDexLeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))
    
    episode_list = []
    task2index = {}

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        # imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        query_tfs, tfs, mp4_file, lang_instruct = load_raw_episode_data(ep_path)
        num_frames = tfs.shape[0]
        task_index = task2index.get(lang_instruct, None)
        if task_index is None:
            task_index = len(task2index)
            task2index[lang_instruct] = task_index

        for i in range(num_frames-1):
            frame = {
                "observation.state": tfs[i],
                "action": tfs[i+1],
                "annotation.human.coarse_action": np.array([task_index]),
                "observation.images.ego_view": np.zeros((3,1080,1920)),
            }

            # for camera, img_array in imgs_per_cam.items():
            #     frame[f"observation.images.{camera}"] = img_array[i]

            # if velocity is not None:
            #     frame["observation.velocity"] = velocity[i]
            # if effort is not None:
            #     frame["observation.effort"] = effort[i]

            dataset.add_frame(frame, task=lang_instruct)

        orig_video_paths = {
            "observation.images.ego_view": mp4_file
        }
        dataset.save_episode(orig_video_paths=orig_video_paths)
        episode_list.append({
            "episode_index": ep_idx,
            "tasks": [lang_instruct],
            "trajectory_id": str(ep_path).split("egodex/")[-1],
            "operator": "unknown",
            "description": ep_path.parent.stem,
            "remarks": lang_instruct,
        })

        # if ep_idx == 5:
        #     break

    state = {}
    action = {}
    for i, key in enumerate(query_tfs):
        state[key+"_pos"] = {
            "original_key": "observation.state",
            "start": i*9,
            "end": i*9+3
        }
        state[key+"_rot"] = {
            "original_key": "observation.state",
            "start": i*9+3,
            "end": i*9+9
        }


        action[key+"_pos"] = {
            "original_key": "action",
            "start": i*9,
            "end": i*9+3,
        }
        action[key+"_rot"] = {
            "original_key": "action",
            "start": i*9+3,
            "end": i*9+9,
            "absolute": True,
            "rotation_type": "rotation_6d"
        }

    modality = {
        "state": state,
        "action": action,
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view"
            }
        },
        "annotation": {
            "human.coarse_action": {
                "original_key": "annotation.human.coarse_action"
            }
        }
    }

    return dataset, task2index, episode_list, modality


def port_aloha(
    raw_dir: Path = Path("data/egodex/part1/add_remove_lid"),
    repo_id: str = "data/egodex_lerobot_v21/part1/add_remove_lid",
    raw_repo_id: str | None = None,
    # task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    overwrite: bool = True,
):
    if overwrite and os.path.exists(repo_id):
        shutil.rmtree(repo_id)

    # if (LEROBOT_HOME / repo_id).exists():
    #     shutil.rmtree(LEROBOT_HOME / repo_id)

    # if not raw_dir.exists():
    #     if raw_repo_id is None:
    #         raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
    #     download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.glob("*.hdf5"), key=lambda item: int(item.stem))
    assert int(hdf5_files[-1].stem)==len(hdf5_files)-1

    # hdf5_files = hdf5_files[:5]

    dataset = create_empty_dataset(
        repo_id,
        robot_type="HumanEgoDex",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset, task2index, episode_list, modality = populate_dataset(
        dataset,
        hdf5_files,
        # task=task,
        episodes=episodes,
    )
    # pprint(task2index)
    # pprint(episode_list)
    with open(os.path.join(repo_id, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, ensure_ascii=False, indent=4)

    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()

    # with open(os.path.join(repo_id, "meta", "episodes.jsonl"), "w") as f:
    #     for episode in tqdm(episode_list):
    #         f.write(json.dumps(episode)+"\n")

    # with open(os.path.join(repo_id, "meta", "tasks.jsonl"), "w") as f:
    #     for task, task_index in tqdm(task2index.items()):
    #         f.write(json.dumps({
    #             "task_index": task_index, 
    #             "task": task
    #         })+"\n")
    

    

if __name__ == "__main__":
    tyro.cli(port_aloha)