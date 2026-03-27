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

import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro

from gr00t.data.dataset import LeRobotSingleDataset, LeRobotSingleDatasetWithGoalImage
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config_dial import load_data_config
from transformers import AutoConfig
from gr00t.model.policy_dial import BasePolicy, DialPolicy
from gr00t.utils.eval_dial import calc_mse_for_single_trajectory, calc_mse_for_single_trajectory_with_pca_overall
from pprint import pprint
import os
import json
from collections import defaultdict
from gr00t.model.transforms import BRIDGE_TOKENS

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy_dial.py --plot --model-path /path/to/checkpoint
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    # dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    # """Path to the dataset."""

    dataset_path: List[str]
    """Path to the dataset directory or directories, we assume all datasets have the same data config"""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ["left_arm", "right_arm", "left_hand", "right_hand", "waist"])
    """Modality keys to evaluate."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.
    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    steps: int = 150
    """Number of steps to evaluate."""

    trajs: int = 1
    """Number of trajectories to evaluate."""

    start_traj: int = 0
    """Start trajectory to evaluate."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "decord"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    # save_plot_path: str = None
    # """Path to save the plot."""

    save_results_path: str = None
    """Path to save the results."""

    plot_state: bool = False
    """Whether to plot the state."""

    data_split: str = "[-10:]"

    vis_pca: bool = False


def main(args: ArgsConfig):
    model_config = AutoConfig.from_pretrained(args.model_path)
    num_bridge_tokens = model_config.bridge_cfg.get("num_bridge_tokens", 64)
    BRIDGE_TOKENS.update(num_bridge_tokens=num_bridge_tokens)

    if hasattr(model_config, 'ignore_lang_prefix'):
        ignore_lang_prefix = model_config.ignore_lang_prefix
    else:
        ignore_lang_prefix = False
    print(f"ignore_lang_prefix: {ignore_lang_prefix}")

    data_config = load_data_config(
        args.data_config,
        vlm_path=model_config.backbone_cfg['vlm_path'],
        use_bridge=model_config.bridge_cfg['use_bridge'],
        ignore_lang_prefix=ignore_lang_prefix,
    )

    action_keys = data_config.exact_action_keys if hasattr(data_config, "exact_action_keys") else data_config.action_keys
    args.modality_keys = [k.replace("action.", "") for k in action_keys]

    # Set action_horizon from data config if not provided
    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config '{args.data_config}'")

    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        tokenizer = modality_transform.transforms[-1].vlm_processor.tokenizer
        tokenizer_len = len(tokenizer)

        policy: BasePolicy = DialPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            tokenizer_len=tokenizer_len,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_bridge_loss=model_config.bridge_cfg['compute_bridge_loss']
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)


    # if args.save_plot_path is not None:
    #     os.makedirs(args.save_plot_path, exist_ok=True)

    if args.save_results_path is not None:
        args.save_plot_path = os.path.join(args.save_results_path, "plots")
        os.makedirs(args.save_plot_path, exist_ok=True)

    task2result = {}
    for p in config.dataset_path:
        task_name = os.path.basename(p)

        # Create the dataset
        dataset = LeRobotSingleDatasetWithGoalImage(
            dataset_path=p,
            modality_configs=modality,
            video_backend=args.video_backend,
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=args.embodiment_tag,
            split=args.data_split
        )

        print(f"len(dataset): {len(dataset)}")
        print("="*50 + task_name + "="*50)
        print("Total trajectories:", len(dataset.trajectory_lengths))
        print("All trajectories:", dataset.trajectory_lengths)
        print("Running on all trajs with modality keys:", args.modality_keys)

        all_action_mse = []
        all_bridge_loss = []
        for i in range(args.start_traj, args.start_traj + args.trajs):
            traj_id = dataset.trajectory_ids[i]
            print("Running trajectory:", traj_id)

            if args.vis_pca:
                action_mse, bridge_loss = calc_mse_for_single_trajectory_with_pca_overall(
                    policy,
                    dataset,
                    traj_id,
                    modality_keys=args.modality_keys,
                    # steps=args.steps,
                    steps=dataset.trajectory_lengths[i],
                    action_horizon=args.action_horizon,
                    plot=args.plot,
                    plot_state=args.plot_state,
                    save_plot_path=os.path.join(args.save_plot_path, f"task{task_name}_traj{traj_id}.png"),
                )
            else:
                action_mse, bridge_loss = calc_mse_for_single_trajectory(
                    policy,
                    dataset,
                    traj_id,
                    modality_keys=args.modality_keys,
                    # steps=args.steps,
                    steps=dataset.trajectory_lengths[i],
                    action_horizon=args.action_horizon,
                    plot=args.plot,
                    plot_state=args.plot_state,
                    save_plot_path=os.path.join(args.save_plot_path, f"task{task_name}_traj{traj_id}.png"),
                )
            print("Action MSE:", action_mse)
            all_action_mse.append(action_mse)
            if bridge_loss is not None:
                print("Bridge Loss:", bridge_loss)
                all_bridge_loss.append(bridge_loss)
        
        avg_action_mse = np.mean(all_action_mse).item()
        task2result[task_name] = {
            "action_mse": avg_action_mse
        }
        print("Average MSE across all trajs:", avg_action_mse)

        if len(all_bridge_loss) > 0:
            avg_bridge_loss = np.mean(all_bridge_loss).item()
            task2result[task_name]["bridge_loss"] = avg_bridge_loss
            print("Average Bridge Loss across all trajs:", avg_bridge_loss)

        
        print("Done")
        # break
        
    total_metrics = {}

    for task_name, metrics in task2result.items():
        for k in metrics:
            if k not in total_metrics:
                total_metrics[k] = []

            total_metrics[k].append(metrics[k])
    for k in total_metrics:
        total_metrics[k] = np.mean(total_metrics[k]).item()

    task2result['total'] = total_metrics
    pprint(task2result)
    with open(os.path.join(args.save_results_path, "results.json"), "w") as f:
        json.dump(task2result, f, ensure_ascii=False, indent=4)

    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
