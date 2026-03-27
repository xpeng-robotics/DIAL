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

import argparse

import numpy as np

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation_augPosRot import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.model.policy_dial import DialPolicy
import json
import os
from transformers import AutoConfig
import numpy as np
import pdb

pdb.set_trace = lambda: None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",  # change this to your model path
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="<EMBODIMENT_TAG>",  # change this to your embodiment tag
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="Name of the environment to run.",
        default="<ENV_NAME>",  # change this to your environment name
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    parser.add_argument("--video_dir", type=str, help="Directory to save videos.", default=None)
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=2)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=1)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    parser.add_argument(
        "--max_episode_steps", type=int, help="Maximum number of steps per episode.", default=1440
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    args = parser.parse_args()

    if args.server:
        # Create a policy
        policy = DialPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        video_delta_indices = np.array([0])

        if args.model_path != "<PATH_TO_YOUR_MODEL>":
            model_config = AutoConfig.from_pretrained(args.model_path)
            if hasattr(model_config, "video_delta_indices"):
                video_delta_indices = np.array(model_config.video_delta_indices)
               
        print(f"video_delta_indices: {video_delta_indices}")

        env_name = args.env_name
        result_path = os.path.join(os.path.split(args.video_dir.replace(f"videos/{env_name}",""))[0], "results.json")
        print(os.path.exists(result_path), result_path)
        if os.path.exists(result_path):
            with open(result_path) as f:
                results = json.load(f)
            print(env_name)
            print(results['tasks'])
            print("\n\n")
            if env_name in results['tasks']:
                print(f"Results for {env_name}:\nSuccess rate: {results['tasks'][env_name]:.2f}")
                exit(0)



        # Create a simulation client
        simulation_client = SimulationInferenceClient(host=args.host, port=args.port)

        print("Available modality configs:")
        pdb.set_trace()
        modality_config = simulation_client.get_modality_config()
        print(modality_config.keys())

        # Create simulation configuration
        config = SimulationConfig(
            env_name=args.env_name,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(video_dir=args.video_dir),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps, max_episode_steps=args.max_episode_steps,
                video_delta_indices=video_delta_indices
            ),
            state_modality_keys=set(modality_config['state'].modality_keys)
        )

        # Run the simulation
        print(f"Running simulation for {args.env_name}...")
        env_name, episode_successes = simulation_client.run_simulation(config)

        # Print results
        print(f"Results for {env_name}:")
        print(f"Success rate: {np.mean(episode_successes):.2f}")

    else:
        raise ValueError("Please specify either --server or --client")
