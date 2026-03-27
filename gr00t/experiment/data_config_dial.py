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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
    CoordinateTransform,
    HierarchicalRelativeTransform,
    LocalAxisTransform
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
    VideoOffsetCrop,
    VideoHorizontalFlip,
)
from gr00t.model.transforms import GR00TTransform, GR00TTransformWithGoalImage
from gr00t.model.backbone.vlm_backbone_dial import DEFAULT_VLM_PATH
import sys
import numpy as np

@dataclass
class BaseDataConfig(ABC):
    vlm_path = DEFAULT_VLM_PATH
    use_bridge = False
    fix_language = None
    random_indices_start = sys.maxsize

    def __init__(self, vlm_path: str = DEFAULT_VLM_PATH, use_bridge: bool = False, ignore_lang_prefix: bool = False):
        self.vlm_path = vlm_path
        self.use_bridge = use_bridge
        self.ignore_lang_prefix = ignore_lang_prefix

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.video_delta_indices,
            modality_keys=self.video_keys,
            random_indices_start=self.random_indices_start,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
            vlm_gap=self.vlm_gap if hasattr(self, "vlm_gap") else None,
            action_gap_range=self.action_gap_range if hasattr(self, "action_gap_range") else None,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


#####################################################################################
# helper functions
#####################################################################################


def import_external_data_config(data_config_str: str) -> Optional[BaseDataConfig]:
    """
    Import and instantiate an external data configuration class.

    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig")
    Supports nested modules like "package.submodule:ClassName"
    """
    if ":" not in data_config_str:
        return None

    import importlib
    import os
    import sys
    from pathlib import Path

    # Add current working directory to Python path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        module_path, class_name = data_config_str.split(":", 1)
        if not module_path or not class_name:
            raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

        print(f"Loading external config: {module_path}.{class_name}")

        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            available = [
                n
                for n in dir(module)
                if not n.startswith("_") and isinstance(getattr(module, n), type)
            ]
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}'. Available: {available}"
            )

        # assert if the class has 'transform' and 'modality_config' methods
        if not hasattr(getattr(module, class_name), "transform"):
            raise AttributeError(f"Class '{class_name}' does not have a 'transform' method")
        if not hasattr(getattr(module, class_name), "modality_config"):
            raise AttributeError(f"Class '{class_name}' does not have a 'modality_config' method")

        return getattr(module, class_name)()

    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        print(f"Config loading failed: {e}")
        print("Example: my_configs:MyConfig, package.submodule:ClassName")
        raise


def load_data_config(data_config_str: str, vlm_path: str=DEFAULT_VLM_PATH, use_bridge: bool=False, ignore_lang_prefix: bool=False) -> BaseDataConfig:
    """
    Get a data config class from a string.
    >>> load_data_config("so100")
    >>> get_data_config("dir.subdir.my_configs:RobotConfig")
    """
    if data_config_str in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[data_config_str](vlm_path=vlm_path, use_bridge=use_bridge, ignore_lang_prefix=ignore_lang_prefix)
    data_config_cls = import_external_data_config(data_config_str)
    if data_config_cls is not None:
        return data_config_cls
    # Yellow warning color
    yellow = "\033[93m"
    reset = "\033[0m"
    raise ValueError(
        f"{yellow}Invalid data_config '{data_config_str}'. "
        f"Available options: {list(DATA_CONFIG_MAP.keys())}, "
        f"or use 'module:ClassName' for external configs{reset}"
    )


class FourierGr1ArmsOnlyDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        print(f"vlm_path: {vlm_path}")
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DualCamDataConfig(So100DataConfig):
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))
###########################################################################################


class UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.rs_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))
###########################################################################################


class FourierGr1FullUpperBodyDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
        "state.neck",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
        "action.neck",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
###########################################################################################


class BimanualPandaGripperDataConfig(BaseDataConfig):
    video_keys = [
        "video.rightHand_view",
        "video.leftHand_view",
        "video.front_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_gripper_qpos",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_gripper_qpos",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_gripper_qpos": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_gripper_qpos": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }
    action_normalization_modes = {
        "action.right_gripper_close": "binary",
        "action.left_gripper_close": "binary",
    }

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaHandDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.rightHand_view",
        "video.leftHand_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_hand",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_hand",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_hand",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_hand": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_hand": "min_max",
    }
    action_normalization_modes = {
        "action.right_hand": "min_max",
        "action.left_hand": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }


###########################################################################################


class SinglePandaGripperDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }
###########################################################################################


class FourierGr1ArmsWaistDataConfig(FourierGr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        return super().transform()
###########################################################################################



class OxeDroidDataConfig(BaseDataConfig):
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class AgibotGenie1DataConfig(BaseDataConfig):
    video_keys = [
        "video.top_head",
        "video.hand_left",
        "video.hand_right",
    ]
    state_keys = [
        "state.left_arm_joint_position",
        "state.right_arm_joint_position",
        "state.left_effector_position",
        "state.right_effector_position",
        "state.head_position",
        "state.waist_position",
    ]
    action_keys = [
        "action.left_arm_joint_position",
        "action.right_arm_joint_position",
        "action.left_effector_position",
        "action.right_effector_position",
        "action.head_position",
        "action.waist_position",
        "action.robot_velocity",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class AgibotGenie1TopHeadGausNormDataConfig(BaseDataConfig):
    video_keys = [
        "video.top_head",
    ]
    state_keys = [
        "state.left_arm_joint_position",
        "state.right_arm_joint_position",
        "state.left_effector_position",
        "state.right_effector_position",
        "state.head_position",
        "state.waist_position",
    ]
    action_keys = [
        "action.left_arm_joint_position",
        "action.right_arm_joint_position",
        "action.left_effector_position",
        "action.right_effector_position",
        "action.head_position",
        "action.waist_position",
        "action.robot_velocity",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "mean_std" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "mean_std" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class EgoDexHandWristGR1OnlyAlignCoordCropDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.rightHand_pos",
        "state.rightHand_rot",

        "state.leftHand_pos",
        "state.leftHand_rot",
    ]
    action_keys = [
        "action.rightHand_pos",
        "action.rightHand_rot",

        "action.leftHand_pos",
        "action.leftHand_rot",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {k: "min_max" for k in state_keys if k.endswith("pos")}
    action_normalization_modes = {k: "min_max" for k in action_keys if k.endswith("pos")}

    # x-right, y-up, z-back -> x-right, y-forward, z-up
    coordintate_transform_matrix = [
        [1.,  0.,  0.],
        [0.,  0., -1.],
        [0.,  1.,  0.]
    ]

    # Left hand: x-fingers forward, y-palm forward, z-thumb backward -> x-fingers forward, y-palm backward, z-thumb backward
    left_hand_local_mat = np.array([
        [ 1,   0,   0], # newX = oldX
        [ 0,  -1,   0], # newY = -oldY
        [ 0,   0,   1]  # newZ = oldZ
    ], dtype=np.float64)


    # Right hand: x-fingers backward, y-palm backward, z-thumb forward -> x-fingers forward, y-palm forward, z-thumb backward
    right_hand_local_mat = np.array([
        [-1,  0,  0], # newX = -oldX
        [ 0, -1,  0], # newY = -oldY
        [ 0,  0, -1]  # newZ = -oldZ
    ], dtype=np.float64)


    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoOffsetCrop(
            apply_to=self.video_keys,
                top=int(1080 * 0.30),     # Start from 30% from top
                left=0,                   # No horizontal offset
                height=int(1080 * 0.70),  # Crop remaining 70% height
                width=1920                 # Width unchanged
            ),
            VideoCrop(apply_to=self.video_keys, scale=0.95, height=756, width=1920),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            CoordinateTransform(
                apply_to=[k for k in self.state_keys if k.endswith("pos")],
                mode="xyz",
                matrix=self.coordintate_transform_matrix,
            ),
            CoordinateTransform(
                apply_to=[k for k in self.state_keys if k.endswith("rot")],
                mode="rot6d",
                matrix=self.coordintate_transform_matrix,
            ),
            LocalAxisTransform(
                apply_to=["state.rightHand_rot"],
                mode="rot6d",
                matrix=self.right_hand_local_mat,
            ),
            LocalAxisTransform(
                apply_to=["state.leftHand_rot"],
                mode="rot6d",
                matrix=self.left_hand_local_mat,
            ),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            CoordinateTransform(
                apply_to=[k for k in self.action_keys if k.endswith("pos")],
                mode="xyz",
                matrix=self.coordintate_transform_matrix,
            ),
            CoordinateTransform(
                apply_to=[k for k in self.action_keys if k.endswith("rot")],
                mode="rot6d",
                matrix=self.coordintate_transform_matrix,
            ),
            LocalAxisTransform(
                apply_to=["action.rightHand_rot"],
                mode="rot6d",
                matrix=self.right_hand_local_mat,
            ),
            LocalAxisTransform(
                apply_to=["action.leftHand_rot"],
                mode="rot6d",
                matrix=self.left_hand_local_mat,
            ),


            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class EgoDexHandWristGR1OnlyAlignCoordGausNormCropDataConfig(EgoDexHandWristGR1OnlyAlignCoordCropDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.rightHand_pos",
        "state.rightHand_rot",

        "state.leftHand_pos",
        "state.leftHand_rot",
    ]
    action_keys = [
        "action.rightHand_pos",
        "action.rightHand_rot",

        "action.leftHand_pos",
        "action.leftHand_rot",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {k: "mean_std" for k in state_keys if k.endswith("pos")}
    action_normalization_modes = {k: "mean_std" for k in action_keys if k.endswith("pos")}



class FourierGr1ArmsWaistAugPosRotFlipDataConfig(FourierGr1ArmsWaistDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        'state.wrist_r_pos', 
        'state.wrist_r_rot6d', 
        'state.thumb_r_pos',
        'state.index_r_pos',
        'state.middle_r_pos',
        'state.ring_r_pos',
        'state.pinky_r_pos',

        'state.wrist_l_pos', 
        'state.wrist_l_rot6d', 
        'state.thumb_l_pos', 
        'state.index_l_pos',  
        'state.middle_l_pos',
        'state.ring_l_pos',
        'state.pinky_l_pos',

        "state.right_arm",
        "state.right_hand",
        "state.left_arm",
        "state.left_hand",
        "state.waist",
    ]
    action_keys = [
        'action.wrist_r_pos', 
        'action.wrist_r_rot6d', 
        'action.thumb_r_pos',
        'action.index_r_pos',
        'action.middle_r_pos',
        'action.ring_r_pos',
        'action.pinky_r_pos',

        'action.wrist_l_pos', 
        'action.wrist_l_rot6d', 
        'action.thumb_l_pos', 
        'action.index_l_pos',  
        'action.middle_l_pos',
        'action.ring_l_pos',
        'action.pinky_l_pos',

        "action.right_arm",
        "action.right_hand",
        "action.left_arm",
        "action.left_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {k: "min_max" for k in state_keys if k.endswith("pos")}
    action_normalization_modes = {k: "min_max" for k in action_keys if not k.endswith("rot6d")}


    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(
                apply_to=[
                    "state.right_arm",
                    "state.right_hand",
                    "state.left_arm",
                    "state.left_hand",
                    "state.waist",
                ]
            ),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class FourierGr1ArmsWaistAugPosRotFlipCropDataConfig(FourierGr1ArmsWaistAugPosRotFlipDataConfig):
    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoOffsetCrop(
                apply_to=self.video_keys,
                top=int(256 * 0.17),      # Start from 15% from top
                left=0,                               # No horizontal offset
                height=int(256 * 0.66),   # Crop middle 70% height
                width=256                  # Width unchanged
            ),
            VideoCrop(apply_to=self.video_keys, scale=0.95, height=168, width=256),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(
                apply_to=[
                    "state.right_arm",
                    "state.right_hand",
                    "state.left_arm",
                    "state.left_hand",
                    "state.waist",
                ]
            ),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransformWithGoalImage(
                vlm_gap=self.vlm_gap if hasattr(self, "vlm_gap") else None,
                action_gap_range=self.action_gap_range if hasattr(self, "action_gap_range") else None,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                vlm_path=self.vlm_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
###########################################################################################

class FourierGr1ArmsWaistAugPosRotFlipWristOnlyCropDataConfig(FourierGr1ArmsWaistAugPosRotFlipCropDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        'state.wrist_r_pos', 
        'state.wrist_r_rot6d', 

        'state.wrist_l_pos', 
        'state.wrist_l_rot6d', 

        "state.right_arm",
        "state.right_hand",
        "state.left_arm",
        "state.left_hand",
        "state.waist",
    ]
    action_keys = [
        'action.wrist_r_pos', 
        'action.wrist_r_rot6d', 

        'action.wrist_l_pos', 
        'action.wrist_l_rot6d', 

        "action.right_arm",
        "action.right_hand",
        "action.left_arm",
        "action.left_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {k: "min_max" for k in state_keys if k.endswith("pos")}
    action_normalization_modes = {k: "min_max" for k in action_keys if not k.endswith("rot6d")}

###########################################################################################

class FourierGr1ArmsWaistAugPosRotFlipWristOnlyGausNormCropDataConfig(FourierGr1ArmsWaistAugPosRotFlipWristOnlyCropDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        'state.wrist_r_pos', 
        'state.wrist_r_rot6d', 

        'state.wrist_l_pos', 
        'state.wrist_l_rot6d', 

        "state.right_arm",
        "state.right_hand",
        "state.left_arm",
        "state.left_hand",
        "state.waist",
    ]
    action_keys = [
        'action.wrist_r_pos', 
        'action.wrist_r_rot6d', 

        'action.wrist_l_pos', 
        'action.wrist_l_rot6d', 

        "action.right_arm",
        "action.right_hand",
        "action.left_arm",
        "action.left_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {k: "mean_std" for k in state_keys if k.endswith("pos")}
    action_normalization_modes = {k: "mean_std" for k in action_keys if not k.endswith("rot6d")}



###########################################################################################

DATA_CONFIG_MAP = {
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig,
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig,
    "fourier_gr1_full_upper_body": FourierGr1FullUpperBodyDataConfig,
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig,
    "bimanual_panda_hand": BimanualPandaHandDataConfig,
    "single_panda_gripper": SinglePandaGripperDataConfig,
    "so100": So100DataConfig,
    "so100_dualcam": So100DualCamDataConfig,
    "unitree_g1": UnitreeG1DataConfig,
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig,
    "oxe_droid": OxeDroidDataConfig,
    "agibot_genie1": AgibotGenie1DataConfig,
    "agibot_genie1_topHead_GausNorm": AgibotGenie1TopHeadGausNormDataConfig,
    "fourier_gr1_arms_waist_aug_pos_rot_flip": FourierGr1ArmsWaistAugPosRotFlipDataConfig,
    "fourier_gr1_arms_waist_aug_pos_rot_flip_crop": FourierGr1ArmsWaistAugPosRotFlipCropDataConfig,
    "fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_crop": FourierGr1ArmsWaistAugPosRotFlipWristOnlyCropDataConfig,
    "fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop": FourierGr1ArmsWaistAugPosRotFlipWristOnlyGausNormCropDataConfig,
    "human_egodex_hand_wrist_gr1_only_align_coord_crop": EgoDexHandWristGR1OnlyAlignCoordCropDataConfig,
    "human_egodex_hand_wrist_gr1_only_align_coord_gausNorm_crop": EgoDexHandWristGR1OnlyAlignCoordGausNormCropDataConfig,
}
