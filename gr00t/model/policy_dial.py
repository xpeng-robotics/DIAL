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

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1_dial import GR00T_N1_5_DIAL

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class DialPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer_len: int = None,
        compute_bridge_loss: bool = False,
        guidance_scale: float = 1.0,
        async_vlm_inference: bool = False, # <-- New: whether to enable async
        vlm_update_period: int = 3,        # <-- New: VLM update frequency
    ):
        """
        Initialize the DialPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path, tokenizer_len=tokenizer_len, compute_bridge_loss=compute_bridge_loss)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()
        self._guidance_scale = guidance_scale

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

        self.async_vlm_inference = async_vlm_inference
        self.vlm_update_period = vlm_update_period
        self._async_step = 0
        self._cached_backbone_feat = None

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_input['guidance_scale'] = self._guidance_scale
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action, normalized_input)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def get_action_and_bridgeloss(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        # print(f"normalized_input.keys: {normalized_input.keys()}")
        # print(f"normalized_input['orig_state'].keys: {normalized_input['orig_state'].keys()}")
        normalized_input['guidance_scale'] = self._guidance_scale

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()

        unnormalized_action = self._get_unnormalized_action(normalized_action, normalized_input)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)



        bridge_loss = model_pred.get('bridge_loss', None)
        if bridge_loss is not None:
            # bridge_loss = {'bridge_loss': bridge_loss.cpu().numpy()}
            # if not is_batch:
            #     bridge_loss = squeeze_dict_values(bridge_loss)
            bridge_loss = np.squeeze(bridge_loss.cpu().numpy(), axis=0)
            unnormalized_action['bridge_loss'] = bridge_loss

        return unnormalized_action



    def get_action_and_debug_features(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)

        # 2. Extract the processed current frame (Current RGB)
        # Typically the key is 'vlm_pixel_values' [B, T, C, H, W]
        obs_tensors = normalized_input['vlm_pixel_values']
        # print(f"obs_tensors.shape: {obs_tensors.shape}")
        # curr_rgb = self._tensor_to_rgb(obs_tensors[0, -1]) # Batch 0, last frame
        # print(f"normalized_input.keys: {normalized_input.keys()}")

        # 2. Find the dynamic grid size (note: Qwen typically uses these names)
        grid_thw = normalized_input['vlm_image_grid_thw'][0]

        # 3. Restore RGB
        curr_rgb = self._tensor_to_rgb_from_qwen(obs_tensors, grid_thw)

        # 3. Extract the processed goal frame (Goal RGB)
        # Depending on your config, the key may be 'goal_image_pixel_values' or 'act_goal_image_pixel_values'
        # This Tensor has already gone through the same Resize and Crop

        if 'goal_image_pixel_values' in normalized_input:
            goal_tensors = normalized_input['goal_image_pixel_values']
            # goal_rgb = self._tensor_to_rgb(goal_tensors[0, -1])
            # Note: goal image may have its own grid_thw
            goal_grid_thw = normalized_input.get('goal_image_image_grid_thw', grid_thw.unsqueeze(0))[0]
            goal_rgb = self._tensor_to_rgb_from_qwen(goal_tensors, goal_grid_thw)
        else:
            # If not in inputs, use original image as placeholder (but recommended to include it)
            goal_rgb = curr_rgb 

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            # We can get features by passing params to model.forward or by directly calling submodules
            # Here we assume the model forward has been modified to return features, or we manually call at policy level
            backbone_inputs, action_inputs = self.model.prepare_input(normalized_input)
            
            # 1. Get Backbone output (containing predicted subgoal features)
            backbone_outputs = self.model.backbone(backbone_inputs)
            
            # 2. Replicate model.forward internal logic to get Obs and GT-Goal features
            batch_size = backbone_inputs['state'].shape[0]
            
            # Predicted Subgoal features (after Projector)
            # Note: based on the code logic, features are at [:, -num_bridge_tokens:]
            raw_pred_feat = backbone_outputs['backbone_features'][:, -self.model.num_bridge_tokens:]
            # pred_goal_feat = self.model.bridge_projector(raw_pred_feat)
            if getattr(self.model, 'use_separate_projector_for_loss', False):
                pred_goal_feat = self.model.bridge_projector_for_loss(raw_pred_feat)
            else:
                pred_goal_feat = self.model.bridge_projector(raw_pred_feat)
            
            # Get Current Obs features
            vlm_prefix = "vlm_"
            obs_input = {k.removeprefix(vlm_prefix): v for k, v in backbone_inputs.items() if k.startswith(vlm_prefix)}
            obs_input.pop("image_sizes", None); obs_input.pop("input_ids", None); obs_input.pop("attention_mask", None)
            obs_feat = self.model.bridge_vision_model(*obs_input.values())
            # Reshape to [B, 64, D]
            obs_feat = obs_feat.reshape(batch_size, -1, self.model.num_bridge_tokens, obs_feat.shape[-1])[:, -1]

            # Get GT Subgoal features
            goal_image_prefix = "goal_image_"
            goal_input = {k.removeprefix(goal_image_prefix): v for k, v in backbone_inputs.items() if k.startswith(goal_image_prefix)}
            gt_goal_feat = self.model.bridge_goal_model(*goal_input.values())
            gt_goal_feat = gt_goal_feat.reshape(batch_size, -1, gt_goal_feat.shape[-1])

            # Normal Action Head execution to get actions
            # Construct bridge_outputs for action_head (simplified, reusing forward logic)
            normalized_input['guidance_scale'] = self._guidance_scale
            res = self.model.forward(normalized_input, action_mode=True)
            
        # Return results
        debug_data = {
            "obs_feat": obs_feat[0].float().detach().cpu().numpy(),        # [64, D]
            "gt_goal_feat": gt_goal_feat[0].float().detach().cpu().numpy(), # [64, D]
            "pred_goal_feat": pred_goal_feat[0].float().detach().cpu().numpy(), # [64, D]
            "curr_rgb": curr_rgb,   # current image
            "goal_rgb": goal_rgb    # goal image
        }
        
        # Also include the action results
        unnormalized_action = self._get_unnormalized_action(res["action_pred"].float(), normalized_input)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        bridge_loss = res.get('bridge_loss', None)
        if bridge_loss is not None:
            bridge_loss = np.squeeze(bridge_loss.cpu().numpy(), axis=0)
            unnormalized_action['bridge_loss'] = bridge_loss
        
        return unnormalized_action, debug_data

    # def _tensor_to_rgb(self, tensor):
    #     """Helper: restore model input Tensor to visualizable RGB"""
    #     print(f"tensor.shape: {tensor.shape} !!!")
    #     img = tensor.permute(1, 2, 0).cpu().float().numpy()
    #     # Simple inverse normalization
    #     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    #     return (img * 255).astype(np.uint8)


    def _tensor_to_rgb_from_qwen(self, tensor, grid_thw):
        """
        Accurately restore based on Qwen2VLImageProcessor source code logic.
        tensor: [L, 1176]
        grid_thw: [grid_t, grid_h, grid_w] -> e.g., [1, 16, 16]
        """
        grid_t, grid_h, grid_w = grid_thw.cpu().numpy()
        
        # Constants defined in source code
        patch_size = 14
        temporal_patch_size = 2
        merge_size = 2
        channel = 3

        # 1. Inverse reshape: reverse of the source code flatten_patches reshape
        # Logic: decompose D (1176) back to the post-transpose order in source code
        # Order: grid_t, grid_h // merge_size, grid_w // merge_size, merge_size, merge_size, channel, temporal_patch_size, patch_size, patch_size
        x = tensor.view(
            grid_t,
            grid_h // merge_size,
            grid_w // merge_size,
            merge_size,
            merge_size,
            channel,
            temporal_patch_size,
            patch_size,
            patch_size,
        )

        # 2. Inverse transpose: reverse of source code patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        # Source index mapping: 
        # 0->0(t), 1->3(h_maj), 2->6(w_maj), 3->4(h_min), 4->7(w_min), 5->2(c), 6->1(t_patch), 7->5(h_patch), 8->8(w_patch)
        # We want to restore to: [grid_t, temporal_patch_size, channel, h_major, h_minor, h_patch, w_major, w_minor, w_patch]
        # Corresponding current indices in x: (0, 6, 5, 1, 3, 7, 2, 4, 8)
        x = x.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).contiguous()

        # 3. Restore to image shape [T, C, H, W]
        # Here H = (grid_h // merge_size) * merge_size * patch_size = grid_h * patch_size
        x = x.view(
            grid_t * temporal_patch_size,
            channel,
            grid_h * patch_size,
            grid_w * patch_size
        )

        # 4. Post-processing: take the last frame and convert
        # Reshape to [H, W, C]
        img = x[-1].permute(1, 2, 0).detach().cpu().float().numpy()

        # 5. Inverse normalization (using OPENAI_CLIP_MEAN / STD from source code)
        # Corresponding to [0.48145466, 0.4578275, 0.40821073] and [0.26862954, 0.26130258, 0.27577711]
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        return (img * 255).astype(np.uint8)

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # --- Async inference core logic ---
        backbone_feat_to_use = None
        
        if self.async_vlm_inference:
            # If not an update step and cache exists, use cached features
            if (self._async_step % self.vlm_update_period != 0) and (self._cached_backbone_feat is not None):
                # print("update!")
                backbone_feat_to_use = self._cached_backbone_feat
        # -----------------------


        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(
                normalized_input,
                backbone_features=backbone_feat_to_use
            )

        # --- Update async state ---
        if self.async_vlm_inference:
            self._cached_backbone_feat = model_pred.get("backbone_features")
            self._async_step += 1
            # print(self._async_step)
        # -------------------

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor, normalized_input: torch.Tensor) -> Dict[str, Any]:
        # return self.unapply_transforms({"action": normalized_action.cpu()})
        data_to_unapply = normalized_input.copy()
        data_to_unapply["action"] = normalized_action.cpu()

        for k, v in data_to_unapply.items():
            if isinstance(v, torch.Tensor):
                data_to_unapply[k] = v.to(torch.float64)
            elif isinstance(v, np.ndarray):
                # If it's a numpy array, also convert to float32 tensor,
                # since subsequent Transform matrix operations require tensors
                data_to_unapply[k] = torch.from_numpy(v).to(torch.float64)

        unnormalized_data = self.unapply_transforms(data_to_unapply)
        action_out = {k: v for k, v in unnormalized_data.items() if k.startswith("action.")}
        return action_out

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path, tokenizer_len=None, compute_bridge_loss=False):
        # print("loading model ...")
        model = GR00T_N1_5_DIAL.from_pretrained(
            model_path, 
            tokenizer_len=tokenizer_len,
            torch_dtype=COMPUTE_DTYPE,
            compute_bridge_loss=compute_bridge_loss
        )
        model.eval()  # Set model to eval mode
        # exit()

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from gr00t.model.action_head.flow_matching_action_head_dial import (
                FlowmatchingActionHeadDIAL,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHeadDIAL(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data
