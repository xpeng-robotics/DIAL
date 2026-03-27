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

import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.vlm_backbone_dial import DEFAULT_VLM_PATH
import sys


class BridgeTokenManager:
    def __init__(self):
        self.tokens = []  # initialize an empty list
        self.update(64)   # populate data

    def update(self, num_bridge_tokens):
        # Key: modify list contents in-place without changing the reference address of self.tokens
        new_list = [f"<|bridge_{i}|>" for i in range(num_bridge_tokens)]
        self.tokens.clear()
        self.tokens.extend(new_list)
        
        # This attribute must also be kept in sync
        self.str_ = "".join(self.tokens)
    
    def __str__(self):
        return self.str_
    
    def __len__(self):
        return len(self.tokens)

    # Make the object directly iterable (like a list)
    def __iter__(self):
        return iter(self.tokens)

    # Make Pydantic error messages clearer
    def __repr__(self):
        return f"BridgeTokenManager(cnt={len(self.tokens)})"

# Key: create an instance here named BRIDGE_TOKENS
# so that other files importing it get this already-initialized object directly
BRIDGE_TOKENS = BridgeTokenManager()



def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_vlm_processor(vlm_path: str) -> ProcessorMixin:
    vlm_processor = AutoProcessor.from_pretrained(
        vlm_path, trust_remote_code=True, use_fast=True
    )
    vlm_processor.tokenizer.padding_side = "left"

    # If it's a Qwen2_5_VLProcessor, import process_vision_info and attach it
    if vlm_processor.__class__.__name__ == "Qwen2_5_VLProcessor":
        from qwen_vl_utils import process_vision_info
        # Bind as a method (passing self as the first argument)
        import types
        vlm_processor.process_vision_info = process_vision_info
    
    print(f"vlm_processor num_bridge_tokens: {len(BRIDGE_TOKENS)}")
    vlm_processor.tokenizer.add_special_tokens({"additional_special_tokens": BRIDGE_TOKENS.tokens})

    return vlm_processor


def collate(features: List[dict], vlm_processor) -> dict:
    batch = {}
    keys = features[0].keys()
    # print(keys)
    # print(vlm_processor.__class__)
    # print(features)

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "vlm_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            # print(text_list)
            # print(image_inputs)
            vlm_inputs = vlm_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            for k, v in vlm_inputs.items():
                k = "vlm_" + k
                batch[k] = v
                # print(k, v.shape)

        elif key == "goal_images":
            # print(f"goal_images.shape: {len(values)}, {values}")
            image_inputs = vlm_processor.image_processor(
                images=values,
                videos=None,
                return_tensors="pt"
            )
            for k, v in image_inputs.items():
                k = "goal_image_" + k
                batch[k] = v
                # print(k, v.shape)

        elif key == "act_goal_images":
            # print(f"act_goal_images.shape: {len(values)}, {values}")
            image_inputs = vlm_processor.image_processor(
                images=values,
                videos=None,
                return_tensors="pt"
            )
            for k, v in image_inputs.items():
                k = "act_goal_image_" + k
                batch[k] = v
                # print(k, v.shape)

        elif key == "act_obs_images":
            image_inputs = vlm_processor.image_processor(
                images=values,
                videos=None,
                return_tensors="pt"
            )
            for k, v in image_inputs.items():
                k = "act_obs_image_" + k
                batch[k] = v
                # print(k, v.shape)

        # --- [New] DINO native image Collate ---
        elif key in ("dino_obs_images", "dino_goal_images", "dino_act_goal_images"):
            # values: list of numpy arrays, each with shape [Num_Imgs, C, H, W]
            # After np.stack, shape becomes [Batch, Num_Imgs, C, H, W]
            # batch[key] = torch.from_numpy(np.stack(values)) # uint8 tensor
            try:
                batch[key] = torch.from_numpy(np.stack(values))
            except ValueError as e:
                print(f"\n[Shape Error] Key: {key}")
                for i, v in enumerate(values):
                    print(f"  Sample {i} shape: {v.shape if hasattr(v, 'shape') else type(v)}")
                raise e
        # ------------------------------------
            
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        elif key == "orig_state":
            batch[key] = values
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, vlm_path: str = DEFAULT_VLM_PATH):
        super().__init__()
        self.vlm_processor = build_vlm_processor(vlm_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.vlm_processor)


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    vlm_path: str = DEFAULT_VLM_PATH

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    # vlm_processor: ProcessorMixin = Field(default=build_vlm_processor(vlm_path))
    vlm_processor: ProcessorMixin | None = None   # deferred initialization

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    use_bridge: bool = False
    ignore_lang_prefix: bool = False

    fix_language: Optional[str] = None
    random_indices_start: int = sys.maxsize
    vlm_gap: Optional[int] = None
    action_gap_range: Optional[list]=None

    def model_post_init(self, __context):
        """Called by Pydantic after model init."""
        print(f"transform.vlm_path: {self.vlm_path}")
        print(f"transform.use_bridge: {self.use_bridge}")
        print(f"transform.ignore_lang_prefix: {self.ignore_lang_prefix}")
        if self.vlm_processor is None:
            self.vlm_processor = build_vlm_processor(self.vlm_path)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        vlm_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        vlm_image = [{"type": "image", "image": img} for img in vlm_images]
        vlm_conversation = [
            {
                "role": "user",
                "content": vlm_image + text_content,
            }
        ]

        text_list = [
            self.vlm_processor.apply_chat_template(
                vlm_conversation, tokenize=False, add_generation_prompt=True
            )
        ]

        if self.use_bridge:
            text_list = [
                text + str(BRIDGE_TOKENS) for text in text_list
            ]

        image_inputs, video_inputs = self.vlm_processor.process_vision_info(vlm_conversation)
        vlm_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["vlm_content"] = vlm_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        # print(data["video"].shape, images.shape)
        return images

    def _prepare_language(self, data: dict):
        if (self.fix_language is not None) and (self.fix_language != "split"):
            # print(self.fix_language)
            return self.fix_language

        """Tokenize data['language'] (or default_instruction if missing)."""
        # print(f"self._language_key: {self._language_key}")

        if self._language_key is not None:
            raw_language = data[self._language_key]
            # print(f"type(raw_language): {type(raw_language)} raw_language: {raw_language}")
            if isinstance(raw_language, list) or isinstance(raw_language, np.ndarray ):
                raw_language = str(raw_language[0])

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        if self.ignore_lang_prefix:
            # print(f"raw_language: {raw_language}")
            raw_language = raw_language.split(":")[-1].strip()
            # print(raw_language)

        if self.fix_language == "split":
            raw_language = raw_language.split()[0]
            # print(raw_language)

        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"
            
        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.vlm_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            processed_data = self.apply_batch(data, batch_size)
        else:
            processed_data = self.apply_single(data)
        if "orig_state" in data:
            processed_data["orig_state"] = data["orig_state"]

        return processed_data

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


class GR00TTransformWithGoalImage(GR00TTransform):

    def _apply_goal_image_processing(self, goal_images) -> BatchFeature:
        """
        Args:
            goal_images: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        np_images = rearrange(goal_images, "v t c h w -> (t v) c h w")
        text_content = []

        vlm_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        vlm_image = [{"type": "image", "image": img} for img in vlm_images]
        vlm_conversation = [
            {
                "role": "user",
                "content": vlm_image,
            }
        ]
        image_inputs, video_inputs = self.vlm_processor.process_vision_info(vlm_conversation)
        return image_inputs

    def apply_single(self, data: dict) -> dict:

        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)

        if self.training and self.random_indices_start != sys.maxsize:
            images = images[:,:-1]
            act_goal_images = images[:,-1:]

        if self.training and self.action_gap_range is not None:
            images = images[:,:-1]
            act_obs_images = images[:,-1:]

    
        # action_gap_range: int=None

        if images.shape[1] > 1:
            obs_images = images[:,:-1]
        else:
            obs_images = images

        batch_data = {"images": obs_images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        if images.shape[1] > 1:
            goal_images = images[:,-1:]
            transformed_data["goal_images"] = self._apply_goal_image_processing(goal_images)
        else:
            goal_images = None

        # --- [New] DINO native image pass-through channel ---
        # obs_images: [V, T_obs, C, H, W] -> flatten to [V * T_obs, C, H, W]
        V, T_obs, C, H, W = obs_images.shape
        transformed_data["dino_obs_images"] = obs_images.reshape(V * T_obs, C, H, W)
        
        if goal_images is not None:
            V, T_goal, C, H, W = goal_images.shape
            transformed_data["dino_goal_images"] = goal_images.reshape(V * T_goal, C, H, W)
        
        if self.training and self.random_indices_start != sys.maxsize:
            V, T_act, C, H, W = act_goal_images.shape
            transformed_data["dino_act_goal_images"] = act_goal_images.reshape(V * T_act, C, H, W)
        # ------------------------------------


        
        if self.training and self.random_indices_start != sys.maxsize:
            act_goal_images = self._apply_goal_image_processing(act_goal_images)
            transformed_data["act_goal_images"] = act_goal_images

        if self.training and self.action_gap_range is not None:
            act_obs_images = self._apply_goal_image_processing(act_obs_images)
            transformed_data["act_obs_images"] = act_obs_images


        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images