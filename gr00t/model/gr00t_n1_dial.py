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

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import torchvision.transforms.functional as TF
from copy import deepcopy
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import timm

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .action_head.cross_attention_dit import BridgeCrossAttentionTransformer
from .action_head.flow_matching_action_head_dial import (
    FlowmatchingActionHeadDIAL,
    FlowmatchingActionHeadDIALConfig,
)
from .backbone import VLMBackbone
from gr00t.model.transforms import BRIDGE_TOKENS

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


def compute_global_mean_std_patch(feat: torch.Tensor):
    """
    feat: [B, P, D]
    Returns:
        global_mean: [P, D]
        global_std:  [P, D]
    """
    x = feat.float()
    B, P, D = x.shape
    device = x.device

    distributed = dist.is_available() and dist.is_initialized()

    # ------ local statistics ------
    local_count = torch.tensor([B], device=device, dtype=torch.float32)  # batch size on current GPU
    local_sum = x.sum(dim=0)               # [P, D]
    local_sum_sq = (x * x).sum(dim=0)      # [P, D]

    # ------ distributed reduce ------
    if distributed:
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)

    # ------ compute global mean & variance ------
    global_mean = local_sum / local_count
    global_var = local_sum_sq / local_count - global_mean ** 2
    global_std = torch.sqrt(global_var)

    global_mean, global_std = global_mean.to(feat.device).detach(), global_std.to(feat.device).detach()

    return global_mean, global_std


@dataclass
class GR00T_N1_5_DIAL_Config(PretrainedConfig):
    model_type = "gr00t_n1_5_dial"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    bridge_cfg: dict = field(init=False, metadata={"help": "Bridge configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    bridge_cross_attention_cfg: dict = field(default=None)

    classifier_free_guidance_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5_DIAL(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_DIAL_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_DIAL_Config,
        local_model_path: str,
        tokenizer_len: int=None,
        bridge_type: str="end2end",
        compute_bridge_loss: bool=False,
        select_layer: int=None,
        goal_image_type: str="future",
        bridge_loss_type: str="mse",
        use_image_type_embedding: bool=False,
        use_vl_mask: bool=False,
        correct_vl_mask: bool=False,
        action_only_one_obs: bool=False,

        noise_tau: float=0,
        omit_image_type_embedding_for_goal: bool=False,
        reweight_noise: bool=False,

        unified_embodiment_id: int=None,
        use_separate_projector_for_loss: bool=False,
        vlm_small_lr: float=None,
        select_layer_for_bridge: int=None,
        matching_coeff: float = None,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        if select_layer is not None:
            config.backbone_cfg['select_layer'] = select_layer
        if select_layer_for_bridge is not None:
            config.backbone_cfg['select_layer_for_bridge'] = select_layer_for_bridge

        config.action_head_cfg['use_vl_mask'] = use_vl_mask
        config.action_head_cfg['correct_vl_mask'] = correct_vl_mask
        if matching_coeff is not None:
            config.action_head_cfg['matching_coeff'] = matching_coeff

        super().__init__(config)
        self.local_model_path = local_model_path

        if tokenizer_len is None:
            tokenizer_len = config.bridge_cfg['tokenizer_len']
        else:
            if 'tokenizer_len' in config.bridge_cfg:
                assert tokenizer_len == config.bridge_cfg['tokenizer_len']
                
        self.backbone = VLMBackbone(**config.backbone_cfg, tokenizer_len=tokenizer_len)
        action_head_cfg = FlowmatchingActionHeadDIALConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHeadDIAL(action_head_cfg, backbone_hidden_size=self.backbone.vlm_model.config.hidden_size)

        print(f"Use bridge: {self.config.bridge_cfg['use_bridge']}")
        self.use_bridge = self.config.bridge_cfg['use_bridge']
        if self.config.bridge_cfg['use_bridge']:
            self.use_dino_vision = self.config.bridge_cfg.get('use_dino_vision', False)
            self.use_dino_goal = self.config.bridge_cfg.get('use_dino_goal', False)
            self.dino_path = self.config.bridge_cfg.get('dino_path', None)

            if self.use_dino_vision or self.use_dino_goal:
                print(f"Loading DINO from {self.dino_path}")
                self.dino_model = timm.create_model(
                    model_name=self.dino_path,
                    pretrained=True,
                    num_classes=0,
                    img_size=224,
                )

                if not hasattr(self.dino_model, '_initialize_weights'):
                    self.dino_model._initialize_weights = lambda *args, **kwargs: None

                dino_dim = self.dino_model.num_features
                llm_hidden_size = self.backbone.vlm_model.config.hidden_size
                # Dimension alignment projection layer
                self.dino_proj = nn.Linear(dino_dim, llm_hidden_size) if dino_dim != llm_hidden_size else nn.Identity()
                # DINO normalization
                self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            if not (self.use_dino_vision and self.use_dino_goal):
                if hasattr(self.backbone.vlm_model, "vision_model"):
                    self.bridge_vision_model = deepcopy(self.backbone.vlm_model.vision_model)
                    self.bridge_goal_model = deepcopy(self.backbone.vlm_model.vision_model)
                else:
                    self.bridge_vision_model = deepcopy(self.backbone.vlm_model.visual)
                    self.bridge_goal_model = deepcopy(self.backbone.vlm_model.visual)

            print(f"GR00T_N1_5_DIAL num_bridge_tokens: {len(BRIDGE_TOKENS)}")
            self.num_bridge_tokens = len(BRIDGE_TOKENS)

            self.bridge_projector = nn.Sequential(
                nn.Linear(self.backbone.vlm_model.config.hidden_size, self.backbone.vlm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.backbone.vlm_model.config.hidden_size, self.backbone.vlm_model.config.hidden_size),
            )

            self.use_separate_projector_for_loss = use_separate_projector_for_loss
            if self.use_separate_projector_for_loss:
                self.bridge_projector_for_loss = deepcopy(self.bridge_projector)

            if config.bridge_cross_attention_cfg is not None:
                print(f"config.bridge_cross_attention_cfg: {config.bridge_cross_attention_cfg}")
                self.bridge_cross_attention = BridgeCrossAttentionTransformer(**config.bridge_cross_attention_cfg)
            else:
                self.bridge_cross_attention = None

            if config.classifier_free_guidance_cfg is not None:
                print(f"config.classifier_free_guidance_cfg: {config.classifier_free_guidance_cfg}")
                self.null_subgoal_embed = nn.Parameter(torch.randn(1, self.num_bridge_tokens, self.backbone.vlm_model.config.hidden_size))
                self.p_uncond = config.classifier_free_guidance_cfg['p_uncond']
                self.register_buffer("null_initialized", torch.tensor(False))
                assert 0 < self.p_uncond < 1
            else:
                self.p_uncond = None

            self.vlm_small_lr = vlm_small_lr
            print(f"vlm_small_lr: {self.vlm_small_lr}")

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.bridge_type = bridge_type
        self.compute_bridge_loss = compute_bridge_loss
        self.goal_image_type = goal_image_type
        self.bridge_loss_type = bridge_loss_type

        self.use_image_type_embedding = use_image_type_embedding
        self.omit_image_type_embedding_for_goal = omit_image_type_embedding_for_goal
        if use_image_type_embedding:
            self.image_type_embedding = nn.Embedding(3, self.backbone.vlm_model.config.hidden_size)
            nn.init.normal_(self.image_type_embedding.weight, mean=0.0, std=0.02)
        
        self.action_only_one_obs = action_only_one_obs
        self.noise_tau = noise_tau
        self.reweight_noise = reweight_noise
        self.action_head.unified_embodiment_id = unified_embodiment_id
        self.register_buffer("bridge_training_steps", torch.tensor(0, dtype=torch.float32), persistent=False)

    def _extract_dino(self, pixel_values, downsample_to_n=None):
        """
        Processing pipeline: Resize(224) -> Norm -> DINO -> (Optional) Pool -> Proj
        pixel_values: [B, C, H, W] or [B*V, C, H, W]
        """
        if pixel_values.shape[-1] != 224 or pixel_values.shape[-2] != 224:
            x = F.interpolate(pixel_values.float(), size=(224, 224), mode='bicubic', align_corners=False)
        else:
            x = pixel_values
        if x.max() > 1.0: x = x / 255.0
        x = (x - self.dino_mean) / self.dino_std

        with torch.set_grad_enabled(getattr(self, 'tune_bridge_visual', False) and self.training):
            feats = self.dino_model.forward_features(x)
            patch_feats = feats[:, 5:, :]

        if downsample_to_n is not None:
            B_V, N, D = patch_feats.shape
            side = int(np.sqrt(N)) # 16
            target_side = int(np.sqrt(downsample_to_n)) # 8
            stride = side // target_side # 2
            
            grid = patch_feats.transpose(1, 2).reshape(B_V, D, side, side)
            grid = F.avg_pool2d(grid, kernel_size=stride, stride=stride)
            patch_feats = grid.flatten(2).transpose(1, 2)

        return self.dino_proj(patch_feats)

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
        action_mode: bool=False,
        backbone_features: torch.Tensor = None,
    ) -> BatchFeature:
        guidance_scale = inputs.pop("guidance_scale", 1.0)

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        if action_mode and (backbone_features is not None):
            backbone_outputs = BatchFeature(data={"backbone_features": backbone_features})
        else:
            backbone_outputs = self.backbone(backbone_inputs)

        if action_mode:
            raw_backbone_features = deepcopy(backbone_outputs["backbone_features"])

        batch_size = inputs['state'].shape[0]

        output_dict = {}
        if self.use_bridge:
            if self.vlm_small_lr is not None:
                backbone_outputs['backbone_features'] = backbone_outputs['backbone_features'] * self.vlm_small_lr + backbone_outputs['backbone_features'].detach() * (1-self.vlm_small_lr)

            backbone_outputs.data = {k: v[:, -self.num_bridge_tokens:] for k, v in backbone_outputs.items()}

            # Extract raw backbone features (before being overwritten by projector)
            raw_backbone_feat = backbone_outputs['backbone_features']
            # If backbone provides dedicated bridge layer features, use them for loss computation
            raw_feat_for_loss = backbone_outputs.get('backbone_features_for_bridge', raw_backbone_feat)

            # Compute features for Action Head (main Projector)
            projected_feat_action = self.bridge_projector(raw_backbone_feat)
            
            # Compute features for Bridge Loss
            if getattr(self, 'use_separate_projector_for_loss', False):
                projected_feat_loss = self.bridge_projector_for_loss(raw_feat_for_loss)
            else:
                projected_feat_loss = self.bridge_projector(raw_feat_for_loss)

            # Put loss features back into backbone_outputs for subsequent bridge_loss computation
            backbone_outputs['backbone_features'] = projected_feat_loss

            # Process current observation (supports multi-view)
            if self.use_dino_vision:
                # dino_img: [Batch, Views, C, H, W]  (C=3, H=224, W=224, uint8)
                dino_img = backbone_inputs['dino_obs_images']
                B, V, C, H, W = dino_img.shape
                
                # Flatten Batch and Views dimensions to feed into DINO
                dino_img_flat = dino_img.view(B * V, C, H, W)

                raw_dino_obs = self._extract_dino(dino_img_flat, downsample_to_n=None)
                num_views = raw_dino_obs.shape[0] // batch_size
                obs_embeds = raw_dino_obs.reshape(batch_size, num_views * 256, -1)
                obs_tokens_per_img = 256
            else:
                if "act_obs_image_pixel_values" in backbone_inputs:
                    obs_image_prefix = "act_obs_image_"
                    obs_input = {
                        k.removeprefix(obs_image_prefix): v
                        for k, v in backbone_inputs.items()
                        if k.startswith(obs_image_prefix)
                    }
                else:
                    # Original VLM extraction logic
                    obs_input = {k.removeprefix("vlm_"): v for k, v in backbone_inputs.items() if k.startswith("vlm_")}
                    obs_input.pop("image_sizes", None); obs_input.pop("input_ids", None); obs_input.pop("attention_mask", None)

                obs_embeds = self.bridge_vision_model(*obs_input.values())
                if self.action_only_one_obs:
                    obs_embeds = obs_embeds.reshape(batch_size, -1, self.num_bridge_tokens, obs_embeds.shape[-1])
                    obs_embeds = obs_embeds[:, -1]
                else:
                    obs_embeds = obs_embeds.reshape(batch_size, -1, obs_embeds.shape[-1])
                obs_tokens_per_img = self.num_bridge_tokens
                num_views = obs_embeds.shape[1] // obs_tokens_per_img

            if self.bridge_loss_type == "cosine":
                obs_embeds = F.normalize(obs_embeds, p=2, dim=-1)

            if self.bridge_loss_type == "cosine":
                backbone_outputs['backbone_features'] = F.normalize(backbone_outputs['backbone_features'], p=2, dim=-1)
                projected_feat_action = F.normalize(projected_feat_action, p=2, dim=-1) # synchronize normalization

            if self.compute_bridge_loss or self.bridge_type == "golden":
                if self.goal_image_type == "future":
                    if self.use_dino_goal:
                        dino_goal_img = backbone_inputs['dino_goal_images'] # [Batch, 1, C, H, W]
                        B, V_goal, C, H, W = dino_goal_img.shape
                        
                        goal_img_flat = dino_goal_img.view(B * V_goal, C, H, W)
                        # Extract and downsample to 64 tokens
                        raw_dino_goal = self._extract_dino(goal_img_flat, downsample_to_n=self.num_bridge_tokens)
                        
                        goal_image_embeds = raw_dino_goal.view(B, V_goal * self.num_bridge_tokens, -1)
                    else:
                        goal_image_prefix = "goal_image_"
                        goal_image_input = {
                            k.removeprefix(goal_image_prefix): v
                            for k, v in backbone_inputs.items()
                            if k.startswith(goal_image_prefix)
                        }
                        goal_image_embeds = self.bridge_goal_model(*goal_image_input.values())
                        goal_image_embeds = goal_image_embeds.reshape(batch_size, -1, goal_image_embeds.shape[-1])
                        
                    if self.bridge_loss_type == "cosine":
                        goal_image_embeds = F.normalize(goal_image_embeds, p=2, dim=-1)

                elif self.goal_image_type == "current":
                    goal_image_embeds = obs_embeds
                else:
                    raise NotImplementedError

            if self.compute_bridge_loss:
                if self.bridge_loss_type == "mse":
                    bridge_loss = F.mse_loss(backbone_outputs['backbone_features'], goal_image_embeds)
                elif self.bridge_loss_type == "cosine":
                    cos_sim = (backbone_outputs['backbone_features'] * goal_image_embeds).sum(dim=-1)
                    bridge_loss = 1 - cos_sim.mean()
                elif self.bridge_loss_type == "mse_cosine":
                    mse_loss = F.mse_loss(backbone_outputs['backbone_features'], goal_image_embeds)
                    cosine_loss = 1 - F.cosine_similarity(backbone_outputs['backbone_features'], goal_image_embeds, dim=-1).mean()
                    bridge_loss = mse_loss + cosine_loss
                    output_dict['mse_loss'] = mse_loss
                    output_dict['cosine_loss'] = cosine_loss
                else:
                    raise NotImplementedError
                output_dict['bridge_loss'] = bridge_loss

            if "end2end" in self.bridge_type:
                # Now use features dedicated to Action instead
                goal_image_embeds = projected_feat_action
            elif self.bridge_type == "golden":
                if self.goal_image_type == "future" and "act_goal_image_pixel_values" in backbone_inputs:
                    if self.use_dino_goal:
                        dino_goal_img = backbone_inputs['dino_act_goal_images'] # [Batch, 1, C, H, W]
                        B, V_goal, C, H, W = dino_goal_img.shape
                        
                        goal_img_flat = dino_goal_img.view(B * V_goal, C, H, W)
                        # Extract and downsample to 64 tokens
                        raw_dino_goal = self._extract_dino(goal_img_flat, downsample_to_n=self.num_bridge_tokens)
                        
                        goal_image_embeds = raw_dino_goal.view(B, V_goal * self.num_bridge_tokens, -1)
                    
                    else:
                        goal_image_prefix = "act_goal_image_"
                        goal_image_input = {
                            k.removeprefix(goal_image_prefix): v
                            for k, v in backbone_inputs.items()
                            if k.startswith(goal_image_prefix)
                        }
                        goal_image_embeds = self.bridge_goal_model(*goal_image_input.values())
                        goal_image_embeds = goal_image_embeds.reshape(batch_size, -1, goal_image_embeds.shape[-1])
                        
                    if self.bridge_loss_type == "cosine":
                        goal_image_embeds = F.normalize(goal_image_embeds, p=2, dim=-1)

                if self.training and self.noise_tau > 0:
                    noise_sigma = self.noise_tau * torch.rand((goal_image_embeds.size(0),) + (1,) * (len(goal_image_embeds.shape) - 1), device=goal_image_embeds.device)
                    noise = torch.randn_like(goal_image_embeds)
                    if self.reweight_noise:
                        mean, std = compute_global_mean_std_patch(goal_image_embeds)
                        noise = noise * std
                    noise = noise_sigma * noise
                    goal_image_embeds = goal_image_embeds + noise
            else:
                raise NotImplementedError

            if self.bridge_cross_attention is not None:
                goal_image_embeds = self.bridge_cross_attention(
                    hidden_states=obs_embeds,
                    encoder_hidden_states=goal_image_embeds,
                )

            if self.training and self.p_uncond is not None:
                if not self.null_initialized:
                    with torch.no_grad():
                        # 1. Compute local batch mean on current GPU
                        local_mean = goal_image_embeds.mean(dim=0, keepdim=True) # [1, L, D]
                        
                        # 2. Global sum
                        dist.all_reduce(local_mean, op=dist.ReduceOp.SUM)
                        
                        # 3. Divide by total number of GPUs to get global mean
                        world_size = dist.get_world_size()
                        global_mean = local_mean / world_size
                        
                        # 4. Assign
                        self.null_subgoal_embed.data.copy_(global_mean)
                        self.null_initialized.fill_(True)

                # Randomly generate mask, B is batch_size
                mask = torch.bernoulli(torch.full((batch_size, 1, 1), self.p_uncond)).to(goal_image_embeds.device)
                # If mask=1, use null_subgoal_embed; if mask=0, use normal goal_image_embeds
                goal_image_embeds = (1 - mask) * goal_image_embeds + mask * self.null_subgoal_embed

            # Construct negative samples for discriminative task (only during training when enabled)
            use_matching = self.config.bridge_cfg.get('use_matching_loss', False)
            if self.training and use_matching:
                assert self.p_uncond is None
                batch_size = goal_image_embeds.shape[0]
                assert batch_size > 1
                # Use roll to construct misaligned negative sample indices: [0,1,2,3] -> [3,0,1,2]
                idx = torch.arange(batch_size, device=goal_image_embeds.device).roll(shifts=1, dims=0)
                neg_goal_embeds = goal_image_embeds[idx]
                
                # Construct double-batch features
                # First half is positive samples (Obs_i, Goal_i), second half is negative samples (Obs_i, Goal_j)
                goal_image_embeds = torch.cat([goal_image_embeds, neg_goal_embeds], dim=0)
                obs_embeds = torch.cat([obs_embeds, obs_embeds], dim=0)
                
                # Construct labels: 1 for match, 0 for mismatch
                match_labels = torch.cat([
                    torch.ones(batch_size, 1, device=goal_image_embeds.device),
                    torch.zeros(batch_size, 1, device=goal_image_embeds.device)
                ], dim=0)
                action_inputs["match_labels"] = match_labels
                
                # Important: must also double the other items in action_inputs
                for k in ["state", "action", "action_mask", "embodiment_id"]:
                    if k in action_inputs:
                        action_inputs[k] = torch.cat([action_inputs[k], action_inputs[k]], dim=0)

            if action_mode and guidance_scale > 1.0 and self.p_uncond is not None:
                # Construct double batch: [Cond, Uncond]
                uncond_goal = self.null_subgoal_embed.expand(batch_size, -1, -1)
                goal_image_embeds = torch.cat([goal_image_embeds, uncond_goal], dim=0)
                obs_embeds = torch.cat([obs_embeds, obs_embeds], dim=0)

            if self.use_image_type_embedding:
                if not self.omit_image_type_embedding_for_goal:
                    goal_image_embeds = goal_image_embeds + self.image_type_embedding.weight[0]

                # Add multi-view encoding in a loop (Index 1, 2...)
                for i in range(num_views):
                    start_i, end_i = i * obs_tokens_per_img, (i + 1) * obs_tokens_per_img
                    obs_embeds[:, start_i:end_i] += self.image_type_embedding.weight[i+1]

            bridge_features = torch.cat([goal_image_embeds, obs_embeds], dim=1)
            bridge_attention_mask = torch.ones(bridge_features.shape[:-1], device=bridge_features.device)

            bridge_outputs = BatchFeature(
                data={
                    "backbone_features": bridge_features,
                    "backbone_attention_mask": bridge_attention_mask
                }
            )
        else:
            bridge_outputs = backbone_outputs

        if not action_mode:
            action_head_outputs = self.action_head(bridge_outputs, action_inputs)
            self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

            output_dict['action_loss'] = action_head_outputs['loss']
            if "match_loss" in action_head_outputs:
                output_dict['match_loss'] = action_head_outputs['match_loss']
                output_dict['raw_action_loss'] = action_head_outputs['action_loss']

            loss = action_head_outputs['loss']
            if 'bridge_loss' in output_dict:
                bridge_cfg = self.config.bridge_cfg

                if bridge_cfg.get('bridge_loss_decay_steps', None) is not None:
                    if self.training:
                        self.bridge_training_steps += 1
                    
                    start_w = bridge_cfg.get('bridge_loss_start_w', 1.0)
                    end_w = bridge_cfg.get('bridge_loss_end_w', 0.1)
                    decay_steps = bridge_cfg['bridge_loss_decay_steps']

                    progress = min(1.0, self.bridge_training_steps.item() / max(1, decay_steps))
                    current_w = start_w - (start_w - end_w) * progress

                    loss = (loss + current_w * output_dict['bridge_loss']) / (1.0 + current_w)
                    output_dict['bridge_w'] = torch.tensor(current_w).to(loss.device)
                else:
                    loss = (loss + output_dict['bridge_loss']) / 2

            output_dict['loss'] = loss
            return BatchFeature(data=output_dict)

        else:
            action_head_outputs = self.action_head.get_action(bridge_outputs, action_inputs, guidance_scale=guidance_scale)
            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
            
            action_head_outputs.update(output_dict)
            action_head_outputs["backbone_features"] = raw_backbone_features
            return action_head_outputs

    def get_action(
        self,
        inputs: dict,
        backbone_features=None
    ) -> BatchFeature:
        return self.forward(inputs=inputs, action_mode=True, backbone_features=backbone_features)

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        for key in ["raw_obs_images", "raw_goal_images", "raw_act_goal_images"]:
            if key in inputs and isinstance(inputs[key], list):
                tensors = [TF.to_tensor(img) for img in inputs[key]]
                inputs[key] = torch.stack(tensors).to(self.device, dtype=torch.float32)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    def state_dict(self, *args, **kwargs):
        full_dict = super().state_dict(*args, **kwargs)
        if getattr(self, '_shared_bridge_models', False):
            full_dict = {k: v for k, v in full_dict.items() if not k.startswith("bridge_goal_model.")}
        return full_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        goal_prefix = prefix + "bridge_goal_model."
        has_goal_keys = any(k.startswith(goal_prefix) for k in state_dict)
        if not has_goal_keys:
            missing_keys[:] = [k for k in missing_keys if not k.startswith(goal_prefix)]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, resume_pretrained_option: str="all", **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        tune_visual = kwargs.pop("tune_visual", model_config.backbone_cfg['tune_visual'])
        tune_llm = kwargs.pop("tune_llm", model_config.backbone_cfg['tune_llm'])
        tune_bridge_embedding = kwargs.pop("tune_bridge_embedding", model_config.backbone_cfg['tune_bridge_embedding'])
        tokenizer_len = kwargs.pop("tokenizer_len", None)
        tune_projector = kwargs.pop("tune_projector", model_config.action_head_cfg['tune_projector'])
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", model_config.action_head_cfg['tune_diffusion_model'])

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune backbone bridge embedding: {tune_bridge_embedding}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        try:
            bridge_type = kwargs.pop("bridge_type", model_config.bridge_cfg.get('bridge_type', "end2end"))
            compute_bridge_loss = kwargs.pop("compute_bridge_loss", model_config.bridge_cfg.get('compute_bridge_loss', False))
            goal_image_type = kwargs.pop("goal_image_type", model_config.bridge_cfg.get('goal_image_type', 'future'))
            bridge_loss_type = kwargs.pop("bridge_loss_type", model_config.bridge_cfg.get('bridge_loss_type', 'mse'))
            tune_all_llm_embedding = kwargs.pop("tune_all_llm_embedding", model_config.bridge_cfg.get('tune_all_llm_embedding', False))
            use_image_type_embedding = kwargs.pop("use_image_type_embedding", model_config.bridge_cfg.get('use_image_type_embedding', False))
            omit_image_type_embedding_for_goal = kwargs.pop("omit_image_type_embedding_for_goal", model_config.bridge_cfg.get('omit_image_type_embedding_for_goal', False))
            action_only_one_obs = kwargs.pop("action_only_one_obs", model_config.bridge_cfg.get('action_only_one_obs', False))
            noise_tau = kwargs.pop("noise_tau", model_config.bridge_cfg.get('noise_tau', 0))
            reweight_noise = kwargs.pop("reweight_noise", model_config.bridge_cfg.get('reweight_noise', None))
            unified_embodiment_id = kwargs.pop("unified_embodiment_id", model_config.bridge_cfg.get('unified_embodiment_id', None))

            bridge_loss_end_w = kwargs.pop("bridge_loss_end_w", model_config.bridge_cfg.get('bridge_loss_end_w', None))
            bridge_loss_decay_steps = kwargs.pop("bridge_loss_decay_steps", model_config.bridge_cfg.get('bridge_loss_decay_steps', None))
            use_separate_projector_for_loss = kwargs.pop("use_separate_projector_for_loss", model_config.bridge_cfg.get('use_separate_projector_for_loss', False))
            vlm_small_lr = kwargs.pop("vlm_small_lr", model_config.bridge_cfg.get('vlm_small_lr', False))
            matching_coeff = kwargs.pop("matching_coeff", model_config.action_head_cfg.get('matching_coeff', None))
        except Exception as e:
            print(kwargs)
            raise e
        print(f"Bridge type: {bridge_type}")
        print(f"Compute bridge loss: {compute_bridge_loss}")
        print(f"Goal image type: {goal_image_type}")
        print(f"Bridge loss type: {bridge_loss_type}")
        print(f"Tune all llm token embeddings: {tune_all_llm_embedding}")
        print(f"Use image type embeddings: {use_image_type_embedding}")
        print(f"Omit image type embeddings for goal images: {omit_image_type_embedding_for_goal}")
        print(f"Action head using only one obs: {action_only_one_obs}")
        print(f"Noise Tau: {noise_tau}")
        print(f"Reweight Noise: {reweight_noise}")
        print(f"unified_embodiment_id: {unified_embodiment_id}")
        print(f"bridge_loss_end_w: {bridge_loss_end_w}")
        print(f"bridge_loss_decay_steps: {bridge_loss_decay_steps}")
        print(f"use_separate_projector_for_loss: {use_separate_projector_for_loss}")
        print(f"vlm_small_lr: {vlm_small_lr}")
        print(f"matching_coeff: {matching_coeff}")

        select_layer = kwargs.pop("select_layer", None)
        select_layer_for_bridge = kwargs.pop("select_layer_for_bridge", None)

        tune_bridge_visual = kwargs.pop("tune_bridge_visual", model_config.bridge_cfg['tune_bridge_visual'])
        tune_bridge_goal = kwargs.pop("tune_bridge_goal", model_config.bridge_cfg['tune_bridge_goal'])
        assert tune_bridge_visual == False and tune_bridge_goal == False
        tune_image_type_embedding = kwargs.pop("tune_image_type_embedding", model_config.bridge_cfg.get('tune_image_type_embedding', True))
        print(f"Tune bridge vision model: {tune_bridge_visual}")
        print(f"Tune bridge goal model: {tune_bridge_goal}")
        print(f"Tune image type embeddings: {tune_image_type_embedding}")

        use_vl_mask = kwargs.pop("use_vl_mask", model_config.action_head_cfg.get('use_vl_mask', False))
        print(f"Use VL mask: {use_vl_mask}")

        correct_vl_mask = kwargs.pop("correct_vl_mask", model_config.action_head_cfg.get('correct_vl_mask', False))
        print(f"Correct VL mask: {correct_vl_mask}")

        resume_vlm_path = kwargs.pop("resume_vlm_path", model_config.bridge_cfg.get('resume_vlm_path', None))
        resume_action_head_path = kwargs.pop("resume_action_head_path", model_config.bridge_cfg.get('resume_action_head_path', None))
        print(f"resume_vlm_path: {resume_vlm_path}")
        print(f"resume_action_head_path: {resume_action_head_path}")

        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        customized_kwargs = {
            "tokenizer_len": tokenizer_len,
            "bridge_type": bridge_type,
            "compute_bridge_loss": compute_bridge_loss,
            "select_layer": select_layer,
            "goal_image_type": goal_image_type,
            "bridge_loss_type": bridge_loss_type,
            "use_image_type_embedding": use_image_type_embedding,
            "use_vl_mask": use_vl_mask,
            "correct_vl_mask": correct_vl_mask,
            "action_only_one_obs": action_only_one_obs,
            "noise_tau": noise_tau,
            "omit_image_type_embedding_for_goal": omit_image_type_embedding_for_goal,
            "reweight_noise": reweight_noise,
            "unified_embodiment_id": unified_embodiment_id,
            "use_separate_projector_for_loss": use_separate_projector_for_loss,
            "vlm_small_lr": vlm_small_lr,
            "select_layer_for_bridge": select_layer_for_bridge,
            "matching_coeff": matching_coeff 
        }

        kwargs["output_loading_info"] = True

        try:
            pretrained_model, loading_info = super().from_pretrained(
                local_model_path, local_model_path=local_model_path, 
                **customized_kwargs,
                **kwargs
            )

            if resume_pretrained_option != "all" or resume_vlm_path or resume_action_head_path:
                assert not (resume_vlm_path and resume_action_head_path)
                # The logic here is not compatible with bridge_projector_for_loss
                if resume_vlm_path is not None:
                    resume_model, loading_info = super().from_pretrained(
                        resume_vlm_path, local_model_path=resume_vlm_path, 
                        **customized_kwargs,
                        **kwargs
                    )
                    pretrained_model.backbone = resume_model.backbone
                    pretrained_model.bridge_projector = resume_model.bridge_projector

                elif resume_action_head_path is not None:
                    resume_model, loading_info = super().from_pretrained(
                        resume_action_head_path, local_model_path=resume_action_head_path, 
                        **customized_kwargs,
                        **kwargs
                    )
                    pretrained_model.action_head = resume_model.action_head
                    if hasattr(pretrained_model, "image_type_embedding"):
                        pretrained_model.image_type_embedding = resume_model.image_type_embedding

                else:
                    config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
                    scratch_model = cls(
                        config, local_model_path=None, 
                        **customized_kwargs,
                    )
                    print(f"Resuming only {resume_pretrained_option} ...")
                    if resume_pretrained_option == "backbone+bridge_projector":
                        pretrained_model.action_head = scratch_model.action_head
                        if hasattr(pretrained_model, "image_type_embedding"):
                            pretrained_model.image_type_embedding = scratch_model.image_type_embedding
                    elif resume_pretrained_option == "action_head":
                        pretrained_model.backbone = scratch_model.backbone
                        pretrained_model.bridge_projector = scratch_model.bridge_projector
                    else:
                        raise NotImplementedError

        except Exception as e:
            print(f"Load pretrained model error!!!! {e}")
            config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
            pretrained_model = cls(
                config, local_model_path=None, 
                **customized_kwargs,
            )
            loading_info = None

        if use_separate_projector_for_loss and (loading_info is not None):
            # Check if missing keys include bridge_projector_for_loss
            is_missing = any("bridge_projector_for_loss" in k for k in loading_info["missing_keys"])
            
            if is_missing:
                print("### [Init] bridge_projector_for_loss not found in checkpoint. Copying from bridge_projector.")
                pretrained_model.bridge_projector_for_loss = deepcopy(pretrained_model.bridge_projector)
            else:
                print("### [Resume] bridge_projector_for_loss loaded from checkpoint. Skipping deepcopy.")

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm, 
            tune_bridge_embedding=tune_bridge_embedding,
            tokenizer_len=tokenizer_len,
            tune_all_llm_embedding=tune_all_llm_embedding,
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        if pretrained_model.use_bridge:
            pretrained_model.set_trainable_parameters(
                tune_bridge_visual=tune_bridge_visual,
                tune_bridge_goal=tune_bridge_goal,
                tune_image_type_embedding=tune_image_type_embedding
            )

        pretrained_model.config.backbone_cfg['tune_visual'] = tune_visual
        pretrained_model.config.backbone_cfg['tune_llm'] = tune_llm
        pretrained_model.config.backbone_cfg['tune_bridge_embedding'] = tune_bridge_embedding
        pretrained_model.config.action_head_cfg['tune_projector'] = tune_projector
        pretrained_model.config.action_head_cfg['tune_diffusion_model'] = tune_diffusion_model
        pretrained_model.config.bridge_cfg['tune_bridge_visual'] = tune_bridge_visual
        pretrained_model.config.bridge_cfg['tune_bridge_goal'] = tune_bridge_goal
        pretrained_model.config.bridge_cfg['tokenizer_len'] = tokenizer_len
        pretrained_model.config.bridge_cfg['bridge_type'] = bridge_type
        pretrained_model.config.bridge_cfg['compute_bridge_loss'] = compute_bridge_loss
        pretrained_model.config.bridge_cfg['goal_image_type'] = goal_image_type
        pretrained_model.config.bridge_cfg['bridge_loss_type'] = bridge_loss_type
        pretrained_model.config.backbone_cfg['tune_all_llm_embedding'] = tune_all_llm_embedding
        pretrained_model.config.bridge_cfg['use_image_type_embedding'] = use_image_type_embedding
        pretrained_model.config.bridge_cfg['action_only_one_obs'] = action_only_one_obs
        pretrained_model.config.bridge_cfg['noise_tau'] = noise_tau
        pretrained_model.config.bridge_cfg['reweight_noise'] = reweight_noise
        pretrained_model.config.bridge_cfg['omit_image_type_embedding_for_goal'] = omit_image_type_embedding_for_goal
        pretrained_model.config.bridge_cfg['tune_image_type_embedding'] = tune_image_type_embedding
        pretrained_model.config.bridge_cfg['unified_embodiment_id'] = unified_embodiment_id
        pretrained_model.config.bridge_cfg['bridge_loss_end_w'] = bridge_loss_end_w
        pretrained_model.config.bridge_cfg['bridge_loss_decay_steps'] = bridge_loss_decay_steps
        pretrained_model.config.bridge_cfg['use_separate_projector_for_loss'] = use_separate_projector_for_loss
        pretrained_model.config.bridge_cfg['vlm_small_lr'] = vlm_small_lr
        if matching_coeff is not None:
            if hasattr(pretrained_model, 'config'):
                if 'action_head_cfg' not in pretrained_model.config.__dict__:
                    pretrained_model.config.action_head_cfg['matching_coeff'] = matching_coeff
            if hasattr(pretrained_model, 'action_head'):
                pretrained_model.action_head.matching_coeff = matching_coeff
        
        return pretrained_model

    def set_trainable_parameters(self, tune_bridge_visual: bool, tune_bridge_goal: bool, tune_image_type_embedding: bool):
        self.tune_bridge_visual = tune_bridge_visual
        self.tune_bridge_goal = tune_bridge_goal
        self.tune_image_type_embedding = tune_image_type_embedding

        if hasattr(self, 'dino_model'):
            self.dino_model.requires_grad_(self.tune_bridge_visual)

        if hasattr(self, 'bridge_vision_model') and hasattr(self, 'bridge_goal_model'):
            if not tune_bridge_visual and not tune_bridge_goal:
                del self.bridge_goal_model
                self.bridge_goal_model = self.bridge_vision_model
                self._shared_bridge_models = True
            else:
                self._shared_bridge_models = False
                self.bridge_vision_model.requires_grad_(tune_bridge_visual)
                self.bridge_goal_model.requires_grad_(tune_bridge_goal)

        if self.use_image_type_embedding:
            self.image_type_embedding.requires_grad_(self.tune_image_type_embedding)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if hasattr(self, 'dino_model'):
                self.dino_model.requires_grad_(self.tune_bridge_visual)

            if not self.tune_bridge_visual and hasattr(self, 'bridge_vision_model'):
                self.bridge_vision_model.eval()
            if not getattr(self, '_shared_bridge_models', False):
                if not self.tune_bridge_goal and hasattr(self, 'bridge_goal_model'):
                    self.bridge_goal_model.eval()
            if self.use_image_type_embedding and not self.tune_image_type_embedding:
                self.image_type_embedding.eval()

# register
AutoConfig.register("gr00t_n1_5_dial", GR00T_N1_5_DIAL_Config)
AutoModel.register(GR00T_N1_5_DIAL_Config, GR00T_N1_5_DIAL)
