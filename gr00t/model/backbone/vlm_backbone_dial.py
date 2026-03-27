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
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_VLM_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)

from gr00t.model.transforms import BRIDGE_TOKENS

class VLMBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        vlm_path: str | None = None,
        project_to_dim: int = 1536,
        tune_bridge_embedding: bool = False,
        tokenizer_len: bool = None,
        tune_all_llm_embedding: bool = False,
        select_layer_for_bridge: int = None,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        self.vlm_model = AutoModel.from_pretrained(vlm_path, trust_remote_code=True)

        if project_to_dim is not None:
            self.vlm_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.vlm_linear = torch.nn.Identity()

        print(f"Selected LLM Layer: {select_layer}")
        if hasattr(self.vlm_model.language_model, "model"):
            while len(self.vlm_model.language_model.model.layers) > select_layer:
                self.vlm_model.language_model.model.layers.pop(-1)
        else:
            while len(self.vlm_model.language_model.layers) > select_layer:
                self.vlm_model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.select_layer_for_bridge = select_layer_for_bridge
        if select_layer_for_bridge is not None:
            print(f"Selected LLM Layer for Bridge: {select_layer_for_bridge}")
            assert select_layer_for_bridge < select_layer
        self.set_trainable_parameters(
            tune_llm, tune_visual, 
            tune_bridge_embedding, tokenizer_len,
            tune_all_llm_embedding
        )

    def set_trainable_parameters(self, 
            tune_llm: bool, 
            tune_visual: bool, 
            tune_bridge_embedding: bool,
            tokenizer_len: int = None,
            tune_all_llm_embedding: bool = False,
        ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.vlm_model.language_model.requires_grad_(False)
        if not tune_visual:
            if hasattr(self.vlm_model, "vision_model"):
                self.vlm_model.vision_model.requires_grad_(False)
                self.vlm_model.mlp1.requires_grad_(False)
            else:
                self.vlm_model.visual.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")

        if not hasattr(self, "_embed_tokens_hook_handle"):
            self._embed_tokens_hook_handle = None

        if tune_bridge_embedding and (not tune_all_llm_embedding):
            if hasattr(self.vlm_model.language_model, "model"):
                embed_tokens = self.vlm_model.language_model.model.embed_tokens
            else:
                embed_tokens = self.vlm_model.language_model.embed_tokens

            embed_tokens.weight.requires_grad = True
                
            if self._embed_tokens_hook_handle is not None:
                self._embed_tokens_hook_handle.remove()
                self._embed_tokens_hook_handle = None

            bridge_token_ids = torch.arange(
                tokenizer_len-len(BRIDGE_TOKENS), tokenizer_len,
                device=embed_tokens.weight.device
            )
            print(f"start_bridge_token_id: {bridge_token_ids[0]}, end_bridge_token_id: {bridge_token_ids[-1]}")

            self._embed_tokens_hook_mask = torch.zeros(embed_tokens.weight.shape[0], device=embed_tokens.weight.device)
            self._embed_tokens_hook_mask[bridge_token_ids] = 1.0
            self._embed_tokens_hook_mask = self._embed_tokens_hook_mask.view(-1, 1)

            def grad_hook(grad):
                if self._embed_tokens_hook_mask.device != grad.device:
                    self._embed_tokens_hook_mask = self._embed_tokens_hook_mask.to(grad.device)
                return grad * self._embed_tokens_hook_mask

            self._embed_tokens_hook_handle = embed_tokens.weight.register_hook(grad_hook)
        
        else:
            if self._embed_tokens_hook_handle is not None:
                self._embed_tokens_hook_handle.remove()
                self._embed_tokens_hook_handle = None

        print(f"Tune backbone bridge embedding: {tune_bridge_embedding}")
        print(f"Tune all llm token embeddings: {tune_all_llm_embedding}")

        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.vlm_model.language_model and not self.tune_llm:
                self.vlm_model.language_model.eval()
            if hasattr(self.vlm_model, "vision_model") and self.vlm_model.vision_model and not self.tune_visual:
                self.vlm_model.vision_model.eval()
            elif self.vlm_model.visual and not self.tune_visual:
                self.vlm_model.visual.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_vlm(self, vl_input: BatchFeature) -> BatchFeature:
        vlm_prefix = "vlm_"
        vlm_input = {
            k.removeprefix(vlm_prefix): v
            for k, v in vl_input.items()
            if k.startswith(vlm_prefix)
        }
        if "image_sizes" in vlm_input:
            del vlm_input["image_sizes"]

        vlm_output = self.vlm_model(**vlm_input, output_hidden_states=True, return_dict=True)
        vlm_features = vlm_output.hidden_states[self.select_layer]
        vlm_features = self.vlm_linear(vlm_features)

        bridge_features = None
        if self.select_layer_for_bridge is not None:
            bridge_features = self.vlm_linear(vlm_output.hidden_states[self.select_layer_for_bridge])

        return vlm_features, vlm_input["attention_mask"], bridge_features

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        vlm_embeds, vlm_mask, bridge_embeds = self.forward_vlm(vl_input)

        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=vlm_embeds.device, dtype=vlm_embeds.dtype, requires_grad=True
            )
            if hasattr(self.vlm_model, "vision_model"):
                for param in self.vlm_model.vision_model.parameters():
                    if param.requires_grad:
                        dummy_term = dummy_term + 0.0 * param.sum()
            else:
                for param in self.vlm_model.visual.parameters():
                    if param.requires_grad:
                        dummy_term = dummy_term + 0.0 * param.sum()
            vlm_embeds = vlm_embeds + dummy_term

        data = {"backbone_features": vlm_embeds, "backbone_attention_mask": vlm_mask}
        if bridge_embeds is not None:
            data["backbone_features_for_bridge"] = bridge_embeds
        return BatchFeature(data=data)
