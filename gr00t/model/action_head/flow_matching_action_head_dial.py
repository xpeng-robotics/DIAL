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

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadDIALConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    use_vl_mask: bool = field(default=False)
    correct_vl_mask: bool = field(default=False)
    shared_embodiment_id: int = None
    shared_state_dim: int = 48

    use_embodiment_token_in: str = None # [bridge, dit]

    use_matching_loss: bool = False
    matching_coeff: float = 0.1

    null_state_prob: float = field(
        default=0.0, metadata={"help": "Probability of dropping state embedding during training."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHeadDIAL(nn.Module):
    config_class = FlowmatchingActionHeadDIALConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadDIALConfig,
        backbone_hidden_size: int=None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        print(f"Action Expert: shared_embodiment_id - {config.shared_embodiment_id}, shared_state_dim - {config.shared_state_dim}")
        self.shared_embodiment_id = config.shared_embodiment_id
        self.shared_state_dim = config.shared_state_dim
        
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.use_vl_mask = config.use_vl_mask
        print(f"Action Head use VL mask: {self.use_vl_mask}")
        self.correct_vl_mask = config.correct_vl_mask
        print(f"Action Head Correct VL mask: {self.correct_vl_mask}")
        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )
        self.use_vlln = config.use_vlln

        self.use_embodiment_token_in = config.use_embodiment_token_in
        print(f"Action Head use_embodiment_token_in: {self.use_embodiment_token_in}")
        if self.use_embodiment_token_in is not None:
            assert self.use_embodiment_token_in in ['bridge', 'dit']
            if self.use_embodiment_token_in == 'bridge':
                self.embodiment_tokens = nn.Embedding(config.max_num_embodiments, backbone_hidden_size)
            else:
                self.embodiment_tokens = nn.Embedding(config.max_num_embodiments, self.input_embedding_dim)
            nn.init.normal_(self.embodiment_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

        # [New] Discriminative task components
        self.use_matching_loss = getattr(config, 'use_matching_loss', False)
        print(f"Action Head use_matching_loss: {self.use_matching_loss}")
        if self.use_matching_loss:
            # Dedicated [MATCH] token
            self.match_token = nn.Parameter(torch.randn(1, 1, self.input_embedding_dim) * 0.02)
            # Discriminator head: map DiT output features to 1-dim logit
            self.match_head = nn.Linear(self.hidden_size, 1)
            self.matching_coeff = getattr(config, 'matching_coeff', 0.1)

        # [New] Initialize null state embedding
        # Shape is typically (1, 1, input_embedding_dim) for convenient broadcasting
        if self.config.null_state_prob > 0:
            print(f"Action Head null_state_prob: {self.config.null_state_prob}")
            self.null_state_embedding = nn.Parameter(
                torch.randn(1, 1, self.input_embedding_dim) * 0.02
            )

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        # print(f"self.tune_projector: {self.tune_projector}")
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        if self.use_vlln:
            backbone_features = backbone_output["backbone_features"]
            backbone_features = self.vlln(backbone_features)
            vl_attn_mask = backbone_output.backbone_attention_mask
            if self.correct_vl_mask:
                vl_attn_mask = (1 - vl_attn_mask) * -10000
            backbone_features = self.vl_self_attention(
                hidden_states=backbone_features,
                attention_mask=vl_attn_mask.float() if self.use_vl_mask else None
            )
            backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id
        if self.use_embodiment_token_in is not None:
            # print(embodiment_id)
            # embodiment_token = self.embodiment_tokens.weight[embodiment_id.clone().detach()] + self.embodiment_tokens.weight.sum(dim=0, keepdim=True)*0.0
            embodiment_token = self.embodiment_tokens.weight[embodiment_id.clone().detach()]
            # embodiment_token = self.embodiment_tokens(embodiment_id)

        if self.use_embodiment_token_in == "bridge":
            # print(f"embodiment_token: {embodiment_token.shape}\tbackbone_features: {backbone_output['backbone_features'].shape}\tbackbone_attention_mask: {backbone_output['backbone_attention_mask'].shape}")
            backbone_output["backbone_features"] = torch.cat([embodiment_token.unsqueeze(1), backbone_output["backbone_features"]], dim=1)
            backbone_output["backbone_attention_mask"] = torch.cat([
                torch.ones(embodiment_token.shape[0], 1, device=embodiment_token.device), 
                backbone_output["backbone_attention_mask"]
            ], dim=1)

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # [Logic fix] Handle Shared ID override
        if self.unified_embodiment_id is not None:
            # Note: this forces all data to go through a single path, be careful during training
            if self.shared_embodiment_id is not None:
                if self.shared_embodiment_id != self.unified_embodiment_id:
                    # Use unified_embodiment_id to specify samples that should fully share state/action encoder/decoder with shared_embodiment_id
                    map_mask = (embodiment_id == self.unified_embodiment_id)
                    # print(f"embodiment_id: {embodiment_id}\nmap_mask: {map_mask}")
                    embodiment_id[map_mask] = self.shared_embodiment_id
                    # print(f"new_embodiment_id: {embodiment_id}")
                else:
                    raise NotImplementedError
            else:
                embodiment_id[:] = self.unified_embodiment_id

        # # Embed state.
        # if self.shared_embodiment_id is None:
        #     state_features = self.state_encoder(action_input.state, embodiment_id)
        # else:
        #     shared_embodiment_id = torch.full_like(embodiment_id, fill_value=self.shared_embodiment_id)
        #     shared_state = torch.zeros_like(action_input.state)
        #     shared_state[..., :self.shared_state_dim] = action_input.state[..., :self.shared_state_dim]
        #     shared_state_features = self.state_encoder(shared_state, shared_embodiment_id)
            
        #     is_hybrid = (embodiment_id != self.shared_embodiment_id)
        #     print(f"is_hybrid: {is_hybrid}\nembodiment_id: {embodiment_id}\nshared_embodiment_id: {self.shared_embodiment_id}")
        #     remain_state = torch.zeros_like(action_input.state)
        #     remain_state[..., self.shared_state_dim:] = action_input.state[..., self.shared_state_dim:]
        #     remain_state_features = self.state_encoder(remain_state, embodiment_id)
        #     state_features = shared_state_features + is_hybrid * remain_state_features


        

        # Embed state.
        if self.shared_embodiment_id is None:
            # Compatible with legacy logic
            state_features = self.state_encoder(action_input.state, embodiment_id)
        else:
            # 1. Construct Shared input (mask out Joint portion)
            # Assume shared_embodiment_id is a scalar, expand to Tensor
            shared_id_tensor = torch.full_like(embodiment_id, fill_value=self.shared_embodiment_id)
            
            shared_state = torch.zeros_like(action_input.state)
            # Keep only Cartesian portion (first shared_state_dim dimensions)
            shared_state[..., :self.shared_state_dim] = action_input.state[..., :self.shared_state_dim]
            
            # Encoder A: extract shared features
            shared_state_features = self.state_encoder(shared_state, shared_id_tensor)
            
            # 2. Construct Residual input (mask out Cartesian portion)
            remain_state = torch.zeros_like(action_input.state)
            # Keep only Joint portion (latter half)
            remain_state[..., self.shared_state_dim:] = action_input.state[..., self.shared_state_dim:]
            
            # Encoder B: extract robot-specific features (reusing state_encoder with Robot ID weights)
            remain_state_features = self.state_encoder(remain_state, embodiment_id)
            
            # 3. Compute Mask
            # is_hybrid: if current ID != Shared ID, it's a Robot in Execution mode
            is_hybrid = (embodiment_id != self.shared_embodiment_id).float()
            # print(f"is_hybrid: {is_hybrid}\nembodiment_id: {embodiment_id}\nshared_embodiment_id: {self.shared_embodiment_id}")
            
            # Broadcast: [B] -> [B, 1, 1] to match (B, Seq, Dim)
            while is_hybrid.dim() < shared_state_features.dim():
                is_hybrid = is_hybrid.unsqueeze(-1)

            # 4. Asymmetric fusion
            state_features = shared_state_features + is_hybrid * remain_state_features
     

        # [New] Replace with null embedding at a certain probability (only during training when prob > 0)
        if self.training and self.config.null_state_prob > 0:
            # Generate mask: (B, 1, 1)
            B = state_features.shape[0]
            mask = (torch.rand(B, 1, 1, device=state_features.device) < self.config.null_state_prob).to(state_features.dtype)
            # If mask is 1, fill with null_state_embedding
            # Use expand_as to ensure shape match (B, Seq_len, Dim)
            null_feat = self.null_state_embedding.expand_as(state_features)
            state_features = (1.0 - mask) * state_features + mask * null_feat


        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        # [New] Inject [MATCH] token
        if self.use_matching_loss:
            m_token = self.match_token.expand(sa_embs.shape[0], -1, -1)
            sa_embs = torch.cat([m_token, sa_embs], dim=1)


        if self.use_embodiment_token_in == "dit":
            sa_embs = torch.cat([embodiment_token.unsqueeze(1), sa_embs], dim=1)
            # print(f"embodiment_token: {embodiment_token.shape}\sa_embs: {sa_embs.shape}\t")
            

        vl_attn_mask = backbone_output.backbone_attention_mask
        if self.correct_vl_mask:
            vl_attn_mask = (1 - vl_attn_mask) * -10000

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask.float() if self.use_vl_mask else None,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )


        output_dict = {}

        # [New] Compute discriminative loss (BCE)
        if self.training and self.use_matching_loss:
            match_logits = self.match_head(model_output[:, 0]) # take the 0th token
            match_labels = action_input.match_labels
            match_loss = F.binary_cross_entropy_with_logits(match_logits, match_labels)
            output_dict["match_loss"] = match_loss

        # Compute Action loss
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask

        # loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        # loss = loss.sum() / action_mask.sum()
        # output_dict = {
        #     "loss": loss,
        # }

        mse_loss_raw = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask

        # [New] Hybrid loss strategy: negative samples do not contribute MSE gradients
        if self.training and self.use_matching_loss:
            pos_mask = match_labels.unsqueeze(-1) # [2B, 1, 1]
            # Only compute MSE for positive samples
            loss_action = (mse_loss_raw * pos_mask).sum() / (action_mask * pos_mask).sum().clamp(min=1.0)
            total_loss = loss_action + self.matching_coeff * match_loss
        else:
            loss_action = mse_loss_raw.sum() / action_mask.sum()
            total_loss = loss_action

        output_dict["loss"] = total_loss
        output_dict["action_loss"] = loss_action

        return BatchFeature(data=output_dict)

    # @torch.no_grad()
    # def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
    #     backbone_output = self.process_backbone_output(backbone_output)

    #     # Get vision and language embeddings.
    #     vl_embs = backbone_output.backbone_features
    #     embodiment_id = action_input.embodiment_id

    #     # Embed state.
    #     state_features = self.state_encoder(action_input.state, embodiment_id)

    #     # Set initial actions as the sampled noise.
    #     batch_size = vl_embs.shape[0]
    #     device = vl_embs.device
    #     actions = torch.randn(
    #         size=(batch_size, self.config.action_horizon, self.config.action_dim),
    #         dtype=vl_embs.dtype,
    #         device=device,
    #     )

    #     num_steps = self.num_inference_timesteps
    #     dt = 1.0 / num_steps

    #     # Run denoising steps.
    #     for t in range(num_steps):
    #         t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
    #         t_discretized = int(t_cont * self.num_timestep_buckets)

    #         # Embed noised action trajectory.
    #         timesteps_tensor = torch.full(
    #             size=(batch_size,), fill_value=t_discretized, device=device
    #         )
    #         action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
    #         # Maybe add position embedding.
    #         if self.config.add_pos_embed:
    #             pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
    #             pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
    #             action_features = action_features + pos_embs

    #         # Join vision, language, state and action embedding along sequence dimension.
    #         future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
    #         sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

    #         # Run model forward.
    #         model_output = self.model(
    #             hidden_states=sa_embs,
    #             encoder_hidden_states=vl_embs,
    #             timestep=timesteps_tensor,
    #         )
    #         pred = self.action_decoder(model_output, embodiment_id)

    #         pred_velocity = pred[:, -self.action_horizon :]

    #         # Update actions using euler integration.
    #         actions = actions + dt * pred_velocity
    #     return BatchFeature(data={"action_pred": actions})


    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature, guidance_scale: float = 1.0) -> BatchFeature:
        # print(f"guidance_scale: {guidance_scale}")
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings
        vl_embs = backbone_output.backbone_features  # If CFG, batch_size here should be 2*N
        embodiment_id = action_input.embodiment_id
        device = vl_embs.device
        
        # Check if CFG inference is enabled: guidance_scale > 1 and backbone output has double Batch (Cond + Uncond)
        do_cfg = guidance_scale > 1.0 and vl_embs.shape[0] == 2 * action_input.state.shape[0]

        # State encoding
        state_features = self.state_encoder(action_input.state, embodiment_id)
        
        # If CFG, duplicate state_features and embodiment_id to match backbone's double Batch
        if do_cfg:
            state_features = torch.cat([state_features, state_features], dim=0)
            cur_embodiment_id = torch.cat([embodiment_id, embodiment_id], dim=0)
        else:
            cur_embodiment_id = embodiment_id

        # Initialize action trajectory as Gaussian noise (keep single Batch, since final output is one copy)
        batch_size = action_input.state.shape[0]
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Iterative denoising
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Prepare inputs for current step
            if do_cfg:
                # Duplicate actions for parallel dual-path inference in the model
                input_actions = torch.cat([actions, actions], dim=0)
                timesteps_tensor = torch.full(size=(batch_size * 2,), fill_value=t_discretized, device=device)
            else:
                input_actions = actions
                timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

            # Action encoding
            action_features = self.action_encoder(input_actions, timesteps_tensor, cur_embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Concatenate all features
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(action_features.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Model forward inference
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, cur_embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            # CFG merge logic: v = v_uncond + scale * (v_cond - v_uncond)
            if do_cfg:
                # Assuming backbone_output order is [Cond, Uncond]
                v_cond, v_uncond = pred_velocity.chunk(2)
                pred_velocity = v_uncond + guidance_scale * (v_cond - v_uncond)

            # Euler integration to update actions
            actions = actions + dt * pred_velocity
            
        return BatchFeature(data={"action_pred": actions})
    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
