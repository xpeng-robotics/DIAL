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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy_dial import BasePolicy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import cv2

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)



def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def save_pca_visualization(debug_features, save_path, step_idx):
    """
    debug_features: dict containing obs_feat, gt_goal_feat, pred_goal_feat, curr_rgb, goal_rgb
    save_path: path to save the image
    """
    # 1. Extract features
    obs = debug_features["obs_feat"]       # [64, D]
    gt = debug_features["gt_goal_feat"]    # [64, D]
    pred = debug_features["pred_goal_feat"] # [64, D]
    
    # Extract processed RGB images (ensure they are post-Policy-transform processed images)
    curr_rgb = debug_features["curr_rgb"]  # [H, W, 3]
    goal_rgb = debug_features["goal_rgb"]  # [H, W, 3]
    h_orig, w_orig = curr_rgb.shape[:2]

    # 2. Feature preprocessing and global PCA fitting
    combined = np.concatenate([obs, gt, pred], axis=0) # [192, D]
    
    # Standardize: eliminate magnitude differences across channels, this greatly affects visualization
    # scaler = StandardScaler()
    # combined_std = scaler.fit_transform(combined) 

    # PCA reduce to 3 dimensions (corresponding to RGB)
    pca = PCA(n_components=3)
    # combined_pca = pca.fit_transform(combined_std)
    combined_pca = pca.fit_transform(combined)

    # Normalize to 0-1 range for color mapping
    c_min = combined_pca.min(axis=0)
    c_max = combined_pca.max(axis=0)
    combined_pca = (combined_pca - c_min) / (c_max - c_min + 1e-8)

    # Split back into three PCA result groups
    obs_pca, gt_pca, pred_pca = np.split(combined_pca, 3, axis=0)

    # 3. Overlay function: interpolate 8x8 PCA results to full size and overlay on RGB
    def overlay_pca_on_rgb(rgb_img, pca_feat, alpha=0.5):
        # return rgb_img

        # Reshape to 8x8x3
        side = int(np.sqrt(pca_feat.shape[0]))
        pca_img = pca_feat.reshape(side, side, 3)
        
        # Upscale 8x8 to original image size (H, W) using interpolation
        # pca_img_resized = cv2.resize(pca_img, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        pca_img_resized = cv2.resize(pca_img, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        # Normalize original image to 0-1
        background = rgb_img.astype(float) / 255.0
        
        # Linear blend: alpha * PCA features + (1 - alpha) * original image
        # This way both feature block colors and the underlying robot arm and objects are visible
        overlaid = (alpha * pca_img_resized + (1 - alpha) * background)
        return np.clip(overlaid * 255, 0, 255).astype(np.uint8)

    # 4. Generate three overlaid images
    # Note: pred_goal should be overlaid on goal_rgb since it's a prediction of the future
    obs_overlaid = overlay_pca_on_rgb(curr_rgb, obs_pca, alpha=0.9)
    gt_overlaid = overlay_pca_on_rgb(goal_rgb, gt_pca, alpha=0.9)
    pred_overlaid = overlay_pca_on_rgb(goal_rgb, pred_pca, alpha=0.9)

    # 5. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(obs_overlaid)
    axes[0].set_title(f"Step {step_idx}: Obs PCA on Current")
    
    axes[1].imshow(gt_overlaid)
    axes[1].set_title("GT Subgoal PCA on Goal")
    
    axes[2].imshow(pred_overlaid)
    axes[2].set_title("Pred Subgoal PCA on Goal")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def save_pca_visualization_overall(debug_features, save_path, step_idx, pca_model=None, global_min=None, global_max=None):
    obs = debug_features["obs_feat"]
    gt = debug_features["gt_goal_feat"]
    pred = debug_features["pred_goal_feat"]
    curr_rgb = debug_features["curr_rgb"]
    goal_rgb = debug_features["goal_rgb"]
    h_orig, w_orig = curr_rgb.shape[:2]

    # Fit PCA on obs + gt only (exclude pred to avoid biasing the basis)
    combined = np.concatenate([obs, gt, pred], axis=0)
    if pca_model is not None:
        combined_pca = pca_model.transform(combined)
        c_min, c_max = global_min, global_max
    else:
        fit_data = np.concatenate([obs, gt], axis=0)
        pca = PCA(n_components=3).fit(fit_data)
        combined_pca = pca.transform(combined)
        c_min, c_max = combined_pca.min(axis=0), combined_pca.max(axis=0)

    combined_pca = (combined_pca - c_min) / (c_max - c_min + 1e-8)
    obs_pca, gt_pca, pred_pca = np.split(combined_pca, 3, axis=0)

    def overlay_pca_on_rgb(rgb_img, pca_feat, alpha=0.9):
        side = int(np.sqrt(pca_feat.shape[0]))
        pca_img = pca_feat.reshape(side, side, 3)
        pca_img_resized = cv2.resize(pca_img, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        background = rgb_img.astype(float) / 255.0
        overlaid = (alpha * pca_img_resized + (1 - alpha) * background)
        return np.clip(overlaid * 255, 0, 255).astype(np.uint8)

    def compute_cos_diff_heatmap(feat_a, feat_b, rgb_bg, alpha=0.4):
        """Per-token cosine distance, rendered as jet colormap overlaid on background image."""
        norm_a = feat_a / (np.linalg.norm(feat_a, axis=1, keepdims=True) + 1e-8)
        norm_b = feat_b / (np.linalg.norm(feat_b, axis=1, keepdims=True) + 1e-8)
        cos_dist = 1.0 - (norm_a * norm_b).sum(axis=1)
        p_low, p_high = np.percentile(cos_dist, 5), np.percentile(cos_dist, 95)
        diff_norm = np.clip((cos_dist - p_low) / (p_high - p_low + 1e-8), 0, 1)
        diff_norm = diff_norm ** 2.0
        side = int(np.sqrt(len(diff_norm)))
        diff_map = cv2.resize(diff_norm.reshape(side, side).astype(np.float32),
                              (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        colormap = plt.cm.jet(diff_map)[:, :, :3]
        background = rgb_bg.astype(float) / 255.0
        overlaid = alpha * colormap + (1 - alpha) * background
        return np.clip(overlaid * 255, 0, 255).astype(np.uint8)

    def compute_pca_diff_heatmap(pca_a, pca_b, rgb_bg, alpha=0.4):
        """PCA-space per-token L2 distance, rendered as plasma colormap overlaid on background image."""
        l2_dist = np.linalg.norm(pca_a - pca_b, axis=1)
        p_low, p_high = np.percentile(l2_dist, 5), np.percentile(l2_dist, 95)
        diff_norm = np.clip((l2_dist - p_low) / (p_high - p_low + 1e-8), 0, 1)
        diff_norm = diff_norm ** 2.0
        side = int(np.sqrt(len(diff_norm)))
        diff_map = cv2.resize(diff_norm.reshape(side, side).astype(np.float32),
                              (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        colormap = plt.cm.plasma(diff_map)[:, :, :3]
        background = rgb_bg.astype(float) / 255.0
        overlaid = alpha * colormap + (1 - alpha) * background
        return np.clip(overlaid * 255, 0, 255).astype(np.uint8)

    obs_overlaid = overlay_pca_on_rgb(curr_rgb, obs_pca)
    gt_overlaid = overlay_pca_on_rgb(goal_rgb, gt_pca)
    pred_overlaid = overlay_pca_on_rgb(goal_rgb, pred_pca)

    diff_pred_obs = compute_cos_diff_heatmap(pred, obs, curr_rgb)
    diff_gt_obs   = compute_cos_diff_heatmap(gt,   obs, curr_rgb)
    diff_pred_gt  = compute_cos_diff_heatmap(pred, gt,  goal_rgb)

    pca_diff_pred_obs = compute_pca_diff_heatmap(pred_pca, obs_pca, curr_rgb)
    pca_diff_gt_obs   = compute_pca_diff_heatmap(gt_pca,   obs_pca, curr_rgb)
    pca_diff_pred_gt  = compute_pca_diff_heatmap(pred_pca, gt_pca,  goal_rgb)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    axes[0, 0].imshow(obs_overlaid);  axes[0, 0].set_title(f"Step {step_idx}: Policy Obs PCA")
    axes[0, 1].imshow(gt_overlaid);   axes[0, 1].set_title("GT Subgoal PCA (Bridge)")
    axes[0, 2].imshow(pred_overlaid); axes[0, 2].set_title("Pred Subgoal PCA (Bridge)")

    axes[1, 0].imshow(diff_pred_obs); axes[1, 0].set_title("Cos-Dist: Pred vs Obs\n(model thinks what changes)")
    axes[1, 1].imshow(diff_gt_obs);   axes[1, 1].set_title("Cos-Dist: GT vs Obs\n(ground-truth change)")
    axes[1, 2].imshow(diff_pred_gt);  axes[1, 2].set_title("Cos-Dist: Pred vs GT\n(prediction error)")

    axes[2, 0].imshow(pca_diff_pred_obs); axes[2, 0].set_title("PCA-L2: Pred vs Obs\n(model thinks what changes)")
    axes[2, 1].imshow(pca_diff_gt_obs);   axes[2, 1].set_title("PCA-L2: GT vs Obs\n(ground-truth change)")
    axes[2, 2].imshow(pca_diff_pred_gt);  axes[2, 2].set_title("PCA-L2: Pred vs GT\n(prediction error)")

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    plot_state=False,
    save_plot_path=None,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    bridge_loss_across_times = []
    action_dim_names = []

    for step_count in range(steps):
        data_point = None
        if plot_state:
            data_point = dataset.get_step_data(traj_id, step_count)
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
            state_joints_across_time.append(concat_state)

        if step_count % action_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data(traj_id, step_count)

            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action_and_bridgeloss(data_point)
            bridge_loss = action_chunk.get('bridge_loss', None)
            if bridge_loss is not None:
                bridge_loss_across_times.append(bridge_loss)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                # print(f"action_chunk.keys: {action_chunk.keys()}")
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

                concat_gt_action = np.concatenate(
                    [data_point[f"action.{key}"][j] for key in modality_keys], axis=0
                )
                gt_action_across_time.append(concat_gt_action)

                if len(action_dim_names) == 0:
                    for key in modality_keys:
                        for k in range(len(action_chunk[f"action.{key}"][0])):
                            action_dim_names.append(f"{key}.{k}")

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    action_mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", action_mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    if len(bridge_loss_across_times) > 0:
        bridge_loss_across_times = np.array(bridge_loss_across_times)
        print("bridge_loss_across_times vs time", bridge_loss_across_times.shape)
        # print(bridge_loss_across_times)
        bridge_loss = np.mean(bridge_loss_across_times)
    else:
        bridge_loss = None
        

    # raise error when pred action has NaN
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Pred action has NaN")

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot or save_plot_path is not None:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "bridge_loss_across_times": bridge_loss_across_times,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "action_mse": action_mse,
            "bridge_loss": bridge_loss,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
            "action_dim_names": action_dim_names,
        }
        plot_trajectory(info, save_plot_path)

    return action_mse, bridge_loss


def calc_mse_for_single_trajectory_with_pca(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    plot_state=False,
    save_plot_path=None,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    bridge_loss_across_times = []
    action_dim_names = []

    # Create feature save directory near the save path
    pca_dir = None
    if save_plot_path:
        pca_dir = os.path.join(os.path.dirname(save_plot_path), "pca_vis")
        os.makedirs(pca_dir, exist_ok=True)

    for step_count in range(steps):
        data_point = None
        if plot_state:
            data_point = dataset.get_step_data(traj_id, step_count)
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
            state_joints_across_time.append(concat_state)

        if step_count % action_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data(traj_id, step_count)

            print("inferencing at step: ", step_count)
            # Call our new function with debug features
            action_chunk, debug_features = policy.get_action_and_debug_features(data_point)
            
            # Save PCA
            if pca_dir:
                pca_save_path = os.path.join(pca_dir, f"traj{traj_id}_step{step_count}_pca.png")
                save_pca_visualization(debug_features, pca_save_path, step_count)

            # action_chunk = policy.get_action_and_bridgeloss(data_point)
            bridge_loss = action_chunk.get('bridge_loss', None)
            if bridge_loss is not None:
                bridge_loss_across_times.append(bridge_loss)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                # print(f"action_chunk.keys: {action_chunk.keys()}")
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

                concat_gt_action = np.concatenate(
                    [data_point[f"action.{key}"][j] for key in modality_keys], axis=0
                )
                gt_action_across_time.append(concat_gt_action)

                if len(action_dim_names) == 0:
                    for key in modality_keys:
                        for k in range(len(action_chunk[f"action.{key}"][0])):
                            action_dim_names.append(f"{key}.{k}")

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    action_mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", action_mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    if len(bridge_loss_across_times) > 0:
        bridge_loss_across_times = np.array(bridge_loss_across_times)
        print("bridge_loss_across_times vs time", bridge_loss_across_times.shape)
        # print(bridge_loss_across_times)
        bridge_loss = np.mean(bridge_loss_across_times)
    else:
        bridge_loss = None
        

    # raise error when pred action has NaN
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Pred action has NaN")

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot or save_plot_path is not None:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "bridge_loss_across_times": bridge_loss_across_times,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "action_mse": action_mse,
            "bridge_loss": bridge_loss,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
            "action_dim_names": action_dim_names,
        }
        plot_trajectory(info, save_plot_path)

    return action_mse, bridge_loss


def calc_mse_for_single_trajectory_with_pca_overall(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    plot_state=False,
    save_plot_path=None,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    bridge_loss_across_times = []
    action_dim_names = []

    all_debug_features = []
    all_feats_for_pca = []


    # Create feature save directory near the save path
    pca_dir = None
    if save_plot_path:
        pca_dir = os.path.join(os.path.dirname(save_plot_path), "pca_vis")
        os.makedirs(pca_dir, exist_ok=True)

    for step_count in range(steps):
        data_point = None
        if plot_state:
            data_point = dataset.get_step_data(traj_id, step_count)
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
            state_joints_across_time.append(concat_state)

        if step_count % action_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data(traj_id, step_count)

            print("inferencing at step: ", step_count)
            # Call our new function with debug features
            action_chunk, debug_features = policy.get_action_and_debug_features(data_point)
            all_debug_features.append((step_count, debug_features))
            all_feats_for_pca.append(np.concatenate([
                debug_features["obs_feat"], 
                debug_features["gt_goal_feat"],
            ], axis=0))

            # action_chunk = policy.get_action_and_bridgeloss(data_point)
            bridge_loss = action_chunk.get('bridge_loss', None)
            if bridge_loss is not None:
                bridge_loss_across_times.append(bridge_loss)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                # print(f"action_chunk.keys: {action_chunk.keys()}")
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

                concat_gt_action = np.concatenate(
                    [data_point[f"action.{key}"][j] for key in modality_keys], axis=0
                )
                gt_action_across_time.append(concat_gt_action)

                if len(action_dim_names) == 0:
                    for key in modality_keys:
                        for k in range(len(action_chunk[f"action.{key}"][0])):
                            action_dim_names.append(f"{key}.{k}")

    # 2. After the entire trajectory is done, compute global PCA
    if save_plot_path and all_feats_for_pca:
        pca_dir = os.path.join(os.path.dirname(save_plot_path), "pca_vis")
        os.makedirs(pca_dir, exist_ok=True)
        
        print(f"Calculating global PCA for trajectory {traj_id}...")
        full_traj_feats = np.concatenate(all_feats_for_pca, axis=0) # [Steps*192, D]
        pca_model = PCA(n_components=3).fit(full_traj_feats)
        
        # Compute global min/max for color normalization
        transformed = pca_model.transform(full_traj_feats)
        g_min, g_max = transformed.min(axis=0), transformed.max(axis=0)

        # 3. Unified rendering and saving
        for step_idx, feat_dict in all_debug_features:
            pca_file_name = os.path.basename(save_plot_path).replace(".png", f"_{step_idx}_pca.png")
            pca_save_path = os.path.join(pca_dir, pca_file_name)
            save_pca_visualization_overall(feat_dict, pca_save_path, step_idx, pca_model, g_min, g_max)


    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    action_mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", action_mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    if len(bridge_loss_across_times) > 0:
        bridge_loss_across_times = np.array(bridge_loss_across_times)
        print("bridge_loss_across_times vs time", bridge_loss_across_times.shape)
        # print(bridge_loss_across_times)
        bridge_loss = np.mean(bridge_loss_across_times)
    else:
        bridge_loss = None
        

    # raise error when pred action has NaN
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Pred action has NaN")

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot or save_plot_path is not None:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "bridge_loss_across_times": bridge_loss_across_times,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "action_mse": action_mse,
            "bridge_loss": bridge_loss,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
            "action_dim_names": action_dim_names,
        }
        plot_trajectory(info, save_plot_path)

    return action_mse, bridge_loss



def plot_trajectory(
    info,
    save_plot_path=None,
):
    """Simple plot of the trajectory with state, gt action, and pred action."""

    # Use non interactive backend for matplotlib if headless
    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = info["action_dim"]
    state_joints_across_time = info["state_joints_across_time"]
    gt_action_across_time = info["gt_action_across_time"]
    pred_action_across_time = info["pred_action_across_time"]
    bridge_loss_across_times = info["bridge_loss_across_times"]
    modality_keys = info["modality_keys"]
    traj_id = info["traj_id"]
    action_mse = info["action_mse"]
    bridge_loss = info["bridge_loss"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]

    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string[:-2]}\nUnnormalized MSE: {action_mse:.6f}"
    if bridge_loss is not None:
        title_text += f"\nBridge Loss: {bridge_loss:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every ACTION_HORIZON
        for k, j in enumerate(range(0, steps, action_horizon)):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

            if len(bridge_loss_across_times) > 0:
                if j == 0:
                    ax.plot(j, bridge_loss_across_times[k], "ko", label="bridge loss", markersize=6)
                else:
                    ax.plot(j, bridge_loss_across_times[k], "ko", markersize=4)

        ax.set_title(f"{info['action_dim_names'][i]}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
