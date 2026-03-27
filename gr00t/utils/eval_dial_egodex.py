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

import pandas as pd
from copy import deepcopy
import colorsys

import cv2
import subprocess
from PIL import Image as PILImage
from gr00t.utils.video import get_all_frames


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
    save_vis_2d_path=None,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    bridge_loss_across_times = []
    action_dim_names = []
    modality_key2dim = {}

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
                    start = 0
                    for key in modality_keys:
                        for k in range(len(action_chunk[f"action.{key}"][0])):
                            action_dim_names.append(f"{key}.{k}")
                        end = start+len(action_chunk[f"action.{key}"][0])
                        modality_key2dim[key] = {
                            "start": start,
                            "end": end
                        }
                        start = end

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


    if save_vis_2d_path is not None:
        visualize_2d(
            dataset=dataset,
            trajectory_id=traj_id,
            modality_key2dim=modality_key2dim,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            action_horizon=action_horizon,
            save_vis_2d_path=save_vis_2d_path,
        )
                


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
    """Plot the trajectory with state, gt action, and pred action (6 plots per row, one shared legend)."""

    # Use non-interactive backend for matplotlib if headless
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

    # Layout: 6 plots per row
    ncols = 6
    nrows = (action_dim + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False
    )
    plt.subplots_adjust(top=0.88, left=0.06, right=0.98, hspace=0.4, wspace=0.3)

    print("Creating visualization...")

    # Prepare title
    modality_string = ""
    # for key in modality_keys:
    #     modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    for i in range(0, len(modality_keys), 5):
        modality_string += ", ".join(modality_keys[i:i+5]) + "\n "

    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string[:-2]}\nUnnormalized MSE: {action_mse:.6f}"
    if bridge_loss is not None:
        title_text += f"\nBridge Loss: {bridge_loss:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.97)

    # Keep one legend for all
    handles, labels = None, None

    # Plot each action dim
    for i in range(action_dim):
        ax = axes[i // ncols][i % ncols]

        # Plot curves
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # Dots per horizon
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

        ax.set_title(f"{info['action_dim_names'][i]}", fontsize=11, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)

        # Capture legend once
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # Hide unused subplots
    for j in range(action_dim, nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    # Add shared legend once (below or beside the title)
    if handles and labels:
        # fig.legend(
        #     handles,
        #     labels,
        #     loc="upper right",
        #     bbox_to_anchor=(0.99, 0.97),
        #     fontsize=9,
        #     framealpha=0.9,
        # )

        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.95),
            fontsize=15,
            framealpha=0.9,
            labelspacing=1.0,
            handletextpad=1.2,
        )
    if save_plot_path:
        print(f"Saving plot to {save_plot_path}")
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()





def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert a batch of 6D rotation representations to a batch of 3x3 rotation matrices.

    Args:
        rot6d (np.ndarray): 6D rotation vectors with shape (..., 6).
                           The first 3 elements are the first column of the rotation matrix,
                           and the last 3 are the second column.

    Returns:
        np.ndarray: Rotation matrices with shape (..., 3, 3).
    """
    # Reshape the input into two columns (a1, a2)
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]

    # Gram-Schmidt orthogonalization process
    # b1 is the unit vector of a1
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    
    # b2 is the unit vector of a2 after removing its projection onto b1
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)

    # b3 is the cross product of b1 and b2, ensuring a right-handed coordinate system
    b3 = np.cross(b1, b2, axis=-1)

    # Stack the three orthogonal basis vectors into a 3x3 matrix
    # np.stack stacks along the last axis, so we need to transpose to match the standard column vector form
    return np.stack((b1, b2, b3), axis=-1)




def get_camera_extrinsics_for_trajectory(
    dataset, 
    trajectory_id
) -> np.ndarray:
    """
    Retrieve the 4x4 camera extrinsic transformation matrices for all steps of a given trajectory_id.

    Args:
        dataset (LeRobotSingleDataset): An instantiated LeRobotSingleDataset object.
        trajectory_id (int): The trajectory ID for which you want to retrieve data.

    Returns:
        np.ndarray: Array with shape (num_steps, 4, 4) containing camera extrinsic matrices for all steps.
    """
    # 1. Load trajectory data
    trajectory_data = dataset.get_trajectory_data(trajectory_id)

    # 2. Get modality metadata
    modality_meta = dataset.lerobot_modality_meta.state
    camera_pos_meta = modality_meta['camera_pos']
    camera_rot_meta = modality_meta['camera_rot']
    
    original_key = camera_pos_meta.original_key
    assert original_key == camera_rot_meta.original_key, \
        "original_key of camera_pos and camera_rot do not match!"

    # 3. Extract "observation.state" and stack into a NumPy array
    all_states = np.stack(trajectory_data[original_key].to_numpy())

    # 4. Extract camera_pos (3D) and camera_rot (6D)
    pos_start, pos_end = camera_pos_meta.start, camera_pos_meta.end
    rot_start, rot_end = camera_rot_meta.start, camera_rot_meta.end
    
    camera_pos_all_steps = all_states[:, pos_start:pos_end]
    camera_rot_all_steps = all_states[:, rot_start:rot_end]

    # 5. Convert 6D rotation to 3x3 rotation matrices
    rotation_matrices = rot6d_to_matrix(camera_rot_all_steps)

    # 6. Build 4x4 transformation matrices
    num_steps = rotation_matrices.shape[0]
    # Initialize a batch of 4x4 identity matrices
    transform_matrices = np.zeros((num_steps, 4, 4))
    transform_matrices[:, 3, 3] = 1.0
    
    # Fill in the rotation part (R)
    transform_matrices[:, :3, :3] = rotation_matrices
    
    # Fill in the translation part (t)
    transform_matrices[:, :3, 3] = camera_pos_all_steps

    return transform_matrices # (num_steps, 4, 4) 


def convert_points_to_camera_frame_batch(world_points_batch, cam_ext_batch):
    """
    Transform a batch of 3D point sets from world coordinates to camera coordinates
    using the corresponding batch of camera extrinsics.
    This operation is fully vectorized with no for-loops.

    Args:
        world_points_batch (np.ndarray): A batch of point sets in world coordinates, shape (B, N, 3).
                                         B: batch size, N: number of points per batch item.
        cam_ext_batch (np.ndarray): A batch of camera extrinsics (camera pose in world coordinates), shape (B, 4, 4).

    Returns:
        np.ndarray: A batch of point sets in camera coordinates, shape (B, N, 3).
    """
    # --- Assert input shapes ---
    try:
        assert world_points_batch.ndim == 3 and world_points_batch.shape[-1] == 3, \
            f"Input points shape should be (B, N, 3), but got {world_points_batch.shape}"
        assert cam_ext_batch.ndim == 3 and cam_ext_batch.shape[-2:] == (4, 4), \
            f"Camera extrinsic matrix shape should be (B, 4, 4), but got {cam_ext_batch.shape}"
        assert world_points_batch.shape[0] == cam_ext_batch.shape[0], \
            f"Batch size of points and camera extrinsics must match: {world_points_batch.shape[0]} vs {cam_ext_batch.shape[0]}"
    except Exception as e:
        print(e)
        cam_ext_batch = cam_ext_batch[:world_points_batch.shape[0]]


    # --- 1. Convert (B, N, 3) point sets to (B, N, 4) homogeneous coordinates ---
    # Create a (B, N, 1) array of ones and concatenate with the original points along the last dimension.
    batch_size, num_points = world_points_batch.shape[:2]
    ones_column = np.ones((batch_size, num_points, 1))
    homogeneous_points = np.concatenate((world_points_batch, ones_column), axis=-1)

    # --- 2. Batch inversion to get world-to-camera transformation matrices ---
    # np.linalg.inv natively supports batch operations, inverting along the last two dimensions.
    world_to_cam_matrices = np.linalg.inv(cam_ext_batch)

    # --- 3. Batch-apply the transformation ---
    # To perform matrix multiplication, we need to transpose the last two dimensions of the
    # homogeneous points, changing shape from (B, N, 4) to (B, 4, N).
    # Then we can use @ (np.matmul) for batched matrix multiplication.
    # (B, 4, 4) @ (B, 4, N) -> (B, 4, N)
    points_in_cam_frame_homo_T = world_to_cam_matrices @ homogeneous_points.swapaxes(-1, -2)

    # --- 4. Transpose back to (B, N, 4) and take the first three columns to get 3D coordinates ---
    points_in_cam_frame = points_in_cam_frame_homo_T.swapaxes(-1, -2)[..., :3]

    return points_in_cam_frame

    
def convert_points_to_camera_frame(world_points, cam_ext):
    """
    Transform a set of 3D points from world coordinates to camera coordinates.

    Args:
        world_points (np.ndarray): A set of points in world coordinates, shape (N, 3).
        cam_ext (np.ndarray): Camera extrinsics (camera pose in world coordinates), shape (4, 4).

    Returns:
        np.ndarray: A set of points in camera coordinates, shape (N, 3).
    """
    # Ensure input dimensions are correct
    assert world_points.shape[1] == 3, "Input points should have shape (N, 3)"
    assert cam_ext.shape == (4, 4), "Camera extrinsic matrix should have shape (4, 4)"

    # 1. Convert (N, 3) point set to (N, 4) homogeneous coordinates
    #    by appending 1 to the last column
    num_points = world_points.shape[0]
    homogeneous_points = np.hstack((world_points, np.ones((num_points, 1))))

    # 2. Compute the world-to-camera transformation matrix (inverse of camera extrinsics)
    world_to_cam_matrix = np.linalg.inv(cam_ext)

    # 3. Apply the transformation
    #    Transpose the point set to (4, N) for matrix multiplication
    #    The result will have shape (4, N)
    points_in_cam_frame_homo = world_to_cam_matrix @ homogeneous_points.T

    # 4. Transpose back to (N, 4) and take the first three columns to get 3D coordinates
    points_in_cam_frame = points_in_cam_frame_homo.T[:, :3]

    return points_in_cam_frame


def adjust_color_brightness(rgb_color, factor):
    """
    Adjust the brightness of an RGB color.
    
    Args:
        rgb_color (tuple): Color in (R, G, B) format, values in range 0-255.
        factor (float): Brightness adjustment factor. > 1 brightens, < 1 darkens.
                        For example, 1.5 increases brightness by 50%.
                        
    Returns:
        tuple: Brightness-adjusted (R, G, B) color.
    """
    # Normalize 0-255 RGB values to 0-1
    r, g, b = [x / 255.0 for x in rgb_color]
    
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    # Adjust lightness L, clamping to [0, 1]
    l = max(0, min(1, l * factor))
    
    # Convert HLS back to RGB
    r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
    
    # Convert 0-1 RGB values back to 0-255
    return (int(r_new * 255), int(g_new * 255), int(b_new * 255))



def lighten_color(rgb_color, factor=0.5):
    """
    Lighten a color by blending it with white.
    
    Args:
        rgb_color (tuple): Color in (R, G, B) format.
        factor (float): Blending factor, 0 means original color, 1 means pure white.
                        For example, 0.5 means 50% blend with white.
                        
    Returns:
        tuple: Lightened (R, G, B) color.
    """
    white = np.array([255, 255, 255])
    original_color = np.array(rgb_color)
    
    # Linear interpolation
    new_color = original_color * (1 - factor) + white * factor
    
    return tuple(new_color.astype(int))


def map_fingers_to_comparison_colors(tf_names, pred_brightness_factor=1.6):
    """
    Generate a high-contrast color scheme for ground-truth and predicted points.
    
    Args:
        tf_names (list of str): List of transform names used to identify fingers.
        pred_brightness_factor (float): Brightness factor for predicted point colors, recommended > 1.2.
        
    Returns:
        dict: Dictionary containing two sets of colors {'gt': np.ndarray, 'pred': np.ndarray}
    """
    gt_colors = []
    pred_colors = []
    
    # base_color_map = {
    #     'little': (0, 152, 191),      # Bright Blue
    #     'ring': (173, 255, 47),     # Green Yellow
    #     'middle': (230, 245, 250),    # Pale Turquoise
    #     'index': (255, 99, 71),       # Tomato Red
    #     'thumb': (238, 130, 238)      # Violet
    # }

    base_color_map = {
        'little': (0, 100, 200),      # Deep Blue
        'ring': (50, 180, 50),       # Green
        'middle': (220, 220, 0),     # Yellow - still bright, but better than before
        'index': (200, 0, 0),        # Red
        'thumb': (150, 0, 200)       # Purple
    }

    for tf in tf_names:
        # Default color
        base_color = (200, 200, 200) # Gray
        # Match finger
        for finger, color in base_color_map.items():
            if finger in tf.lower():
                base_color = color
                break
        
        # Ground-truth uses the original color
        gt_colors.append(base_color)
        
        # Predicted uses the lightened color
        # light_color = adjust_color_brightness(base_color, pred_brightness_factor)
        light_color = lighten_color(base_color)
        pred_colors.append(light_color)

    return {
        'gt': np.array(gt_colors),
        'pred': np.array(pred_colors)
    }

def imgs_to_mp4(img_list, mp4_path, fps=30):
    """Write RGB frames to MP4 using ffmpeg pipe; fallback to GIF on failure."""
    if not img_list:
        raise ValueError("img_list is empty")
    H, W, _ = img_list[0].shape
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        mp4_path
    ]
    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for img in img_list:
            if img.dtype != np.uint8:
                frame = (np.clip(img, 0, 255)).astype(np.uint8)
            else:
                frame = img
            if frame.shape[2] == 3:
                proc.stdin.write(frame.tobytes())
            else:
                raise ValueError("Expected RGB frames with 3 channels")
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed to write MP4")
    except Exception:
        # Fallback to GIF in same folder
        base = mp4_path.rsplit('.', 1)[0]
        gif_path = base + '.gif'
        frames = [PILImage.fromarray(img) for img in img_list]
        if frames:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000/max(1, fps)), loop=0)

def draw_line(pointa, pointb, image, intrinsic, color=(0,255,0), thickness=5, point_size=15):
    # print(f"pointa: {pointa}\tpointb: {pointb}")
    # project 3d points into 2d
    pointa2, _ = cv2.projectPoints(pointa, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5))  
    pointb2, _ = cv2.projectPoints(pointb, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5))  
    pointa2 = pointa2.squeeze()
    pointb2 = pointb2.squeeze()

    # don't draw if the line is out of bounds
    H, W, _ = image.shape
    if (pointb2[0] < 0 and pointa2[0] > W) or (pointb2[1] < 0 and pointa2[1] > H) or (pointa2[0] < 0 and pointb2[0] > W) or (pointa2[1] < 0 and pointb2[1] > H):
        return 

    # draws a line in-place
    cv2.line(image, pointa2.astype(int), pointb2.astype(int), color=color, thickness=thickness)
    cv2.circle(image, (int(pointa2[0]), int(pointa2[1])), point_size, color, -1)
    cv2.circle(image, (int(pointb2[0]), int(pointb2[1])), point_size, color, -1)

def draw_line_sequence(points_list, image, intrinsic, color=(0,255,0), thickness=5, point_size=15):
    # draw a sequence of lines in-place
    ptm = points_list[0]
    for pt in points_list[1:]:
        draw_line(ptm, pt, image, intrinsic, color, thickness=thickness, point_size=point_size)
        ptm = pt


def visualize_2d(
    dataset,
    trajectory_id,
    modality_key2dim,
    gt_action_across_time,
    pred_action_across_time,
    action_horizon,
    save_vis_2d_path,
):
    # get video frames
    video_key = dataset.modality_keys['video'][0].replace("video.", "")
    video_path = dataset.get_video_path(trajectory_id, video_key)
    frames = get_all_frames(
        video_path.as_posix(),
        video_backend=dataset.video_backend,
        video_backend_kwargs=dataset.video_backend_kwargs
    ) # (N, H, W, C)
    assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
    assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"

    # Get the 4x4 camera extrinsic matrices
    camera_extrinsics = get_camera_extrinsics_for_trajectory(
        dataset, 
        trajectory_id
    )

    # Set camera intrinsics
    camera_intrinsic = np.array(
        [[736.6339,   0.    , 960.    ],
        [  0.    , 736.6339, 540.    ],
        [  0.    ,   0.    ,   1.    ]]
    )


    # Get actions for visualization
    vis_modality_key2dim = {}
    vis_gt_action_across_time = []
    vis_pred_action_across_time = []
    
    index = 0
    for modality_key, dim in modality_key2dim.items():
        if "camera" in modality_key:
            continue
        if "rot" in modality_key:
            continue

        vis_modality_key = modality_key.replace("_pos", "")
        vis_gt_action_across_time.append(gt_action_across_time[:, dim["start"]:dim["end"]])
        vis_pred_action_across_time.append(pred_action_across_time[:, dim["start"]:dim["end"]])

        # end = start + dim["end"] - dim["start"]
        vis_modality_key2dim[vis_modality_key] = index
        index += 1
        # {
        #     "start": start,
        #     "end": end
        # }
        # start = end

    print(f"modality_key2dim: {modality_key2dim}")
    print(f"vis_modality_key2dim: {vis_modality_key2dim}")
    
    vis_gt_action_across_time = np.stack(vis_gt_action_across_time, axis=1)
    vis_pred_action_across_time = np.stack(vis_pred_action_across_time, axis=1)
    print(f"vis_pred_action_across_time: {vis_pred_action_across_time.shape}")
    # camera_extrinsics = camera_extrinsics[1:]
    # assert camera_extrinsics.shape[0] == vis_gt_action_across_time.shape[0] == vis_pred_action_across_time.shape[0] == frames.shape[0]-1
        
    vis_gt_action_across_time_in_cam = convert_points_to_camera_frame_batch(vis_gt_action_across_time, camera_extrinsics)
    vis_pred_action_across_time_in_cam = convert_points_to_camera_frame_batch(vis_pred_action_across_time, camera_extrinsics) # (num_steps, num_points, 3)
    initial_frame = frames[0]
    vis_frames = frames[1:]

    left_hand_dim = vis_modality_key2dim.pop("leftHand")
    right_hand_dim = vis_modality_key2dim.pop("rightHand")

    out_imgs = [initial_frame]
    for i in range(vis_gt_action_across_time.shape[0]):
        cam_img = vis_frames[i]

        if i % action_horizon == 0:
            point_size = 30
        else:
            point_size = 15

        vis_gt_action_in_cam = vis_gt_action_across_time_in_cam[i]
        vis_pred_action_in_cam = vis_pred_action_across_time_in_cam[i]

        
        left_hand_pos_pred = vis_pred_action_in_cam[left_hand_dim]
        left_hand_pos_gt = vis_gt_action_in_cam[left_hand_dim]

        # print(f"left_hand_dim: {left_hand_dim}\tleft_hand_pos_gt: {left_hand_pos_gt}")
        # print(f"vis_gt_action_in_cam: {vis_gt_action_in_cam.shape}")

        right_hand_pos_gt = vis_gt_action_in_cam[right_hand_dim]
        right_hand_pos_pred = vis_pred_action_in_cam[right_hand_dim]


        for vis_modality_key, dim in vis_modality_key2dim.items():
            point_gt = vis_gt_action_in_cam[dim]
            point_pred = vis_pred_action_in_cam[dim]

            color = map_fingers_to_comparison_colors([vis_modality_key])
            color_gt = color["gt"][0].tolist()
            color_pred = color["pred"][0].tolist()

            if "left" in vis_modality_key:
                hand_pos_gt = left_hand_pos_gt
                hand_pos_pred = left_hand_pos_pred
            elif "right" in vis_modality_key:
                hand_pos_gt = right_hand_pos_gt
                hand_pos_pred = right_hand_pos_pred
            else:
                raise NotImplementedError

            draw_line_sequence(
                [hand_pos_gt, point_gt], 
                cam_img, camera_intrinsic,
                color=color_gt,
                point_size=point_size
            )

            draw_line_sequence(
                [hand_pos_pred, point_pred], 
                cam_img, camera_intrinsic,
                color=color_pred,
                point_size=point_size
            )

        out_imgs.append(cam_img)
    
    imgs_to_mp4(out_imgs, save_vis_2d_path)
    print('Done. Video saved to: {}'.format(save_vis_2d_path))

            


