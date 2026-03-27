import argparse
import cv2
import json
import os
import time
import pandas as pd
from pathlib import Path

import h5py
import imageio
import numpy as np
import robocasa
import robosuite
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from termcolor import colored
from tqdm import tqdm
import xml.etree.ElementTree as ET

import pdb

pdb.set_trace = lambda: None

# Define joint points to track
# Structure: "joint_name": ("Mujoco element type", "element name", "visualization color (BGR)")
# Type can be 'body' or 'site'
JOINTS_TO_TRACK = {
    # Left hand
    "wrist_l": ("body", "gripper0_left_l_palm", (255, 0, 0)),          # Blue
    "thumb_l": ("body", "gripper0_left_L_thumb_distal_link", (0, 255, 0)), # Green
    "index_l": ("body", "gripper0_left_L_index_intermediate_link", (0, 0, 255)),  # Red
    "middle_l": ("body", "gripper0_left_L_middle_intermediate_link", (0, 255, 255)), # Yellow
    "ring_l": ("body", "gripper0_left_L_ring_intermediate_link", (255, 0, 255)), # Magenta
    "pinky_l": ("body", "gripper0_left_L_pinky_intermediate_link", (255, 255, 0)), # Cyan
    # Right hand
    "wrist_r": ("body", "gripper0_right_r_palm", (255, 0, 0)),         # Blue
    "thumb_r": ("body", "gripper0_right_R_thumb_distal_link", (0, 255, 0)), # Green
    "index_r": ("body", "gripper0_right_R_index_intermediate_link", (0, 0, 255)),  # Red
    "middle_r": ("body", "gripper0_right_R_middle_intermediate_link", (0, 255, 255)),# Yellow
    "ring_r": ("body", "gripper0_right_R_ring_intermediate_link", (255, 0, 255)), # Magenta
    "pinky_r": ("body", "gripper0_right_R_pinky_intermediate_link", (255, 255, 0)), # Cyan
}


def remove_dicts_with_keyword(data, keyword="book"):
    """
    Recursively traverse the data structure and remove dicts containing the specified keyword.
    Primarily filters dict elements within lists.
    """
    if isinstance(data, dict):
        # If it's a dict, recursively process each key-value pair
        for key, value in data.items():
            data[key] = remove_dicts_with_keyword(value, keyword)
        return data
    
    elif isinstance(data, list):
        # If it's a list, this is where we primarily filter
        new_list = []
        for item in data:
            # Check if item is a dict
            if isinstance(item, dict):
                # Convert dict to string to check for keyword
                # Use json.dumps to ensure all nested levels are converted to strings for searching
                item_str = json.dumps(item)
                if keyword in item_str:
                    # If it contains "book", skip this item (i.e. remove it)
                    # print(f"Removed an object containing '{keyword}'...") 
                    continue
            
            # If keyword not found, or not a dict, recursively process its internals (for nesting) and keep it
            new_list.append(remove_dicts_with_keyword(item, keyword))
        return new_list
    
    else:
        # Other primitive types are returned directly
        return data

def remove_book_nodes(xml_string):
    # Parse XML string
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        return None

    # 1. Clean up Asset section
    assets = root.find('asset')
    if assets is not None:
        nodes_to_remove = []
        for node in assets:
            # Get all attributes that may contain the keyword
            name = node.get('name', '').lower()
            file_path = node.get('file', '').lower()
            # Key fix: get the texture attribute referenced by material nodes
            texture_ref = node.get('texture', '').lower()
            
            # Condition: name, file path, or referenced texture name contains 'book'
            if ('book' in name) or ('book' in file_path) or ('book' in texture_ref):
                nodes_to_remove.append(node)
        
        # Perform removal
        for node in nodes_to_remove:
            assets.remove(node)
            # print(f"Removed Asset node [{node.tag}]: name='{node.get('name')}' texture='{node.get('texture')}'")

    # 2. Clean up Worldbody section
    worldbody = root.find('worldbody')
    if worldbody is not None:
        bodies_to_remove = []
        
        # Iterate over all direct child bodies under worldbody
        for body in worldbody.findall('body'):
            should_remove = False
            body_name = body.get('name', '').lower()
            
            # A. If body name directly contains 'book'
            if 'book' in body_name:
                should_remove = True
            
            # B. Check if geoms inside the body reference a mesh or material containing 'book'
            else:
                for geom in body.findall('geom'):
                    mesh_name = geom.get('mesh', '').lower()
                    material_name = geom.get('material', '').lower() # Also check material reference
                    if ('book' in mesh_name) or ('book' in material_name):
                        should_remove = True
                        break
            
            if should_remove:
                bodies_to_remove.append(body)

        # Perform removal
        for body in bodies_to_remove:
            worldbody.remove(body)
            # print(f"Removed Worldbody node: body name='{body.get('name')}'")

    return ET.tostring(root, encoding='unicode')


def process_img_cotrain(img):
    assert img.shape[0] == 800 and img.shape[1] == 1280

    oh, ow = 256, 256
    crop = (310, 770, 110, 1130)
    img = img[crop[0] : crop[1], crop[2] : crop[3]]

    img_resized = cv2.resize(img, (720, 480), cv2.INTER_AREA)
    width_pad = (img_resized.shape[1] - img_resized.shape[0]) // 2
    img_pad = np.pad(
        img_resized,
        ((width_pad, width_pad), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    img_resized = cv2.resize(img_pad, (oh, ow), cv2.INTER_AREA)
    return img_resized

def project_world_to_pixel(point_3d, camera_pos, camera_mat, fovy, height, width):
    """
    Project a 3D world coordinate point to 2D pixel coordinates.
    """
    point_cam = camera_mat.T @ (point_3d - camera_pos)
    if point_cam[2] >= 0:
        return None
    proj_h = np.tan(fovy / 2.0) * abs(point_cam[2])
    aspect_ratio = width / height
    proj_w = proj_h * aspect_ratio
    ndc_x = point_cam[0] / proj_w
    ndc_y = point_cam[1] / proj_h
    if not (-1 <= ndc_x <= 1 and -1 <= ndc_y <= 1):
        return None
    pixel_x = int((ndc_x + 1) * width / 2.0)
    pixel_y = int((1 - (ndc_y + 1) / 2.0) * height)
    return (pixel_x, pixel_y)

def rotation_matrix_to_6d(matrix):
    """Extract 6D rotation representation (first two columns) from a 3x3 rotation matrix"""
    return matrix[:, :2].T.flatten()

def playback_trajectory_with_env(
    args,
    ep,
    env,
    initial_state,
    states,
    actions=None,
    video_writer=None,
    camera_names=None,
    verbose=False,
):
    """
    Replay a single trajectory while extracting joint and camera data, and generating a visualization video.
    """
    write_video = video_writer is not None
    
    # Initialize lists for storing trajectory data
    trajectory_data = []

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    
    reset_to(env, initial_state)

    traj_len = states.shape[0]
    action_playback = actions is not None

    print(colored(f"Running episode {ep} and extracting data...", "yellow"))
    
    for i in tqdm(range(traj_len), desc=f"Processing {ep}"):
        if action_playback:
            env.step(actions[i])
        else:
            reset_to(env, {"states": states[i]})
        
        # --- Core data extraction logic ---
        step_data = {"timestep": i}

        # 1. Extract 3D coordinates and 6D rotation for all joint points
        for joint_name, (obj_type, obj_name, color) in JOINTS_TO_TRACK.items():
            pos, rot_6d = None, None
            try:
                if obj_type == "body":
                    body_id = env.sim.model.body_name2id(obj_name)
                    pos = env.sim.data.body_xpos[body_id].copy()
                    rot_mat = env.sim.data.body_xmat[body_id].copy().reshape(3, 3)
                    rot_6d = rotation_matrix_to_6d(rot_mat)
                else: # site
                    site_id = env.sim.model.site_name2id(obj_name)
                    pos = env.sim.data.site_xpos[site_id].copy()
                    rot_mat = env.sim.data.site_xmat[site_id].copy().reshape(3, 3)
                    rot_6d = rotation_matrix_to_6d(rot_mat)
                
                step_data[f"{joint_name}_pos"] = pos
                step_data[f"{joint_name}_rot6d"] = rot_6d
            except KeyError:
                # If the point cannot be found in the model, store NaN
                step_data[f"{joint_name}_pos"] = np.full(3, np.nan)
                step_data[f"{joint_name}_rot6d"] = np.full(6, np.nan)
                if verbose: print(f"Warning: Could not find {obj_type} '{obj_name}' in model.")

        # 2. Extract 3D coordinates and 6D rotation for all cameras
        for cam_name in camera_names:
            cam_id = env.sim.model.camera_name2id(cam_name)
            cam_pos = env.sim.data.cam_xpos[cam_id].copy()
            cam_mat = env.sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
            cam_rot_6d = rotation_matrix_to_6d(cam_mat)
            step_data[f"camera_{cam_name}_pos"] = cam_pos
            step_data[f"camera_{cam_name}_rot6d"] = cam_rot_6d
        
        trajectory_data.append(step_data)
        
        # --- Video rendering logic ---
        if write_video:
            video_img_list = []
            for cam_name in camera_names:
                im = env.sim.render(
                    height=args.render_height,
                    width=args.render_width,
                    camera_name=cam_name,
                )[::-1].copy()
                
                cam_id = env.sim.model.camera_name2id(cam_name)
                cam_pos = env.sim.data.cam_xpos[cam_id].copy()
                cam_mat = env.sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
                fovy_rad = np.deg2rad(env.sim.model.cam_fovy[cam_id])

                # Draw all joint points
                for joint_name, (obj_type, obj_name, color) in JOINTS_TO_TRACK.items():
                    world_pos_key = f"{joint_name}_pos"
                    if world_pos_key in step_data and step_data[world_pos_key] is not None:
                        world_pos = step_data[world_pos_key]
                        pixel_coords = project_world_to_pixel(
                            world_pos, cam_pos, cam_mat, fovy_rad, 
                            args.render_height, args.render_width
                        )
                        if pixel_coords is not None:
                            cv2.circle(im, pixel_coords, radius=5, color=color, thickness=-1)

                im = process_img_cotrain(im)
                video_img_list.append(im)

            video_img = np.concatenate(video_img_list, axis=1)
            video_writer.append_data(video_img)

    return trajectory_data

def reset_to(env, state):
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
            ep_meta = remove_dicts_with_keyword(ep_meta, "book")
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])

        state["model"] = remove_book_nodes(state["model"])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()
    return None

def get_env_metadata_from_dataset(dataset_path):
    with h5py.File(os.path.expanduser(dataset_path), "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    return env_meta

def make_env_from_args(args):
    env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs.update({
        "env_name": env_meta["env_name"],
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": False,
    })
    if "env_lang" in env_kwargs:
        env_kwargs.pop("env_lang")
    if args.verbose:
        print(colored(f"Initializing environment for {env_kwargs['env_name']}...", "yellow"))
    return robosuite.make(**env_kwargs)


def process_demo(args, ep):
    """
    Process a single demo: extract data, save as parquet, and generate visualization video.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = output_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    video_path = video_dir / f"{ep}.mp4"
    parquet_path = parquet_dir / f"{ep}.parquet"

    if os.path.exists(video_path) and os.path.exists(parquet_path):
        print(f"File paths {video_path} and {parquet_path} already exist!!!")
        return

    # Initialize video writer
    video_writer = imageio.get_writer(video_path, fps=20)

    try:
        with h5py.File(args.dataset, "r") as f:
            env = make_env_from_args(args)

            def make_ik_indicator_invisible(str_xml):
                import xml.etree.ElementTree as ET

                raw_xml = ET.fromstring(str_xml)
                for site in raw_xml.findall(".//site"):
                    name = site.get("name", "")
                    if "pinch_spheres" in name:
                        print(
                            colored(
                                "make site invisible: {}".format(name),
                                "yellow",
                            )
                        )
                        site.set("rgba", "0 0 0 0")
                return ET.tostring(raw_xml)

            states = f[f"data/{ep}/states"][()]
            initial_state = {
                "states": states[0],
                "model": make_ik_indicator_invisible(f[f"data/{ep}"].attrs["model_file"]),
                "ep_meta": f[f"data/{ep}"].attrs.get("ep_meta", None)
            }
            
            actions = f[f"data/{ep}/actions"][()] if args.use_actions else None

            # Execute playback, data extraction and video recording
            trajectory_data = playback_trajectory_with_env(
                args=args,
                ep=ep,
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                video_writer=video_writer,
                camera_names=args.render_image_names,
                verbose=args.verbose,
            )

            # Save extracted data to Parquet file
            if trajectory_data:
                df = pd.DataFrame(trajectory_data)
                df.to_parquet(parquet_path)
                print(colored(f"Saved data to {parquet_path}", "green"))

            env.close()

    except Exception as e:
        print(colored(f"Error processing episode {ep}: {e}", "red"))
        import traceback
        traceback.print_exc()
    finally:
        if video_writer:
            video_writer.close()
            print(colored(f"Saved video to {video_path}", "green"))


def main(args):
    # Auto-fill camera names
    if args.render_image_names is None:
        args.render_image_names = ["agentview"] # Default to agentview

    with h5py.File(args.dataset, "r") as f:
        demos = list(f["data"].keys())
        demos = sorted(demos, key=lambda x: int(x.split('_')[1]))

    if args.n is not None:
        demos = demos[:args.n]

    if args.num_parallel_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.num_parallel_jobs) as executor:
            list(tqdm(executor.map(partial(process_demo, args), demos), total=len(demos), desc="Overall Progress"))
    else:
        for ep in tqdm(demos, desc="Overall Progress"):
            process_demo(args, ep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to hdf5 dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output parquet and mp4 files")
    parser.add_argument("--n", type=int, default=None, help="Number of trajectories to process")
    parser.add_argument("--num_parallel_jobs", type=int, default=1, help="Number of parallel jobs to use")
    parser.add_argument("--use-actions", action="store_true", help="Use open-loop action playback")
    parser.add_argument("--render_image_names", type=str, nargs="+", default=None, help="Camera name(s) to use for rendering")
    parser.add_argument("--render_height", type=int, default=512, help="Height of rendered video frames")
    parser.add_argument("--render_width", type=int, default=512, help="Width of rendered video frames")
    parser.add_argument("--verbose", action="store_true", help="Log additional information")
    
    # Removed parameters from the original script that are not directly related to this functionality
    # e.g. --render, --filter_key, --first, etc.

    args = parser.parse_args()
    main(args)