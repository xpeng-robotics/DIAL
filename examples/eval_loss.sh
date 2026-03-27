#!/bin/bash
# Offline evaluation: compute action/goal loss on held-out data.
#
# Usage:
#   bash examples/eval_loss.sh <model_path> [data_config] [embodiment_tag] [trajs]
#
# Examples:
#   bash examples/eval_loss.sh checkpoints/DIAL-3B-fewshot
#   bash examples/eval_loss.sh checkpoints/DIAL-3B-fulldata fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop gr1 10

MODEL_PATH=${1:?"Usage: bash examples/eval_loss.sh <model_path> [data_config] [embodiment_tag] [trajs]"}
DATA_CONFIG=${2:-"fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop"}
EMBODIMENT_TAG=${3:-"gr1"}
TRAJS=${4:-10}

GR1_DATASET_DIR=/dataset_rc_mm/share/datasets/huggingface.co/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot-AugPosRot-Correct

PYTHONBREAKPOINT=0 python3 -u scripts/eval_policy_dial.py \
    --dataset-path \
        ${GR1_DATASET_DIR}/gr1_unified.PnPPotatoToMicrowaveClose \
        ${GR1_DATASET_DIR}/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA \
    --model_path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --embodiment_tag "$EMBODIMENT_TAG" \
    --trajs "$TRAJS" \
    --save_results_path "$MODEL_PATH/eval_action_goal_loss/" \
    --plot_state \
    --vis_pca
