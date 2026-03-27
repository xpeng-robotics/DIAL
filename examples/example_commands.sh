#!/bin/bash
# =====================================================================
# DIAL Example Training & Evaluation Commands
# =====================================================================
# This file contains ready-to-use commands for all training scenarios.
# Adjust GR1_DIR, EGODEX_DIR, and output paths to your setup.
# Run from the project root directory.
# =====================================================================

# ---------- Common paths (modify these) ----------
GR1_DIR=/path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot-AugPosRot-Correct
EGODEX_DIR=/path/to/egodex_lerobot_gr00t

# ---------- 24 GR1 task names (shared across all scenarios) ----------
GR1_TASK_NAMES=(
    PnPPotatoToMicrowaveClose
    PnPMilkToMicrowaveClose
    PnPCanToDrawerClose
    PnPCupToDrawerClose
    PnPBottleToCabinetClose
    PnPWineToCabinetClose
    PosttrainPnPNovelFromPlacematToBowlSplitA
    PosttrainPnPNovelFromPlateToPlateSplitA
    PosttrainPnPNovelFromPlacematToPlateSplitA
    PosttrainPnPNovelFromCuttingboardToPotSplitA
    PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
    PosttrainPnPNovelFromCuttingboardToPanSplitA
    PosttrainPnPNovelFromTrayToCardboardboxSplitA
    PosttrainPnPNovelFromTrayToTieredshelfSplitA
    PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
    PosttrainPnPNovelFromPlacematToTieredshelfSplitA
    PosttrainPnPNovelFromPlateToCardboardboxSplitA
    PosttrainPnPNovelFromPlacematToBasketSplitA
    PosttrainPnPNovelFromPlateToPanSplitA
    PosttrainPnPNovelFromTrayToTieredbasketSplitA
    PosttrainPnPNovelFromTrayToPotSplitA
    PosttrainPnPNovelFromPlateToBowlSplitA
    PosttrainPnPNovelFromCuttingboardToBasketSplitA
    PosttrainPnPNovelFromTrayToPlateSplitA
)

# Build full dataset paths from task names
GR1_DATASETS=()
for t in "${GR1_TASK_NAMES[@]}"; do
    GR1_DATASETS+=("${GR1_DIR}/gr1_unified.${t}")
done

# ---------- Shared training flags ----------
COMMON_FLAGS=(
    --compute-bridge-loss
    --tune-llm
    --no-tune-visual
    --tune-projector
    --tune-diffusion-model
    --no-tune-bridge-visual
    --no-tune-bridge-goal
    --tune-bridge-embedding
    --select_layer 36
    --use_image_type_embedding
    --ignore_lang_prefix
    --report_to tensorboard
)

GR1_DATA_CONFIG=fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop
EGODEX_DATA_CONFIG=human_egodex_hand_wrist_gr1_only_align_coord_gausNorm_crop


# =====================================================================
# Example 1: Co-pretrain with EgoDex + GR1 sim (100-shot per task)
# =====================================================================

# Build per-dataset arrays for mixed co-training
ALL_PATHS=("${EGODEX_DIR}/part2/basic_pick_place" "${GR1_DATASETS[@]}")

ALL_CONFIGS=("${EGODEX_DATA_CONFIG}")
ALL_TAGS=(human_egodex)
ALL_SPLITS=("[:-10]")
for t in "${GR1_TASK_NAMES[@]}"; do
    ALL_CONFIGS+=("${GR1_DATA_CONFIG}")
    ALL_TAGS+=(gr1)
    ALL_SPLITS+=("[:100]")
done

bash examples/train.sh decoupled \
    --dataset-path "${ALL_PATHS[@]}" \
    --data-config "${ALL_CONFIGS[@]}" \
    --embodiment_tag "${ALL_TAGS[@]}" \
    --data_splits "${ALL_SPLITS[@]}" \
    --unified_embodiment_tag gr1 \
    --num-gpus 8 --batch-size 32 --max-steps 20000 --save-steps 20000 \
    --base_model_path gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json \
    "${COMMON_FLAGS[@]}" \
    --output_dir outputs/co-pretrain-decoupled/


# =====================================================================
# Example 2: Co-finetune with EgoDex + GR1 (end2end, from co-pretrain)
# =====================================================================
# Same mixed data as co-pretrain, but bridge_type=end2end.
# No --use_separate_projector_for_loss at this stage.

bash examples/train.sh end2end \
    --dataset-path "${ALL_PATHS[@]}" \
    --data-config "${ALL_CONFIGS[@]}" \
    --embodiment_tag "${ALL_TAGS[@]}" \
    --data_splits "${ALL_SPLITS[@]}" \
    --unified_embodiment_tag gr1 \
    --num-gpus 8 --batch-size 32 --max-steps 20000 --save-steps 20000 \
    --base_model_path outputs/co-pretrain-decoupled/checkpoint-20000 \
    "${COMMON_FLAGS[@]}" \
    --output_dir outputs/co-finetune-end2end/


# =====================================================================
# Example 3: Finetune GR1 only from co-finetune (100-shot)
# =====================================================================
# GR1 data only, with --use_separate_projector_for_loss.
# --data-config and --embodiment_tag broadcast automatically.

bash examples/train.sh end2end \
    --dataset-path "${GR1_DATASETS[@]}" \
    --data-config "${GR1_DATA_CONFIG}" \
    --embodiment_tag gr1 \
    --data_split "[:100]" \
    --num-gpus 8 --batch-size 32 --max-steps 20000 --save-steps 5000 \
    --base_model_path outputs/co-finetune-end2end/checkpoint-20000 \
    "${COMMON_FLAGS[@]}" \
    --use_separate_projector_for_loss \
    --output_dir outputs/finetune-end2end-from-cofinetune/


# =====================================================================
# Example 4: Pretrain on sim data only (all data, no human data)
# =====================================================================

bash examples/train.sh decoupled \
    --dataset-path "${GR1_DATASETS[@]}" \
    --data-config "${GR1_DATA_CONFIG}" \
    --embodiment_tag gr1 \
    --data_split "[:-10]" \
    --num-gpus 8 --batch-size 32 --max-steps 80000 --save-steps 20000 \
    --base_model_path gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json \
    "${COMMON_FLAGS[@]}" \
    --output_dir outputs/pretrain-decoupled-sim-only/


# =====================================================================
# Example 5: End-to-end finetune from sim-only pretrain (all data)
# =====================================================================

bash examples/train.sh end2end \
    --dataset-path "${GR1_DATASETS[@]}" \
    --data-config "${GR1_DATA_CONFIG}" \
    --embodiment_tag gr1 \
    --data_split "[:-10]" \
    --num-gpus 8 --batch-size 32 --max-steps 80000 --save-steps 20000 \
    --base_model_path outputs/pretrain-decoupled-sim-only/checkpoint-80000 \
    "${COMMON_FLAGS[@]}" \
    --use_separate_projector_for_loss \
    --output_dir outputs/finetune-end2end-sim-only/
