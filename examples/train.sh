#!/bin/bash
# DIAL Training Script
# Usage: bash examples/train.sh <decoupled|end2end> [extra args...]
#
# MODE controls the bridge_type:
#   decoupled -> bridge_type=golden  (Stage 1: learn latent world model)
#   end2end   -> bridge_type=end2end (Stage 2: end-to-end fine-tuning)
#
# All other arguments are passed directly to the training script.
# Run from the project root directory.
#
# See examples below. The 24 GR1 simulation task names are defined in
# GR1_TASK_NAMES for reuse across examples.

set -euo pipefail

MODE=${1:?"Usage: bash examples/train.sh <decoupled|end2end> [extra args...]"}
shift

case "$MODE" in
  decoupled) BRIDGE_TYPE="golden" ;;
  end2end)   BRIDGE_TYPE="end2end" ;;
  *) echo "Error: Unknown MODE '$MODE'. Use 'decoupled' or 'end2end'."; exit 1 ;;
esac

echo "=============================================="
echo "DIAL Training"
echo "  MODE:        $MODE"
echo "  BRIDGE_TYPE: $BRIDGE_TYPE"
echo "=============================================="

python -u scripts/dual_system_train.py \
  --bridge_type "$BRIDGE_TYPE" \
  "$@"
