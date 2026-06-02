#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO_PATH="${VIDEO_PATH:-${PROJECT_ROOT}/data/kitchen_small_room.MOV}"
VGGT_OMEGA_ROOT="${VGGT_OMEGA_ROOT:-${PROJECT_ROOT}/../vggt-omega}"
VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vggt_omega/vggt_omega_1b_512.pt}"
RUN_NAME="${RUN_NAME:-kitchen_small_room_vggt_omega_1b_512}"

export PYTHONPATH="${PROJECT_ROOT}/src:${VGGT_OMEGA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python -m video_to_3d.cli \
  --video "${VIDEO_PATH}" \
  --vggt-root "${VGGT_OMEGA_ROOT}" \
  --checkpoint "${VGGT_OMEGA_CHECKPOINT}" \
  --output-dir "${PROJECT_ROOT}/outputs" \
  --run-name "${RUN_NAME}" \
  --image-resolution 512 \
  --sample-fps 1.0 \
  --conf-thres 50 \
  --max-points 1000000 \
  --ply-conf-thres 5 \
  --ply-max-points 8000000 \
  --overwrite
