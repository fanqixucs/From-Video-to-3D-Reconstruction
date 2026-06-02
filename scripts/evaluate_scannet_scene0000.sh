#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCANNET_BUNDLE="${SCANNET_BUNDLE:-${PROJECT_ROOT}/data/scannet_scene0000_00}"
VGGT_OMEGA_ROOT="${VGGT_OMEGA_ROOT:-${PROJECT_ROOT}/../vggt-omega}"
VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vggt_omega/vggt_omega_1b_512.pt}"

export PYTHONPATH="${PROJECT_ROOT}/src:${VGGT_OMEGA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python -m video_to_3d.evaluate_scannet \
  --bundle "${SCANNET_BUNDLE}" \
  --vggt-root "${VGGT_OMEGA_ROOT}" \
  --checkpoint "${VGGT_OMEGA_CHECKPOINT}" \
  --output-dir "${PROJECT_ROOT}/outputs" \
  --run-name scannet_scene0000_00_vggt_omega_1b_512 \
  --image-resolution 512 \
  --conf-thres 50 \
  --max-points 1000000 \
  --ply-conf-thres 5 \
  --ply-max-points 8000000 \
  "$@"
