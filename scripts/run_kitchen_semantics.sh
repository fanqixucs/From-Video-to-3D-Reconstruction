#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VGGT_OMEGA_ROOT="${VGGT_OMEGA_ROOT:-${PROJECT_ROOT}/../vggt-omega}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/kitchen_small_room_vggt_omega_1b_512}"

export PYTHONPATH="${PROJECT_ROOT}/src:${VGGT_OMEGA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python -m video_to_3d.semantics \
  --run-dir "${RUN_DIR}" \
  --frame-stride 4 \
  --max-semantic-frames 20 \
  --box-threshold 0.20 \
  --text-threshold 0.20 \
  --detector-id IDEA-Research/grounding-dino-tiny \
  --sam-id facebook/sam-vit-huge
