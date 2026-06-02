#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/kitchen_small_room_vggt_omega_1b_512}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m video_to_3d.evaluate_no_gt \
  --predictions "${RUN_DIR}/predictions.npz" \
  --out "${RUN_DIR}/no_gt_eval_metrics.json"

