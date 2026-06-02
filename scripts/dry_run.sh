#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VGGT_OMEGA_ROOT="${VGGT_OMEGA_ROOT:-${PROJECT_ROOT}/../vggt-omega}"
VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vggt_omega/vggt_omega_1b_512.pt}"

export PYTHONPATH="${PROJECT_ROOT}/src:${VGGT_OMEGA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python -m video_to_3d.cli \
  --vggt-root "${VGGT_OMEGA_ROOT}" \
  --checkpoint "${VGGT_OMEGA_CHECKPOINT}" \
  --dry-run
