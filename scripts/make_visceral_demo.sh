#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/kitchen_small_room_vggt_omega_1b_512}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m video_to_3d.reconstruction_player \
  --run-dir "${RUN_DIR}" \
  --out-mp4 "${RUN_DIR}/visceral_demo.mp4" \
  --out-html "${RUN_DIR}/reconstruction_player.html" \
  --data-dir "${RUN_DIR}/reconstruction_player_data" \
  --final-cloud "${RUN_DIR}/pointcloud_full.ply" \
  --mode flythrough \
  --frames 300 \
  --fps 30 \
  --max-points 750000

# Re-encode the OpenCV (mp4v) output to web-friendly H.264; shrinks the file ~4x
# and makes it playable in browsers / GitHub previews.
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -loglevel error -i "${RUN_DIR}/visceral_demo.mp4" \
    -c:v libx264 -pix_fmt yuv420p -crf 23 -preset slow -movflags +faststart \
    "${RUN_DIR}/visceral_demo.tmp.mp4"
  mv "${RUN_DIR}/visceral_demo.tmp.mp4" "${RUN_DIR}/visceral_demo.mp4"
fi
