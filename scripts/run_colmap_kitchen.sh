#!/usr/bin/env bash
# Classical SfM + dense MVS baseline with COLMAP, on the same sampled kitchen frames that
# VGGT-Omega consumes. Used as a cross-method comparison in the README.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN="${RUN_DIR:-${ROOT}/outputs/kitchen_small_room_vggt_omega_1b_512}"
IMG="${RUN}/frames"
WS="${RUN}/colmap"
DB="${WS}/database.db"
# COLMAP lives in its own conda env (CUDA build). Override with COLMAP_ENV=<name>.
CONDA_BIN="${CONDA_EXE:-conda}"
CR="${CONDA_BIN} run -n ${COLMAP_ENV:-colmap}"

mkdir -p "${WS}/sparse" "${WS}/dense"

echo "[colmap] feature extraction (CPU SIFT, single shared camera)"
${CR} colmap feature_extractor --database_path "${DB}" --image_path "${IMG}" \
  --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.use_gpu 0 --SiftExtraction.max_image_size 1600

echo "[colmap] exhaustive matching (CPU)"
${CR} colmap exhaustive_matcher --database_path "${DB}" --SiftMatching.use_gpu 0

echo "[colmap] sparse mapping (SfM)"
${CR} colmap mapper --database_path "${DB}" --image_path "${IMG}" --output_path "${WS}/sparse"

echo "[colmap] sparse model stats"
MODEL="${WS}/sparse/0"
${CR} colmap model_analyzer --path "${MODEL}" 2>&1 | tee "${WS}/sparse_stats.txt" || true
${CR} colmap model_converter --input_path "${MODEL}" --output_path "${WS}/sparse_points.ply" --output_type PLY || true

echo "[colmap] dense: undistort -> patch_match_stereo (GPU CUDA) -> fusion"
${CR} colmap image_undistorter --image_path "${IMG}" --input_path "${MODEL}" \
  --output_path "${WS}/dense" --output_type COLMAP --max_image_size 1600
${CR} colmap patch_match_stereo --workspace_path "${WS}/dense" --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true --PatchMatchStereo.gpu_index 0
${CR} colmap stereo_fusion --workspace_path "${WS}/dense" --workspace_format COLMAP \
  --input_type geometric --output_path "${WS}/dense/fused.ply"

echo "COLMAP_PIPELINE_DONE"
ls -la "${WS}"/*.ply "${WS}/dense/fused.ply" 2>/dev/null
