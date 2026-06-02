#!/usr/bin/env bash
# Fast smoke test: validates imports and the analysis/evaluation pipeline on the committed
# kitchen outputs WITHOUT needing the VGGT-Omega checkpoint or a GPU. Runs in ~30s.
# (For a deeper check of the VGGT-Omega wiring + checkpoint load, use scripts/dry_run.sh.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN="${RUN_DIR:-${ROOT}/outputs/kitchen_small_room_vggt_omega_1b_512}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

echo "[1/4] import check"
python -c "import video_to_3d.scene_analysis, video_to_3d.novel_view, video_to_3d.reconstruction_player, video_to_3d.export, video_to_3d.metrics, video_to_3d.semantics; print('  imports OK')"

# Write to a throwaway dir so the smoke test never clobbers the canonical artifacts.
SMOKE="${RUN}/_smoke"
trap 'rm -rf "${SMOKE}"' EXIT
rm -rf "${SMOKE}"; mkdir -p "${SMOKE}"

echo "[2/4] gravity-aligned scene analysis (scale / planes / anchors / BEV)"
python -m video_to_3d.scene_analysis --run-dir "${RUN}" --out-dir "${SMOKE}/scene_analysis" >/dev/null
echo "  scene_analysis OK"

echo "[3/4] held-out novel-view metrics (4 views)"
python -m video_to_3d.novel_view --run-dir "${RUN}" --out-dir "${SMOKE}/novel_view" --num-views 4 >/dev/null
echo "  novel_view OK"

echo "[4/4] artifact check"
for f in scene_analysis/object_anchors.json scene_analysis/bev_traversability.png novel_view/novel_view_metrics.json; do
  if [ -s "${SMOKE}/${f}" ]; then echo "  ok ${f}"; else echo "  MISSING ${f}"; exit 1; fi
done

echo "SMOKE TEST PASSED"
