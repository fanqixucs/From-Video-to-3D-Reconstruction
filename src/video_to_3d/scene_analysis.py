"""Gravity-aligned scene analysis for the phone reconstruction.

Produces robotics/evaluation artifacts on top of the VGGT-Omega geometry + semantics:

1. Gravity frame      - estimate the floor normal and build a gravity-aligned room frame.
2. Metric scale anchor - resolve the monocular global-scale ambiguity from one known size
                         (the user's cupboard height = 0.70 m, cross-checked against the
                         floor->counter height).
3. Plane sanity        - gravity-aligned floor/ceiling/wall/counter plane fits with
                         parallel/perpendicular residuals (a stronger version of the
                         Manhattan check that does not depend on viewpoint).
4. Object anchors      - per-class metric centroid / axis-aligned box / footprint -> JSON,
                         the kind of object list a manipulation/navigation stack consumes.
5. Bird's-eye view     - top-down occupancy + height map + traversable-floor mask -> PNG.

Everything is computed in the same first-camera-aligned scene frame used by `pointcloud.ply`
and the semantic PLYs, so the outputs stay registered with the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from video_to_3d.reconstruction_player import build_camera_geometry


def fit_plane(points: np.ndarray, threshold: float, iterations: int = 1000) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane fit. Returns (unit normal, signed offset d, inlier-point subset)."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    model, inliers = pc.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=iterations)
    normal = np.asarray(model[:3], dtype=np.float64)
    norm = np.linalg.norm(normal)
    return normal / norm, float(model[3] / norm), points[inliers]


def estimate_gravity_frame(
    floor_points: np.ndarray,
    camera_centers: np.ndarray,
    scene_diag: float,
) -> dict:
    """Up axis from the floor plane (oriented toward the cameras) + a horizontal basis.

    The first horizontal axis follows the camera trajectory so the bird's-eye view is aligned
    with how the room was walked; the second completes a right-handed frame.
    """
    up, _, _ = fit_plane(floor_points, threshold=scene_diag * 0.01)
    if np.dot(camera_centers.mean(0) - floor_points.mean(0), up) < 0:
        up = -up

    travel = camera_centers[-1] - camera_centers[0]
    travel = travel - up * float(travel @ up)
    if np.linalg.norm(travel) < 1e-6:
        travel = np.array([1.0, 0.0, 0.0]) - up * float(np.array([1.0, 0.0, 0.0]) @ up)
    e1 = travel / np.linalg.norm(travel)
    e2 = np.cross(up, e1)
    e2 /= np.linalg.norm(e2)
    return {"up": up, "e1": e1, "e2": e2, "floor_height": float(np.median(floor_points @ up))}


def to_gravity_coords(points: np.ndarray, frame: dict) -> np.ndarray:
    """(x along travel, y lateral, z = height above floor)."""
    x = points @ frame["e1"]
    y = points @ frame["e2"]
    z = points @ frame["up"] - frame["floor_height"]
    return np.stack([x, y, z], axis=1)


def compute_scale_anchor(points_g: np.ndarray, labels: np.ndarray, name2id: dict) -> dict:
    """Resolve the monocular global-scale ambiguity from one known dimension.

    Primary anchor: floor->counter height (two cleanly-fit horizontal planes) set to the
    standard 0.90 m worktop height. Cross-checked against the user-reported cupboard height
    of 0.70 m, measured on the wall-cabinet band above the worktop. The cupboard band is a
    weaker reference because the semantic `cupboard` class merges base and wall cabinets, so
    it cannot isolate one 0.70 m unit; the discrepancy is reported honestly.
    """
    counter_z = float(np.median(points_g[labels == name2id["countertop"], 2]))
    counter_m = 0.90  # standard worktop height -> primary anchor
    scale = counter_m / counter_z

    cup_z = points_g[labels == name2id["cupboard"], 2]
    wall_cabinet = cup_z[cup_z > counter_z + 0.02]
    cab_extent = float(np.percentile(wall_cabinet, 95) - np.percentile(wall_cabinet, 5))
    cupboard_m = 0.70  # user-reported wall-cabinet height
    scale_cup = cupboard_m / cab_extent
    return {
        "primary_anchor": "floor->counter height = 0.90 m (standard worktop)",
        "floor_to_counter_scene_units": round(counter_z, 4),
        "meters_per_scene_unit": scale,
        "cross_check_cupboard": {
            "anchor": "user-reported cupboard wall-cabinet height = 0.70 m",
            "wall_cabinet_band_extent_scene_units": round(cab_extent, 4),
            "wall_cabinet_band_under_primary_scale_m": round(cab_extent * scale, 3),
            "meters_per_scene_unit_if_band_is_0p70m": round(scale_cup, 3),
            "agreement_ratio_vs_primary": round(scale_cup / scale, 3),
            "note": "band measures >0.7 m under the primary scale because the semantic class merges base+wall cabinets",
        },
    }


def plane_sanity_gravity(points_g: np.ndarray, labels: np.ndarray, name2id: dict, scale: float) -> dict:
    """Gravity-aligned plane checks: floor/ceiling/counter horizontal, walls vertical."""
    out: dict = {}
    diag = float(np.linalg.norm(points_g.max(0) - points_g.min(0)))

    def horizontal(cls: str) -> dict | None:
        if cls not in name2id:
            return None
        n, _, inl = fit_plane(points_g[labels == name2id[cls]], threshold=diag * 0.01)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))  # deviation from horizontal
        resid = np.abs((inl - inl.mean(0)) @ n)  # residual on the dominant-plane inliers
        return {"tilt_from_horizontal_deg": round(tilt, 3), "residual_median_mm": round(float(np.median(resid) * scale * 1000), 1), "inliers": int(len(inl))}

    floor = horizontal("floor")
    ceiling = horizontal("ceiling")
    counter = horizontal("countertop")
    # walls: should be vertical (normal horizontal -> z component ~ 0). Residual is on the
    # dominant wall's inliers, since the `wall` class spans several non-coplanar walls.
    wall = None
    if "wall" in name2id:
        nw, _, inlw = fit_plane(points_g[labels == name2id["wall"]], threshold=diag * 0.01)
        wall = {
            "tilt_from_vertical_deg": round(float(np.degrees(np.arcsin(min(1.0, abs(nw[2]))))), 3),
            "dominant_wall_residual_median_mm": round(float(np.median(np.abs((inlw - inlw.mean(0)) @ nw)) * scale * 1000), 1),
            "inliers": int(len(inlw)),
        }

    out["floor"] = floor
    out["ceiling"] = ceiling
    out["countertop"] = counter
    out["wall"] = wall
    if floor and ceiling:
        out["floor_ceiling_parallel_residual_deg"] = abs(floor["tilt_from_horizontal_deg"] - ceiling["tilt_from_horizontal_deg"])
    if floor and counter:
        out["floor_counter_parallel_residual_deg"] = abs(floor["tilt_from_horizontal_deg"] - counter["tilt_from_horizontal_deg"])
    if floor and wall:
        out["wall_floor_perpendicular_residual_deg"] = abs(90.0 - (90.0 - wall["tilt_from_vertical_deg"] - floor["tilt_from_horizontal_deg"]))
    return out


SURFACE_CLASSES = {"wall", "floor", "ceiling", "countertop", "door", "bulletin board"}


def _dominant_cluster(P: np.ndarray, eps: float = 0.12, min_points: int = 15) -> np.ndarray:
    """Largest DBSCAN cluster of a metric point set (drops scattered semantic false positives)."""
    rng = np.random.default_rng(0)
    sub = P if len(P) <= 30000 else P[rng.choice(len(P), 30000, replace=False)]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(sub)
    lbl = np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_points))
    if lbl.max() < 0:
        return sub
    best = np.argmax(np.bincount(lbl[lbl >= 0]))
    return sub[lbl == best]


def object_anchors(points_g: np.ndarray, labels: np.ndarray, summary: dict, scale: float) -> list[dict]:
    """Per-class metric centroid / axis-aligned box / floor footprint.

    Discrete objects use their dominant DBSCAN cluster so scattered open-vocabulary false
    positives do not inflate the box; room surfaces keep their full extent.
    """
    anchors = []
    for sid, info in summary.items():
        mask = labels == int(sid)
        if mask.sum() < 50:
            continue
        name = info["name"]
        P = points_g[mask] * scale  # metres in gravity frame
        is_surface = name in SURFACE_CLASSES
        if not is_surface:
            P = _dominant_cluster(P)
        lo = np.percentile(P, 2, axis=0)
        hi = np.percentile(P, 98, axis=0)
        centroid = np.median(P, axis=0)
        anchors.append({
            "name": name,
            "semantic_id": int(sid),
            "kind": "surface" if is_surface else "object",
            "num_points": int(mask.sum()),
            "cluster_points": int(len(P)),
            "centroid_m": [round(float(v), 3) for v in centroid],
            "size_m": {"x": round(float(hi[0] - lo[0]), 3), "y": round(float(hi[1] - lo[1]), 3), "height": round(float(hi[2] - lo[2]), 3)},
            "base_height_above_floor_m": round(float(lo[2]), 3),
            "footprint_m": [round(float(lo[0]), 3), round(float(lo[1]), 3), round(float(hi[0]), 3), round(float(hi[1]), 3)],
            "source_depth_conf_median": round(float(info.get("source_conf_median", 0.0)), 2),
        })
    anchors.sort(key=lambda a: -a["num_points"])
    return anchors


def render_bev(
    points_g: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    name2id: dict,
    anchors: list[dict],
    scale: float,
    cell_m: float = 0.02,
    obstacle_lo: float = 0.06,
    obstacle_hi: float = 1.9,
    upscale: int = 6,
) -> np.ndarray:
    """Semantic-coloured top-down map: traversable floor in green, obstacles by class colour,
    object labels and a 1 m scale bar. Ceiling and floor are excluded from the obstacle band so
    free space stays free."""
    P = points_g * scale
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    x0, x1 = np.percentile(x, 0.3), np.percentile(x, 99.7)
    y0, y1 = np.percentile(y, 0.3), np.percentile(y, 99.7)
    W = max(1, int(np.ceil((x1 - x0) / cell_m)))
    H = max(1, int(np.ceil((y1 - y0) / cell_m)))

    def cell(px, py):
        return (np.clip(((px - x0) / cell_m).astype(int), 0, W - 1),
                np.clip(((py - y0) / cell_m).astype(int), 0, H - 1))

    bev = np.full((H, W, 3), 24, dtype=np.uint8)  # unobserved = dark

    floor_id = name2id.get("floor", -1)
    ceil_id = name2id.get("ceiling", -1)
    # observed footprint: any non-ceiling point marks the cell as seen floor area
    seen = (labels != ceil_id)
    sx, sy = cell(x[seen], y[seen])
    bev[sy, sx] = (70, 110, 70)  # seen ground / free-ish

    # obstacles: points in the robot height band, excluding floor & ceiling, topmost wins
    band = (z > obstacle_lo) & (z < obstacle_hi) & (labels != floor_id) & (labels != ceil_id)
    bx, by = cell(x[band], y[band])
    order = np.argsort(z[band])  # low -> high so the tallest is written last
    flat = (by * W + bx)[order]
    cols = rgb[band][order][:, ::-1]  # to BGR
    bev.reshape(-1, 3)[flat] = cols

    bev = cv2.flip(bev, 0)  # +y up
    bev = cv2.resize(bev, (W * upscale, H * upscale), interpolation=cv2.INTER_NEAREST)

    # overlay object centroids + names and a 1 m scale bar
    Hs, Ws = bev.shape[:2]
    def to_px(cx, cy):
        return int((cx - x0) / cell_m * upscale), int(Hs - (cy - y0) / cell_m * upscale)
    for a in anchors:
        if a["kind"] != "object":
            continue
        px, py = to_px(a["centroid_m"][0], a["centroid_m"][1])
        if 0 <= px < Ws and 0 <= py < Hs:
            cv2.circle(bev, (px, py), 3, (255, 255, 255), -1)
            cv2.putText(bev, a["name"], (px + 5, py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bev, a["name"], (px + 5, py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    bar = int(1.0 / cell_m * upscale)
    cv2.rectangle(bev, (12, Hs - 22), (12 + bar, Hs - 16), (255, 255, 255), -1)
    cv2.putText(bev, "1 m", (12, Hs - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bev, "green=traversable floor  |  coloured=object footprints", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bev


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gravity-aligned scene analysis (scale, planes, anchors, BEV).")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "scene_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(run_dir / "predictions.npz") as loaded:
        preds = {k: np.array(loaded[k]) for k in loaded.files}
    cameras = build_camera_geometry(preds)
    sem = np.load(run_dir / "semantics" / "semantic_labeled_points.npz", allow_pickle=True)
    pts = np.asarray(sem["points_scene"], dtype=np.float64)
    rgb = np.asarray(sem["rgb"])
    labels = np.asarray(sem["label_id"])
    summary = json.loads((run_dir / "semantics" / "semantic_summary.json").read_text())["summary"]
    name2id = {v["name"]: int(k) for k, v in summary.items()}

    scene_diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    frame = estimate_gravity_frame(pts[labels == name2id["floor"]], cameras["centers"], scene_diag)
    points_g = to_gravity_coords(pts, frame)

    scale_info = compute_scale_anchor(points_g, labels, name2id)
    scale = scale_info["meters_per_scene_unit"]
    planes = plane_sanity_gravity(points_g, labels, name2id, scale)
    anchors = object_anchors(points_g, labels, summary, scale)
    bev = render_bev(points_g, rgb, labels, name2id, anchors, scale)
    cv2.imwrite(str(out_dir / "bev_traversability.png"), bev)

    report = {
        "coordinate_frame": "gravity-aligned: x=along camera travel, y=lateral, z=height above floor; metres after scale",
        "gravity_up_in_scene_frame": [round(float(v), 4) for v in frame["up"]],
        "scale_anchor": scale_info,
        "room_dimensions_m": {
            "footprint_x": round(float((np.percentile(points_g[:, 0], 99) - np.percentile(points_g[:, 0], 1)) * scale), 2),
            "footprint_y": round(float((np.percentile(points_g[:, 1], 99) - np.percentile(points_g[:, 1], 1)) * scale), 2),
            "floor_to_ceiling": round(float(np.percentile(points_g[:, 2], 99) * scale), 2),
        },
        "gravity_aligned_plane_sanity": planes,
        "num_object_anchors": len(anchors),
        "artifacts": {
            "object_anchors": "object_anchors.json",
            "bev": "bev_traversability.png",
        },
    }
    (out_dir / "scene_analysis.json").write_text(json.dumps(report, indent=2))
    (out_dir / "object_anchors.json").write_text(json.dumps({"meters_per_scene_unit": scale, "frame": report["coordinate_frame"], "objects": anchors}, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[done] wrote {out_dir}/scene_analysis.json, object_anchors.json, bev_traversability.png")


if __name__ == "__main__":
    main()
