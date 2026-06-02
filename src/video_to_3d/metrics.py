from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: np.array(loaded[key]) for key in loaded.files}


def save_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def extrinsic_to_cam2world(extrinsic: np.ndarray) -> np.ndarray:
    mats = np.tile(np.eye(4, dtype=np.float64), (len(extrinsic), 1, 1))
    mats[:, :3, :4] = extrinsic[:, :3, :4]
    return np.linalg.inv(mats)


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("src and dst must have the same length >= 3")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    cov = dst_centered.T @ src_centered / len(src)
    u, singular_values, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    rotation = u @ d @ vt
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ d) / variance) if variance > 1e-12 else 1.0
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def camera_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict:
    pred_centers = pred_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    scale, rotation, translation = umeyama_align(pred_centers, gt_centers)
    pred_centers_aligned = apply_sim3(pred_centers, scale, rotation, translation)
    trans_err = np.linalg.norm(pred_centers_aligned - gt_centers, axis=1)

    rot_err = []
    for pred_pose, gt_pose in zip(pred_c2w, gt_c2w):
        pred_rot_aligned = rotation @ pred_pose[:3, :3]
        delta = pred_rot_aligned.T @ gt_pose[:3, :3]
        cos_angle = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        rot_err.append(np.degrees(np.arccos(cos_angle)))
    rot_err = np.asarray(rot_err)

    return {
        "n_frames": int(len(pred_c2w)),
        "ate_rmse_m": float(np.sqrt(np.mean(trans_err**2))),
        "ate_mean_m": float(np.mean(trans_err)),
        "ate_median_m": float(np.median(trans_err)),
        "ate_max_m": float(np.max(trans_err)),
        "rotation_mean_deg": float(np.mean(rot_err)),
        "rotation_median_deg": float(np.median(rot_err)),
        "sim3_scale": float(scale),
        "sim3_rotation": rotation.tolist(),
        "sim3_translation": translation.tolist(),
    }


def depth_maps_to_world_points(
    depths: np.ndarray,
    intrinsic: np.ndarray,
    cam2world: np.ndarray,
    max_points: int = 500_000,
) -> np.ndarray:
    rng = np.random.default_rng(0)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    all_points = []

    for depth, pose in zip(depths, cam2world):
        valid = np.isfinite(depth) & (depth > 0)
        ys, xs = np.where(valid)
        if len(xs) == 0:
            continue
        z = depth[ys, xs]
        points_cam = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], axis=1)
        points_world = (pose[:3, :3] @ points_cam.T).T + pose[:3, 3]
        all_points.append(points_world.astype(np.float32))

    if not all_points:
        return np.zeros((0, 3), dtype=np.float32)

    points = np.concatenate(all_points, axis=0)
    if max_points > 0 and len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def predicted_world_points_for_eval(
    predictions: dict[str, np.ndarray],
    max_points: int = 500_000,
) -> np.ndarray:
    points = predictions["world_points_from_depth"].reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(1)
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points.astype(np.float32)


def chamfer_fscore(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    thresholds_m: tuple[float, ...] = (0.01, 0.05, 0.10),
    max_sample: int = 200_000,
) -> dict:
    import open3d as o3d

    rng = np.random.default_rng(2)
    if len(pred_points) > max_sample:
        pred_points = pred_points[rng.choice(len(pred_points), max_sample, replace=False)]
    if len(gt_points) > max_sample:
        gt_points = gt_points[rng.choice(len(gt_points), max_sample, replace=False)]

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

    pred_to_gt = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    gt_to_pred = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))
    result = {
        "n_pred_eval": int(len(pred_points)),
        "n_gt_eval": int(len(gt_points)),
        "accuracy_mean_m": float(pred_to_gt.mean()),
        "completeness_mean_m": float(gt_to_pred.mean()),
        "chamfer_l1_m": float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0),
        "chamfer_l2_m": float(np.sqrt((np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)) / 2.0)),
    }
    for threshold in thresholds_m:
        precision = float(np.mean(pred_to_gt < threshold))
        recall = float(np.mean(gt_to_pred < threshold))
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        label = f"{int(round(threshold * 100))}cm"
        result[f"precision@{label}"] = precision
        result[f"recall@{label}"] = recall
        result[f"f1@{label}"] = f1
    return result


def multiview_depth_color_consistency(
    predictions: dict[str, np.ndarray],
    max_samples_per_pair: int = 1500,
    strides: tuple[int, ...] = (1, 4, 8),
) -> dict:
    depths = predictions["depth"][..., 0]
    points = predictions["world_points_from_depth"]
    extrinsics = predictions["extrinsic"]
    intrinsics = predictions["intrinsic"]
    images = _images_to_hwc(predictions["images"])
    num_frames, height, width = depths.shape
    rng = np.random.default_rng(3)

    rel_errors = []
    color_errors = []
    valid_projection_count = 0
    attempted = 0

    for stride in strides:
        for i in range(0, num_frames - stride):
            j = i + stride
            valid = np.isfinite(points[i]).all(axis=-1) & np.isfinite(depths[i]) & (depths[i] > 0)
            ys, xs = np.where(valid)
            if len(xs) == 0:
                continue
            take = min(max_samples_per_pair, len(xs))
            idx = rng.choice(len(xs), take, replace=False)
            ys = ys[idx]
            xs = xs[idx]
            world = points[i, ys, xs]
            attempted += len(world)

            cam = (extrinsics[j, :3, :3] @ world.T).T + extrinsics[j, :3, 3]
            z = cam[:, 2]
            positive = z > 1e-6
            u = intrinsics[j, 0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + intrinsics[j, 0, 2]
            v = intrinsics[j, 1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + intrinsics[j, 1, 2]
            ui = np.rint(u).astype(np.int32)
            vi = np.rint(v).astype(np.int32)
            inside = positive & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            if not inside.any():
                continue

            z = z[inside]
            ui = ui[inside]
            vi = vi[inside]
            src_y = ys[inside]
            src_x = xs[inside]
            target_depth = depths[j, vi, ui]
            valid_depth = np.isfinite(target_depth) & (target_depth > 0)
            if not valid_depth.any():
                continue
            z = z[valid_depth]
            ui = ui[valid_depth]
            vi = vi[valid_depth]
            src_y = src_y[valid_depth]
            src_x = src_x[valid_depth]
            target_depth = target_depth[valid_depth]
            valid_projection_count += len(z)

            rel = np.abs(z - target_depth) / np.maximum(np.maximum(np.abs(z), np.abs(target_depth)), 1e-6)
            rel_errors.append(rel)
            visible = rel < 0.10
            if visible.any():
                src_rgb = images[i, src_y[visible], src_x[visible]]
                dst_rgb = images[j, vi[visible], ui[visible]]
                color_errors.append(np.mean(np.abs(src_rgb - dst_rgb), axis=1))

    rel_errors_arr = np.concatenate(rel_errors) if rel_errors else np.array([], dtype=np.float32)
    color_errors_arr = np.concatenate(color_errors) if color_errors else np.array([], dtype=np.float32)
    return {
        "attempted_projections": int(attempted),
        "valid_projected_depths": int(valid_projection_count),
        "valid_projection_ratio": float(valid_projection_count / attempted) if attempted else 0.0,
        "depth_rel_median": _safe_percentile(rel_errors_arr, 50),
        "depth_rel_p90": _safe_percentile(rel_errors_arr, 90),
        "depth_rel_under_5pct": float(np.mean(rel_errors_arr < 0.05)) if rel_errors_arr.size else 0.0,
        "depth_rel_under_10pct": float(np.mean(rel_errors_arr < 0.10)) if rel_errors_arr.size else 0.0,
        "color_l1_median_visible": _safe_percentile(color_errors_arr, 50),
        "color_l1_p90_visible": _safe_percentile(color_errors_arr, 90),
    }


def feature_reprojection_consistency(
    predictions: dict[str, np.ndarray],
    max_pairs: int = 24,
    stride: int = 1,
) -> dict:
    images = (_images_to_hwc(predictions["images"]) * 255).clip(0, 255).astype(np.uint8)
    extrinsics = predictions["extrinsic"]
    intrinsics = predictions["intrinsic"]
    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    residuals = []
    pair_count = 0
    match_count = 0
    for i in range(0, min(len(images) - stride, max_pairs)):
        j = i + stride
        gray_i = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        gray_j = cv2.cvtColor(images[j], cv2.COLOR_RGB2GRAY)
        kp_i, des_i = orb.detectAndCompute(gray_i, None)
        kp_j, des_j = orb.detectAndCompute(gray_j, None)
        if des_i is None or des_j is None:
            continue
        matches = sorted(matcher.match(des_i, des_j), key=lambda m: m.distance)[:300]
        if len(matches) < 12:
            continue
        pts_i = np.array([kp_i[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts_j = np.array([kp_j[m.trainIdx].pt for m in matches], dtype=np.float64)
        _, inlier_mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 1.5, 0.99)
        if inlier_mask is None:
            continue
        inlier_mask = inlier_mask.reshape(-1).astype(bool)
        pts_i = pts_i[inlier_mask]
        pts_j = pts_j[inlier_mask]
        if len(pts_i) < 12:
            continue
        residual = _sampson_residual_px(pts_i, pts_j, extrinsics[i], extrinsics[j], intrinsics[i], intrinsics[j])
        if residual.size:
            residuals.append(residual)
            pair_count += 1
            match_count += len(residual)

    residuals_arr = np.concatenate(residuals) if residuals else np.array([], dtype=np.float32)
    return {
        "pairs_evaluated": int(pair_count),
        "matches_evaluated": int(match_count),
        "sampson_px_median": _safe_percentile(residuals_arr, 50),
        "sampson_px_p90": _safe_percentile(residuals_arr, 90),
        "sampson_px_under_2": float(np.mean(residuals_arr < 2.0)) if residuals_arr.size else 0.0,
        "sampson_px_under_5": float(np.mean(residuals_arr < 5.0)) if residuals_arr.size else 0.0,
    }


def plane_sanity_metrics(predictions: dict[str, np.ndarray], max_points: int = 300_000) -> dict:
    import open3d as o3d

    points = predicted_world_points_for_eval(predictions, max_points=max_points)
    if len(points) < 1000:
        return {"planes_found": 0}

    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    scene_diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    threshold = max(scene_diag * 0.01, 1e-4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    planes = []
    remaining = pcd
    for _ in range(5):
        if len(remaining.points) < 2000:
            break
        model, inliers = remaining.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=500)
        if len(inliers) < max(2000, int(0.03 * len(points))):
            break
        normal = np.asarray(model[:3], dtype=np.float64)
        normal /= max(np.linalg.norm(normal), 1e-12)
        inlier_cloud = remaining.select_by_index(inliers)
        inlier_points = np.asarray(inlier_cloud.points)
        distances = np.abs(inlier_points @ normal + model[3])
        planes.append(
            {
                "normal": normal,
                "inlier_count": int(len(inliers)),
                "residual_mean": float(distances.mean()),
                "residual_median": float(np.median(distances)),
            }
        )
        remaining = remaining.select_by_index(inliers, invert=True)

    angle_residuals = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            angle = np.degrees(np.arccos(np.clip(abs(float(planes[i]["normal"] @ planes[j]["normal"])), 0.0, 1.0)))
            angle_residuals.append(min(abs(angle - 0.0), abs(angle - 90.0)))

    residuals = np.array([plane["residual_median"] for plane in planes], dtype=np.float64)
    return {
        "planes_found": int(len(planes)),
        "scene_diag_world_units": scene_diag,
        "plane_distance_threshold_world_units": threshold,
        "plane_residual_median_world_units": float(np.median(residuals)) if residuals.size else None,
        "plane_residual_median_pct_scene_diag": float(np.median(residuals) / scene_diag) if residuals.size else None,
        "orthogonality_angle_residual_median_deg": float(np.median(angle_residuals)) if angle_residuals else None,
        "planes": [
            {
                "inlier_count": plane["inlier_count"],
                "residual_mean": plane["residual_mean"],
                "residual_median": plane["residual_median"],
                "normal": plane["normal"].tolist(),
            }
            for plane in planes
        ],
    }


def no_gt_quality_gates(metrics: dict) -> dict:
    depth = metrics["multi_view"]["depth_rel_median"]
    feature = metrics["feature_reprojection"]["sampson_px_median"]
    plane = metrics["plane_sanity"].get("plane_residual_median_pct_scene_diag")
    ortho = metrics["plane_sanity"].get("orthogonality_angle_residual_median_deg")

    gates = {
        "multi_view_depth": _gate(depth, green=0.05, yellow=0.10, lower_is_better=True),
        "feature_reprojection": _gate(feature, green=2.0, yellow=5.0, lower_is_better=True),
        "plane_residual": _gate(plane, green=0.01, yellow=0.025, lower_is_better=True),
        "manhattan_orthogonality": _gate(ortho, green=5.0, yellow=10.0, lower_is_better=True),
    }
    score_map = {"green": 1.0, "yellow": 0.5, "red": 0.0, "n/a": 0.0}
    gates["coherence_score_0_to_1"] = float(np.mean([score_map[value] for value in gates.values()]))
    return gates


def run_no_gt_evaluation(predictions: dict[str, np.ndarray]) -> dict:
    metrics = {
        "multi_view": multiview_depth_color_consistency(predictions),
        "feature_reprojection": feature_reprojection_consistency(predictions),
        "plane_sanity": plane_sanity_metrics(predictions),
    }
    metrics["quality_gates"] = no_gt_quality_gates(metrics)
    metrics["thresholds"] = {
        "multi_view_depth_rel_median": "green < 0.05, yellow < 0.10",
        "feature_sampson_px_median": "green < 2 px, yellow < 5 px",
        "plane_residual_pct_scene_diag": "green < 1%, yellow < 2.5%",
        "orthogonality_angle_residual_median": "green < 5 deg, yellow < 10 deg",
        "note": "These are engineering gates, not universal benchmarks; thresholds should be tightened/loosened by scene scale and capture difficulty.",
    }
    return metrics


def _images_to_hwc(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _sampson_residual_px(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    ext_i: np.ndarray,
    ext_j: np.ndarray,
    k_i: np.ndarray,
    k_j: np.ndarray,
) -> np.ndarray:
    r_i, t_i = ext_i[:3, :3], ext_i[:3, 3]
    r_j, t_j = ext_j[:3, :3], ext_j[:3, 3]
    rel_r = r_j @ r_i.T
    rel_t = t_j - rel_r @ t_i
    essential = _skew(rel_t) @ rel_r
    fundamental = np.linalg.inv(k_j).T @ essential @ np.linalg.inv(k_i)

    ones = np.ones((len(pts_i), 1), dtype=np.float64)
    x1 = np.concatenate([pts_i, ones], axis=1)
    x2 = np.concatenate([pts_j, ones], axis=1)
    fx1 = (fundamental @ x1.T).T
    ftx2 = (fundamental.T @ x2.T).T
    numerator = np.sum(x2 * fx1, axis=1) ** 2
    denominator = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _gate(value: float | None, green: float, yellow: float, lower_is_better: bool) -> str:
    if value is None:
        return "n/a"
    if lower_is_better:
        if value < green:
            return "green"
        if value < yellow:
            return "yellow"
        return "red"
    if value > green:
        return "green"
    if value > yellow:
        return "yellow"
    return "red"
