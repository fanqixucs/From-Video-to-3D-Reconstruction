from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def tensor_predictions_to_numpy(predictions: dict) -> dict[str, np.ndarray]:
    import torch

    predictions_np: dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value
    return predictions_np


def save_predictions(predictions: dict[str, np.ndarray], output_path: Path, compressed: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(output_path, **predictions)
    else:
        np.savez(output_path, **predictions)
    return output_path


def export_glb(
    predictions: dict[str, np.ndarray],
    output_path: Path,
    target_dir: Path,
    conf_thres: float = 50.0,
    max_points: int = 1_000_000,
    show_cam: bool = True,
) -> Path:
    from visual_util import predictions_to_glb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=show_cam,
        mask_sky=False,
        target_dir=str(target_dir),
        max_points=max_points,
    )
    scene.export(file_obj=str(output_path))
    return output_path


def export_ply(
    predictions: dict[str, np.ndarray],
    output_path: Path,
    conf_thres: float = 50.0,
    max_points: int = 1_000_000,
    align_to_first_camera: bool = True,
    filter_depth_edges: bool = True,
) -> tuple[Path, int]:
    vertices, colors = point_cloud_arrays(
        predictions,
        conf_thres=conf_thres,
        max_points=max_points,
        align_to_first_camera=align_to_first_camera,
        filter_depth_edges=filter_depth_edges,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_ply(output_path, vertices, colors)
    return output_path, int(len(vertices))


def point_cloud_arrays(
    predictions: dict[str, np.ndarray],
    conf_thres: float = 50.0,
    max_points: int = 1_000_000,
    align_to_first_camera: bool = True,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    conf_thres = max(2.0, float(conf_thres))

    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[_depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0

    vertices = points.reshape(-1, 3)
    colors = _images_to_rgb(predictions["images"]).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(conf[mask], conf_thres)
        mask &= conf >= conf_threshold
    mask &= conf > 1e-5

    vertices = vertices[mask]
    colors = colors[mask]
    vertices, colors = _limit_points(vertices, colors, max_points)

    if align_to_first_camera and len(vertices) > 0:
        alignment = np.linalg.inv(_extrinsics_4x4(predictions["extrinsic"])[0]) @ _opengl_conversion_matrix()
        vertices = _transform_points(alignment, vertices)

    return vertices.astype(np.float32, copy=False), colors


def write_camera_json(extrinsic: np.ndarray, intrinsic: np.ndarray, output_path: Path) -> Path:
    centers = camera_centers_from_extrinsic(extrinsic)
    payload = {
        "coordinate_convention": "extrinsic is world_to_camera; camera_center_world = -R^T t",
        "num_cameras": int(len(extrinsic)),
        "camera_centers_world": centers.tolist(),
        "extrinsics_world_to_camera": extrinsic.tolist(),
        "intrinsics": intrinsic.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def camera_centers_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation)


def summarize_predictions(predictions: dict[str, np.ndarray]) -> dict:
    depth = predictions["depth"][..., 0]
    conf = predictions["depth_conf"]
    world_points = predictions["world_points_from_depth"]
    centers = camera_centers_from_extrinsic(predictions["extrinsic"])
    path_deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.array([])

    finite_points = np.isfinite(world_points).all(axis=-1)
    finite_depth = np.isfinite(depth)

    return {
        "num_frames": int(depth.shape[0]),
        "image_shape_hw": [int(depth.shape[1]), int(depth.shape[2])],
        "raw_point_count": int(np.prod(depth.shape)),
        "finite_point_count": int(finite_points.sum()),
        "depth_min_p02": _percentile(depth[finite_depth], 2),
        "depth_median": _percentile(depth[finite_depth], 50),
        "depth_max_p98": _percentile(depth[finite_depth], 98),
        "confidence_p10": _percentile(conf[np.isfinite(conf)], 10),
        "confidence_median": _percentile(conf[np.isfinite(conf)], 50),
        "confidence_p90": _percentile(conf[np.isfinite(conf)], 90),
        "camera_path_length_world_units": float(path_deltas.sum()) if path_deltas.size else 0.0,
    }


def write_depth_preview(predictions: dict[str, np.ndarray], output_path: Path) -> Path:
    import cv2
    from matplotlib import colormaps

    depth = predictions["depth"][..., 0]
    frame_indices = _preview_indices(depth.shape[0])
    tiles = []
    cmap = colormaps.get_cmap("magma")

    for idx in frame_indices:
        tile = depth[idx]
        finite = np.isfinite(tile)
        if finite.any():
            lo, hi = np.percentile(tile[finite], [2, 98])
        else:
            lo, hi = 0.0, 1.0
        normalized = np.clip((tile - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        tiles.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    canvas = np.full((height, width * len(tiles), 3), 0, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        canvas[: tile.shape[0], idx * width : idx * width + tile.shape[1]] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"Failed to write depth preview: {output_path}")
    return output_path


def _preview_indices(num_frames: int) -> list[int]:
    if num_frames <= 3:
        return list(range(num_frames))
    return [0, num_frames // 2, num_frames - 1]


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _write_binary_ply(output_path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data["x"] = vertices[:, 0]
    vertex_data["y"] = vertices[:, 1]
    vertex_data["z"] = vertices[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")

    with output_path.open("wb") as f:
        f.write(header)
        f.write(vertex_data.tobytes())


def _images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _limit_points(vertices: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[indices], colors[indices]


def _depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def _extrinsics_4x4(extrinsic: np.ndarray) -> np.ndarray:
    extrinsics = np.zeros((len(extrinsic), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = extrinsic
    extrinsics[:, 3, 3] = 1.0
    return extrinsics


def _opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def _transform_points(transformation: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    transformed = homogeneous @ transformation.T
    return transformed[:, :3]
