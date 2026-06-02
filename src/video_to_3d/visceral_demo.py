from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from video_to_3d.semantics import DEFAULT_CLASSES, class_maps, images_from_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a visceral MP4 demo for the reconstruction.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-points", type=int, default=90_000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = args.out or (args.run_dir / "visceral_demo.mp4")

    with np.load(args.run_dir / "predictions.npz") as loaded:
        predictions = {key: np.array(loaded[key]) for key in loaded.files}
    images = images_from_predictions(predictions)
    mean_conf = predictions["depth_conf"].reshape(len(images), -1).mean(axis=1)
    point_counts = np.isfinite(predictions["world_points_from_depth"]).all(axis=-1).reshape(len(images), -1).sum(axis=1)

    geom_points, geom_colors = load_point_cloud(args.run_dir / "pointcloud.ply", args.max_points)
    semantic_path = args.run_dir / "semantics" / "semantic_pointcloud.ply"
    if semantic_path.exists():
        sem_points, sem_colors = load_point_cloud(semantic_path, args.max_points)
    else:
        sem_points, sem_colors = geom_points, geom_colors

    no_gt_path = args.run_dir / "no_gt_eval_metrics.json"
    no_gt = json.loads(no_gt_path.read_text()) if no_gt_path.exists() else {}
    semantic_summary_path = args.run_dir / "semantics" / "semantic_summary.json"
    semantic_summary = json.loads(semantic_summary_path.read_text()) if semantic_summary_path.exists() else {}

    width, height = 1280, 720
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out}")

    for t in range(args.frames):
        frame_idx = int(round(t / max(args.frames - 1, 1) * (len(images) - 1)))
        stage = min(2, int(t / max(args.frames, 1) * 3))
        angle = 2.0 * np.pi * t / max(args.frames, 1)
        canvas = np.full((height, width, 3), 18, dtype=np.uint8)

        draw_input_panel(canvas, images[frame_idx], frame_idx)
        if stage == 0:
            title = "Step 1: sampled video frames"
            subtitle = "ffmpeg orientation fix -> VGGT input tensor"
            pts, cols = geom_points, geom_colors
            color_mode = "RGB geometry"
        elif stage == 1:
            title = "Step 2: VGGT-Omega geometry"
            subtitle = "depth + camera poses -> dense 3D point cloud"
            pts, cols = geom_points, geom_colors
            color_mode = "8M dense PLY"
        else:
            title = "Step 3: geometry-aligned semantics"
            subtitle = "Grounded-SAM masks lifted through VGGT points"
            pts, cols = sem_points, sem_colors
            color_mode = "semantic labels"

        render = render_points(pts, cols, 600, 620, angle)
        canvas[60:680, 330:930] = render
        draw_text(canvas, title, (340, 32), scale=0.72, color=(255, 255, 255), thickness=2)
        draw_text(canvas, subtitle, (340, 55), scale=0.48, color=(190, 210, 230), thickness=1)
        draw_text(canvas, color_mode, (340, 700), scale=0.50, color=(190, 190, 190), thickness=1)
        draw_metrics_panel(canvas, mean_conf, point_counts, frame_idx, no_gt, semantic_summary)
        writer.write(canvas)

    writer.release()
    print(f"[done] wrote {out}")


def load_point_cloud(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = (np.asarray(pcd.colors, dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
    if len(points) > max_points:
        rng = np.random.default_rng(11)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points, colors


def render_points(points: np.ndarray, colors: np.ndarray, width: int, height: int, angle: float) -> np.ndarray:
    canvas = np.full((height, width, 3), 10, dtype=np.uint8)
    if len(points) == 0:
        return canvas
    pts = points - np.median(points, axis=0, keepdims=True)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    pts = pts @ rot.T
    extent = np.percentile(np.linalg.norm(pts[:, :2], axis=1), 98)
    scale = 0.44 * min(width, height) / max(extent, 1e-6)
    x = (width * 0.5 + pts[:, 0] * scale).astype(np.int32)
    y = (height * 0.52 - pts[:, 1] * scale).astype(np.int32)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, z, col = x[inside], y[inside], pts[inside, 2], colors[inside]
    order = np.argsort(z)
    canvas[y[order], x[order]] = col[order][:, ::-1]
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas


def draw_input_panel(canvas: np.ndarray, image_rgb: np.ndarray, frame_idx: int) -> None:
    panel = cv2.resize(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), (280, 500), interpolation=cv2.INTER_AREA)
    canvas[95:595, 25:305] = panel
    cv2.rectangle(canvas, (25, 95), (305, 595), (230, 230, 230), 1)
    draw_text(canvas, "Input video", (25, 42), scale=0.72, color=(255, 255, 255), thickness=2)
    draw_text(canvas, f"sampled frame {frame_idx:02d}", (25, 68), scale=0.50, color=(190, 210, 230), thickness=1)


def draw_metrics_panel(
    canvas: np.ndarray,
    mean_conf: np.ndarray,
    point_counts: np.ndarray,
    frame_idx: int,
    no_gt: dict,
    semantic_summary: dict,
) -> None:
    x0, y0, w, h = 955, 70, 300, 610
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (38, 38, 38), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1)
    draw_text(canvas, "Live metrics", (x0 + 18, y0 + 36), scale=0.70, color=(255, 255, 255), thickness=2)
    draw_text(canvas, f"mean depth confidence: {mean_conf[frame_idx]:.2f}", (x0 + 18, y0 + 74), scale=0.45)
    draw_text(canvas, f"points from frame: {int(point_counts[frame_idx]):,}", (x0 + 18, y0 + 98), scale=0.45)

    draw_sparkline(canvas, mean_conf, frame_idx, (x0 + 18, y0 + 122), (w - 36, 70), "confidence")
    draw_sparkline(canvas, point_counts / max(point_counts.max(), 1), frame_idx, (x0 + 18, y0 + 222), (w - 36, 70), "points added")

    gates = no_gt.get("quality_gates", {})
    draw_text(canvas, f"coherence score: {gates.get('coherence_score_0_to_1', 0):.2f}/1.00", (x0 + 18, y0 + 334), scale=0.48)
    rows = [
        ("multi-view depth", gates.get("multi_view_depth", "n/a")),
        ("feature reproj", gates.get("feature_reprojection", "n/a")),
        ("plane residual", gates.get("plane_residual", "n/a")),
        ("Manhattan", gates.get("manhattan_orthogonality", "n/a")),
    ]
    yy = y0 + 366
    for name, status in rows:
        color = {"green": (80, 220, 120), "yellow": (80, 210, 235), "red": (80, 90, 235)}.get(status, (180, 180, 180))
        draw_text(canvas, f"{name}: {status}", (x0 + 18, yy), scale=0.42, color=color)
        yy += 24

    summary = semantic_summary.get("summary", {})
    detected = [v["name"] for v in summary.values() if v.get("points", 0) > 0]
    draw_text(canvas, "semantic labels", (x0 + 18, y0 + 490), scale=0.48, color=(255, 255, 255), thickness=1)
    point_total = semantic_summary.get("num_semantic_points", 0)
    draw_text(canvas, f"{len(detected)} classes, {point_total/1_000_000:.2f}M lifted points", (x0 + 18, y0 + 518), scale=0.36, color=(210, 210, 210))
    for row, start in enumerate(range(0, min(len(detected), 16), 4)):
        draw_text(canvas, ", ".join(detected[start : start + 4]), (x0 + 18, y0 + 540 + row * 20), scale=0.32, color=(210, 210, 210))


def draw_sparkline(canvas: np.ndarray, values: np.ndarray, idx: int, origin: tuple[int, int], size: tuple[int, int], label: str) -> None:
    x0, y0 = origin
    w, h = size
    vals = values.astype(np.float32)
    vals = (vals - vals.min()) / max(float(vals.max() - vals.min()), 1e-6)
    pts = []
    for i, val in enumerate(vals):
        x = x0 + int(i / max(len(vals) - 1, 1) * w)
        y = y0 + h - int(val * h)
        pts.append((x, y))
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (120, 210, 255), 2)
    marker = pts[min(idx, len(pts) - 1)]
    cv2.circle(canvas, marker, 4, (80, 255, 160), -1)
    draw_text(canvas, label, (x0, y0 - 8), scale=0.36, color=(200, 200, 200))


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.45,
    color: tuple[int, int, int] = (220, 220, 220),
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    main()
