"""Render a fixed 3/4 overview of a PLY point cloud for side-by-side stage comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from video_to_3d.reconstruction_player import load_ordered_ply_xyzrgb, make_view, render_cloud


def render_one(ply: Path, azim: float, elev: float, max_points: int, size: int, point_radius: int, label: str) -> np.ndarray:
    pts, cols, _, _ = load_ordered_ply_xyzrgb(ply, max_points)
    c = np.median(pts, axis=0).astype(np.float32)
    r = float(np.percentile(np.linalg.norm(pts - c, axis=1), 94))
    a, e = np.radians(azim), np.radians(elev)
    direction = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], dtype=np.float32)
    eye = c + direction * r * 2.2
    view = make_view(c, eye)
    canvas = np.full((size, size, 3), 12, dtype=np.uint8)
    render_cloud(canvas, pts, cols, view, c, r, point_radius=point_radius)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(pts):,} pts shown", (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(pts):,} pts shown", (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--azim", type=float, default=35.0)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--max-points", type=int, default=800_000)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--point-radius", type=int, default=1)
    p.add_argument("--label", type=str, default="")
    args = p.parse_args()
    img = render_one(args.ply, args.azim, args.elev, args.max_points, args.size, args.point_radius, args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
