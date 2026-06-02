from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from video_to_3d.export import (
    _extrinsics_4x4,
    _images_to_rgb,
    _opengl_conversion_matrix,
    _transform_points,
)
from video_to_3d.semantics import images_from_predictions


WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
DEFAULT_FORWARD = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a VGGT-style reconstruction player and MP4.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-mp4", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=192)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-points", type=int, default=750_000)
    parser.add_argument("--final-cloud", type=Path, default=None)
    parser.add_argument("--conf-percentile", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--mode",
        choices=("doorway", "flythrough"),
        default="doorway",
        help="doorway: fixed entrance view with fusion. flythrough: perspective camera that "
        "follows the trajectory and reveals the cloud from the camera's point of view.",
    )
    parser.add_argument(
        "--skip-player",
        action="store_true",
        help="Skip writing the browser player HTML/points.bin (lets the MP4 use the full cloud "
        "without emitting a multi-hundred-MB bin).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_mp4 = args.out_mp4 or (args.run_dir / "visceral_demo.mp4")
    out_html = args.out_html or (args.run_dir / "reconstruction_player.html")
    data_dir = args.data_dir or (args.run_dir / "reconstruction_player_data")

    with np.load(args.run_dir / "predictions.npz") as loaded:
        predictions = {key: np.array(loaded[key]) for key in loaded.files}

    final_cloud = args.final_cloud or choose_final_cloud(args.run_dir)
    if final_cloud is not None:
        print(f"[demo] using final point cloud: {final_cloud}", flush=True)
        progressive = build_progressive_points_from_ply(
            final_cloud,
            num_frames=len(predictions["extrinsic"]),
            max_points=args.max_points,
        )
    else:
        print("[demo] final PLY not found; falling back to predictions.npz points", flush=True)
        progressive = build_progressive_points_from_predictions(
            predictions,
            max_points=args.max_points,
            conf_percentile=args.conf_percentile,
        )
    cameras = build_camera_geometry(predictions)
    if args.skip_player:
        print("[demo] skipping browser player assets (--skip-player)", flush=True)
    else:
        write_player_assets(args.run_dir, data_dir, out_html, progressive, cameras, final_cloud)
    if args.mode == "flythrough":
        write_flythrough_video(
            args.run_dir,
            out_mp4,
            predictions,
            progressive,
            cameras,
            frames=args.frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
    else:
        write_story_video(
            args.run_dir,
            out_mp4,
            predictions,
            progressive,
            cameras,
            frames=args.frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
    print(f"[done] wrote {out_mp4}")
    print(f"[done] wrote {out_html}")


def choose_final_cloud(run_dir: Path) -> Path | None:
    for name in ("pointcloud_full.ply", "pointcloud.ply", "pointcloud_dense.ply"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def build_progressive_points_from_ply(
    path: Path,
    num_frames: int,
    max_points: int,
) -> dict[str, np.ndarray]:
    points, colors, source_indices, total_vertices = load_ordered_ply_xyzrgb(path, max_points=max_points)
    if total_vertices <= 0:
        frame_ids = np.empty((0,), dtype=np.int32)
    else:
        frame_ids = np.floor(source_indices.astype(np.float64) / total_vertices * num_frames).astype(np.int32)
        frame_ids = np.clip(frame_ids, 0, max(num_frames - 1, 0))
    order = np.argsort(frame_ids, kind="stable")
    points = points[order]
    colors = colors[order]
    frame_ids = frame_ids[order]
    cumulative = np.searchsorted(frame_ids, np.arange(num_frames, dtype=np.int32) + 1, side="left").astype(np.int32)
    return {
        "points": points,
        "colors": colors,
        "frame_ids": frame_ids,
        "cumulative": cumulative,
        "source": np.array(str(path)),
    }


def build_progressive_points_from_predictions(
    predictions: dict[str, np.ndarray],
    max_points: int,
    conf_percentile: float,
) -> dict[str, np.ndarray]:
    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    colors = (_images_to_rgb(predictions["images"]) * 255).clip(0, 255).astype(np.uint8)
    alignment = scene_alignment(predictions["extrinsic"])

    num_frames = points.shape[0]
    per_frame = max(1, int(np.ceil(max_points / max(num_frames, 1))))
    rng = np.random.default_rng(2026)

    sampled_points = []
    sampled_colors = []
    sampled_frame_ids = []
    cumulative = []
    total = 0

    for frame_idx in range(num_frames):
        p = points[frame_idx].reshape(-1, 3)
        c = colors[frame_idx].reshape(-1, 3)
        q = conf[frame_idx].reshape(-1)
        finite = np.isfinite(p).all(axis=1) & np.isfinite(q) & (q > 1e-5)
        if finite.any():
            threshold = np.percentile(q[finite], conf_percentile)
            candidates = np.flatnonzero(finite & (q >= threshold))
        else:
            candidates = np.array([], dtype=np.int64)
        if candidates.size > per_frame:
            candidates = rng.choice(candidates, per_frame, replace=False)
        if candidates.size:
            sampled_points.append(_transform_points(alignment, p[candidates]).astype(np.float32))
            sampled_colors.append(c[candidates])
            sampled_frame_ids.append(np.full(candidates.size, frame_idx, dtype=np.int32))
            total += int(candidates.size)
        cumulative.append(total)

    if sampled_points:
        all_points = np.concatenate(sampled_points, axis=0)
        all_colors = np.concatenate(sampled_colors, axis=0)
        frame_ids = np.concatenate(sampled_frame_ids, axis=0)
    else:
        all_points = np.empty((0, 3), dtype=np.float32)
        all_colors = np.empty((0, 3), dtype=np.uint8)
        frame_ids = np.empty((0,), dtype=np.int32)

    return {
        "points": all_points,
        "colors": all_colors,
        "frame_ids": frame_ids,
        "cumulative": np.array(cumulative, dtype=np.int32),
        "source": np.array("predictions.npz"),
    }


def load_ordered_ply_xyzrgb(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    metadata = parse_binary_ply_header(path)
    vertex_count = metadata["vertex_count"]
    dtype = metadata["dtype"]
    offset = metadata["offset"]
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = required.difference(dtype.names or ())
    if missing:
        raise ValueError(f"{path} is missing PLY fields: {sorted(missing)}")

    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(vertex_count,))
    if max_points > 0 and vertex_count > max_points:
        source_indices = np.linspace(0, vertex_count - 1, max_points).astype(np.int64)
        sampled = data[source_indices]
    else:
        source_indices = np.arange(vertex_count, dtype=np.int64)
        sampled = data

    points = np.stack([sampled["x"], sampled["y"], sampled["z"]], axis=1).astype(np.float32, copy=False)
    colors = np.stack([sampled["red"], sampled["green"], sampled["blue"]], axis=1).astype(np.uint8, copy=False)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite], source_indices[finite], int(vertex_count)


def parse_binary_ply_header(path: Path) -> dict:
    type_map = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    properties = []
    vertex_count = None
    in_vertex = False
    offset = 0
    with path.open("rb") as f:
        first = f.readline()
        offset += len(first)
        if first.strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise ValueError(f"{path} has no end_header")
            offset += len(line_bytes)
            line = line_bytes.decode("ascii").strip()
            if line == "format binary_little_endian 1.0":
                continue
            if line.startswith("format ") and line != "format binary_little_endian 1.0":
                raise ValueError(f"Only binary_little_endian PLY is supported, got: {line}")
            if line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if line.startswith("property ") and in_vertex:
                parts = line.split()
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(f"Unsupported vertex property in {path}: {line}")
                if parts[1] not in type_map:
                    raise ValueError(f"Unsupported PLY property type in {path}: {parts[1]}")
                properties.append((parts[2], type_map[parts[1]]))
                continue
            if line == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"{path} has no vertex element")
    return {"vertex_count": vertex_count, "dtype": np.dtype(properties), "offset": offset}


def build_camera_geometry(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    extrinsic = predictions["extrinsic"]
    intrinsic = predictions["intrinsic"]
    depth = predictions["depth"][..., 0]
    height, width = depth.shape[1:3]
    alignment = scene_alignment(extrinsic)
    extrinsics = _extrinsics_4x4(extrinsic)
    c2w = np.linalg.inv(extrinsics)

    centers = _transform_points(alignment, c2w[:, :3, 3])
    first_axes = alignment[:3, :3] @ c2w[0, :3, :3]
    entrance_right = normalize_vector(first_axes @ np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    entrance_up = normalize_vector(first_axes @ np.array([0.0, -1.0, 0.0], dtype=np.float32), WORLD_UP)
    entrance_forward = normalize_vector(first_axes @ np.array([0.0, 0.0, 1.0], dtype=np.float32), DEFAULT_FORWARD)
    z = float(np.nanmedian(depth)) * 0.18
    segments = []

    for idx in range(len(extrinsic)):
        k = intrinsic[idx]
        corners_px = np.array(
            [
                [0.0, 0.0],
                [float(width), 0.0],
                [float(width), float(height)],
                [0.0, float(height)],
            ],
            dtype=np.float64,
        )
        x = (corners_px[:, 0] - k[0, 2]) / k[0, 0] * z
        y = (corners_px[:, 1] - k[1, 2]) / k[1, 1] * z
        camera_corners = np.stack([x, y, np.full(4, z)], axis=1)
        world_corners = _transform_points(c2w[idx], camera_corners)
        scene_corners = _transform_points(alignment, world_corners)
        center = centers[idx]
        frame_segments = []
        for corner in scene_corners:
            frame_segments.append([center, corner])
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            frame_segments.append([scene_corners[a], scene_corners[b]])
        segments.append(np.asarray(frame_segments, dtype=np.float32))

    return {
        "centers": centers.astype(np.float32),
        "segments": np.asarray(segments, dtype=np.float32),
        "entrance_right": entrance_right.astype(np.float32),
        "entrance_up": entrance_up.astype(np.float32),
        "entrance_forward": entrance_forward.astype(np.float32),
    }


def write_player_assets(
    run_dir: Path,
    data_dir: Path,
    out_html: Path,
    progressive: dict[str, np.ndarray],
    cameras: dict[str, np.ndarray],
    final_cloud: Path | None,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    points = progressive["points"].astype(np.float32)
    colors = (progressive["colors"].astype(np.float32) / 255.0).astype(np.float32)
    frame_ids = progressive["frame_ids"].astype(np.float32)[:, None]
    packed = np.concatenate([points, colors, frame_ids], axis=1).astype("<f4", copy=False)
    (data_dir / "points.bin").write_bytes(packed.tobytes())

    centers = cameras["centers"]
    center, radius = scene_center_radius(points, centers)
    travel_axis = travel_axis_from_centers(centers)
    finite = np.isfinite(points).all(axis=1)
    if finite.any():
        bounds_min = points[finite].min(axis=0)
        bounds_max = points[finite].max(axis=0)
    else:
        bounds_min = np.array([-1, -1, -1], dtype=np.float32)
        bounds_max = np.array([1, 1, 1], dtype=np.float32)
    metadata = {
        "num_frames": int(len(progressive["cumulative"])),
        "num_points": int(len(points)),
        "point_stride": 7,
        "points_bin": str((data_dir / "points.bin").relative_to(out_html.parent)),
        "frames_dir": str((run_dir / "frames").relative_to(out_html.parent)),
        "source_cloud": str(final_cloud.relative_to(run_dir)) if final_cloud is not None and final_cloud.is_relative_to(run_dir) else str(final_cloud or "predictions.npz"),
        "cumulative": progressive["cumulative"].astype(int).tolist(),
        "camera_centers": centers.astype(float).tolist(),
        "camera_segments": cameras["segments"].astype(float).tolist(),
        "entrance_right": cameras["entrance_right"].astype(float).tolist(),
        "entrance_up": cameras["entrance_up"].astype(float).tolist(),
        "entrance_forward": cameras["entrance_forward"].astype(float).tolist(),
        "scene_center": center.astype(float).tolist(),
        "scene_radius": float(radius),
        "travel_axis": travel_axis.astype(float).tolist(),
        "world_up": WORLD_UP.astype(float).tolist(),
        "bounds_min": bounds_min.astype(float).tolist(),
        "bounds_max": bounds_max.astype(float).tolist(),
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_player_html(
        out_html,
        metadata_path=str((data_dir / "metadata.json").relative_to(out_html.parent)),
    )


def write_player_html(out_html: Path, metadata_path: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VGGT-Style Reconstruction Player</title>
  <style>
    html, body {{ margin: 0; min-height: 100%; background: #101010; color: #f4f4f4; font-family: Inter, system-ui, sans-serif; }}
    body {{ overflow: hidden; }}
    #app {{ position: fixed; inset: 0; display: grid; grid-template-columns: minmax(270px, 28vw) 1fr; gap: 0; }}
    #input {{ padding: 18px; border-right: 1px solid #333; background: #151515; display: flex; flex-direction: column; min-width: 0; }}
    #input h1, #scene h2 {{ margin: 0 0 8px; font-size: 18px; font-weight: 700; }}
    #input p, #scene p {{ margin: 0; color: #bdbdbd; font-size: 13px; }}
    #frame {{ width: 100%; flex: 1; min-height: 0; margin-top: 18px; object-fit: contain; background: #050505; border: 1px solid #333; }}
    #scene {{ position: relative; min-width: 0; }}
    #scene-title {{ position: absolute; top: 18px; left: 20px; z-index: 2; }}
    #three {{ position: absolute; inset: 0; }}
    #hud {{ position: absolute; left: 20px; right: 20px; bottom: 18px; z-index: 2; display: grid; grid-template-columns: 1fr auto auto; gap: 14px; align-items: center; }}
    #bar {{ height: 4px; background: #333; overflow: hidden; }}
    #fill {{ height: 100%; width: 0%; background: #64d892; }}
    #scrub {{ width: 100%; accent-color: #64d892; }}
    button {{ color: #f4f4f4; background: #242424; border: 1px solid #555; padding: 8px 12px; cursor: pointer; }}
    button:hover {{ background: #303030; }}
  </style>
</head>
<body>
  <div id="app">
    <section id="input">
      <h1>Input Video</h1>
      <p>Selected key frames from the video.</p>
      <img id="frame" src="" alt="current input frame">
    </section>
    <section id="scene">
      <div id="scene-title">
        <h2>Reconstruction</h2>
        <p>Doorway view from the room entrance; points are fused only up to the current video frame.</p>
      </div>
      <div id="three"></div>
      <div id="hud">
        <div><input id="scrub" type="range" min="0" value="0" step="1"><div id="bar"><div id="fill"></div></div></div>
        <button id="reset">Reset View</button>
        <button id="toggle">Pause</button>
      </div>
    </section>
  </div>
  <script type="importmap">
    {{"imports": {{"three": "https://unpkg.com/three@0.160.0/build/three.module.js"}}}}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

    const meta = await fetch('{metadata_path}').then(r => r.json());
    const raw = await fetch(meta.points_bin).then(r => r.arrayBuffer());
    const packed = new Float32Array(raw);
    const positions = new Float32Array(meta.num_points * 3);
    const colors = new Float32Array(meta.num_points * 3);
    for (let i = 0; i < meta.num_points; i++) {{
      positions.set(packed.subarray(i * 7, i * 7 + 3), i * 3);
      colors.set(packed.subarray(i * 7 + 3, i * 7 + 6), i * 3);
    }}

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101010);
    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    document.getElementById('three').appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    const b0 = new THREE.Vector3(...meta.bounds_min);
    const b1 = new THREE.Vector3(...meta.bounds_max);
    const center = new THREE.Vector3(...meta.scene_center);
    const radius = Math.max(0.4, meta.scene_radius || b0.distanceTo(b1) * 0.55);
    const worldUp = new THREE.Vector3(...meta.world_up).normalize();
    const entranceRight = new THREE.Vector3(...meta.entrance_right).normalize();
    const entranceUp = new THREE.Vector3(...meta.entrance_up).normalize();
    const entranceForward = new THREE.Vector3(...meta.entrance_forward).normalize();
    const centers = meta.camera_centers.map(p => new THREE.Vector3(...p));
    const entranceTarget = center.clone().addScaledVector(entranceUp, radius * 0.04);
    const entranceEye = centers[0].clone()
      .addScaledVector(entranceForward, -radius * 0.88)
      .addScaledVector(entranceUp, radius * 0.18);
    const pointSize = Math.max(0.0025, Math.min(0.012, radius * 0.0065));

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setDrawRange(0, 0);
    const cloud = new THREE.Points(geometry, new THREE.PointsMaterial({{ size: pointSize, vertexColors: true }}));
    scene.add(cloud);

    const freshGeometry = new THREE.BufferGeometry();
    freshGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    freshGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    freshGeometry.setDrawRange(0, 0);
    const freshCloud = new THREE.Points(
      freshGeometry,
      new THREE.PointsMaterial({{ size: pointSize * 1.75, vertexColors: true, transparent: true, opacity: 0.96 }})
    );
    scene.add(freshCloud);

    const pathGeom = new THREE.BufferGeometry().setFromPoints(centers);
    scene.add(new THREE.Line(pathGeom, new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.28 }})));

    const travelledPathGeom = new THREE.BufferGeometry().setFromPoints(centers);
    travelledPathGeom.setDrawRange(0, 1);
    scene.add(new THREE.Line(travelledPathGeom, new THREE.LineBasicMaterial({{ color: 0x64d892, transparent: true, opacity: 0.95 }})));

    const currentFrustum = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({{ color: 0x64d892 }})
    );
    scene.add(currentFrustum);
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    const frameImg = document.getElementById('frame');
    const fill = document.getElementById('fill');
    const toggle = document.getElementById('toggle');
    const reset = document.getElementById('reset');
    const scrub = document.getElementById('scrub');
    scrub.max = String(Math.max(0, meta.num_frames - 1));
    let playing = true;
    let frameIdx = 0;
    let last = performance.now();
    let manualInspect = false;
    toggle.onclick = () => {{
      if (!playing && frameIdx >= meta.num_frames - 1) {{
        setFrame(0, true);
      }}
      playing = !playing;
      toggle.textContent = playing ? 'Pause' : 'Play';
    }};
    reset.onclick = () => {{
      manualInspect = false;
      setFrame(frameIdx, true);
    }};
    controls.addEventListener('start', () => {{
      manualInspect = true;
    }});
    scrub.oninput = () => {{
      playing = false;
      toggle.textContent = 'Play';
      setFrame(Number(scrub.value), true);
    }};

    function followCamera(idx, force = false) {{
      if (manualInspect && !force) return;
      camera.up.copy(entranceUp);
      camera.position.copy(entranceEye);
      camera.lookAt(entranceTarget);
      controls.target.copy(entranceTarget);
    }}

    function setFrame(idx, forceCamera = false) {{
      frameIdx = Math.max(0, Math.min(meta.num_frames - 1, idx));
      const drawCount = meta.cumulative[frameIdx];
      const start = frameIdx > 0 ? meta.cumulative[frameIdx - 1] : 0;
      geometry.setDrawRange(0, drawCount);
      freshGeometry.setDrawRange(start, Math.max(0, drawCount - start));
      travelledPathGeom.setDrawRange(0, frameIdx + 1);
      frameImg.src = `${{meta.frames_dir}}/${{String(frameIdx).padStart(6, '0')}}.jpg`;
      fill.style.width = `${{100 * (frameIdx + 1) / meta.num_frames}}%`;
      scrub.value = String(frameIdx);
      const segments = meta.camera_segments[frameIdx].flat(2);
      currentFrustum.geometry.dispose();
      currentFrustum.geometry = new THREE.BufferGeometry();
      currentFrustum.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segments), 3));
      followCamera(frameIdx, forceCamera);
    }}

    function resize() {{
      const host = document.getElementById('three');
      const w = host.clientWidth;
      const h = host.clientHeight;
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }}
    window.addEventListener('resize', resize);
    resize();
    setFrame(0);

    function animate(now) {{
      requestAnimationFrame(animate);
      if (playing && now - last > 95) {{
        if (frameIdx >= meta.num_frames - 1) {{
          playing = false;
          toggle.textContent = 'Replay';
        }} else {{
          setFrame(frameIdx + 1);
        }}
        last = now;
      }}
      controls.update();
      renderer.render(scene, camera);
    }}
    animate(performance.now());
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def write_story_video(
    run_dir: Path,
    out_path: Path,
    predictions: dict[str, np.ndarray],
    progressive: dict[str, np.ndarray],
    cameras: dict[str, np.ndarray],
    frames: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    images = images_from_predictions(predictions)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    points = progressive["points"]
    colors = progressive["colors"]
    frame_ids = progressive["frame_ids"]
    center, radius = scene_center_radius(points, cameras["centers"])

    input_x, input_y = 32, 102
    input_w, input_h = 292, 520
    view_x, view_y = 360, 48
    view_w, view_h = width - view_x - 28, height - 92
    view, focus, view_radius = entrance_view(cameras, center, radius)

    for t in range(frames):
        alpha = t / max(frames - 1, 1)
        source_idx = int(round(alpha * (len(images) - 1)))
        canvas = np.full((height, width, 3), 16, dtype=np.uint8)

        draw_input_video(canvas, images[source_idx], source_idx, len(images), input_x, input_y, input_w, input_h)

        viewport = canvas[view_y : view_y + view_h, view_x : view_x + view_w]
        viewport[:] = (8, 8, 8)
        active = frame_ids <= source_idx
        fresh = frame_ids == source_idx
        render_cloud(viewport, points[active], colors[active], view, focus, view_radius)
        render_cloud(
            viewport,
            points[fresh],
            colors[fresh],
            view,
            focus,
            view_radius,
            point_radius=2,
        )
        draw_camera_path(viewport, cameras, source_idx, view, focus, view_radius)

        draw_text(canvas, "Input Video", (input_x, 48), scale=0.78, color=(255, 255, 255), thickness=2)
        draw_text(canvas, "selected key frames", (input_x, 76), scale=0.44, color=(190, 196, 205))
        draw_text(canvas, "Reconstruction", (view_x, 32), scale=0.78, color=(255, 255, 255), thickness=2)
        draw_text(canvas, "doorway view + frame-synchronized point fusion", (view_x + 184, 32), scale=0.44, color=(190, 196, 205))
        draw_progress(canvas, source_idx, len(images), points_added=int(active.sum()), origin=(view_x, height - 34), width=view_w)
        writer.write(canvas)

        if t == frames // 2:
            preview_path = out_path.with_name(out_path.stem + "_preview.jpg")
            cv2.imwrite(str(preview_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    writer.release()


def camera_orientations(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame camera forward and up axes, expressed in the aligned scene frame."""
    extrinsic = predictions["extrinsic"]
    alignment = scene_alignment(extrinsic)
    c2w = np.linalg.inv(_extrinsics_4x4(extrinsic))
    rot = alignment[:3, :3] @ c2w[:, :3, :3]  # (N, 3, 3): camera axes in scene frame
    forward = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)  # camera looks down +Z
    up = rot @ np.array([0.0, -1.0, 0.0], dtype=np.float64)  # image y points down
    forward = forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
    up = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    return forward.astype(np.float32), up.astype(np.float32)


def smooth_track(values: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing along axis 0 with edge padding (keeps shape)."""
    if window <= 1 or len(values) <= 2:
        return values.astype(np.float32)
    window = min(window, len(values))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(values.shape[1])], axis=1)
    return out.astype(np.float32)


def interp_track(track: np.ndarray, fidx: float) -> np.ndarray:
    """Linearly interpolate a (N, 3) track at a continuous frame index."""
    n = len(track)
    if n == 0:
        return np.zeros(3, dtype=np.float32)
    lo = int(np.floor(fidx))
    lo = max(0, min(lo, n - 1))
    hi = min(lo + 1, n - 1)
    frac = float(fidx - lo)
    return ((1.0 - frac) * track[lo] + frac * track[hi]).astype(np.float32)


def flythrough_view(
    pos: np.ndarray,
    forward: np.ndarray,
    radius: float,
    back: float = 0.16,
    rise: float = 0.07,
    look_ahead: float = 0.50,
) -> dict[str, np.ndarray]:
    """A near-first-person camera just behind the moving camera, looking along its forward axis.

    Staying close to the real camera pose keeps the densely-reconstructed forward cone filling
    the frame, so sparse side floaters fall outside view. Roll is removed by re-orthogonalizing
    against world up so the flythrough stays level.
    """
    fwd = forward - WORLD_UP * float(forward @ WORLD_UP) * 0.20  # mostly forward, slight level-out
    fwd = normalize_vector(fwd, DEFAULT_FORWARD)
    eye = pos - fwd * radius * back + WORLD_UP * radius * rise
    target = pos + fwd * radius * look_ahead
    return make_view(target, eye)


def perspective_project(
    points: np.ndarray,
    view: dict[str, np.ndarray],
    focal: float,
    width: int,
    height: int,
    near: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world points through a pinhole camera. Returns x, y (px), depth, valid-mask."""
    if len(points) == 0:
        empty_i = np.empty((0,), dtype=np.int32)
        return empty_i, empty_i, np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool)
    rel = points.astype(np.float32) - view["eye"][None]
    xc = rel @ view["right"]
    yc = rel @ view["up"]
    zc = rel @ view["forward"]
    safe = zc > near
    z = np.where(safe, zc, 1.0)
    x = (width * 0.5 + focal * xc / z).astype(np.int32)
    y = (height * 0.5 - focal * yc / z).astype(np.int32)
    inside = safe & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x, y, zc, inside


def _disk_offsets(max_radius: int) -> list[tuple[int, int, int]]:
    """(ring, dx, dy) offsets covering a filled disk, ordered outer-ring-first, centre last."""
    offsets = []
    for dy in range(-max_radius, max_radius + 1):
        for dx in range(-max_radius, max_radius + 1):
            ring = int(np.ceil(np.hypot(dx, dy) - 1e-6))
            if ring <= max_radius:
                offsets.append((ring, dx, dy))
    offsets.sort(key=lambda item: -item[0])  # draw outer rings first so centres land on top
    return offsets


def fill_interior_holes(canvas: np.ndarray, bg_sum: int = 36, exterior_frac: float = 0.16) -> None:
    """Inpaint background holes enclosed by geometry (e.g. the sparsely-sampled ceiling).

    Background pixels are grouped into connected components; everything is inpainted from the
    surrounding surface EXCEPT large components that reach the frame border, which are treated as
    true exterior void (open doorways/windows) and left black. Operates in place.
    """
    h, w = canvas.shape[:2]
    background = (canvas.astype(np.int32).sum(axis=2) <= bg_sum).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    if num <= 1:
        return
    fill = np.zeros((h, w), np.uint8)
    area_total = h * w
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches_border and area > exterior_frac * area_total:
            continue  # genuine exterior opening — keep it black
        fill[labels == i] = 1
    if fill.any():
        canvas[:] = cv2.inpaint(canvas, fill, 3, cv2.INPAINT_TELEA)


def render_cloud_pov(
    canvas: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    view: dict[str, np.ndarray],
    focal: float,
    near: float,
    splat_world: float = 0.011,
    max_radius: int = 3,
    radius_cap: np.ndarray | None = None,
    fade_start: float | None = None,
    fade_end: float | None = None,
) -> None:
    """Perspective *surfel* splat: each point covers a depth-scaled disk so close surfaces
    merge into a solid sheet instead of dotted pixels. Painter's order keeps occlusion correct.

    `radius_cap` (aligned with `points`) caps each point's splat by local density, so sparse
    floaters stay 1px (or drop out) instead of being fattened. `fade_start`/`fade_end` apply
    distance fog: points fade to black past `fade_start` and vanish by `fade_end`, so far,
    under-sampled geometry dissolves into the void instead of snowing it.
    """
    if len(points) == 0:
        return
    h, w = canvas.shape[:2]
    x, y, z, inside = perspective_project(points, view, focal, w, h, near)
    x, y, z, cols = x[inside], y[inside], z[inside], colors[inside]
    if len(x) == 0:
        return
    # Pixel radius from a fixed world-space footprint: nearer points splat larger.
    radii = np.clip(np.round(splat_world * focal / np.maximum(z, near)).astype(np.int32), 1, max_radius)
    order = np.argsort(z)[::-1]  # far first so nearer points overwrite
    x, y, radii, bgr, zo = x[order], y[order], radii[order], cols[order][:, ::-1], z[order]
    if fade_end is not None and fade_start is not None and fade_end > fade_start:
        fog = np.clip((fade_end - zo) / (fade_end - fade_start), 0.0, 1.0)
        bgr = (bgr.astype(np.float32) * fog[:, None]).astype(np.uint8)
    if radius_cap is not None:
        radii = np.minimum(radii, radius_cap[inside][order])
        keep = radii >= 1
        x, y, radii, bgr = x[keep], y[keep], radii[keep], bgr[keep]
        if len(x) == 0:
            return
    for ring, dx, dy in _disk_offsets(int(max_radius)):
        if ring == 0:
            tx, ty, c = x + dx, y + dy, bgr
        else:
            sel = radii >= ring
            tx, ty, c = x[sel] + dx, y[sel] + dy, bgr[sel]
        ok = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        canvas[ty[ok], tx[ok]] = c[ok]


def write_flythrough_video(
    run_dir: Path,
    out_path: Path,
    predictions: dict[str, np.ndarray],
    progressive: dict[str, np.ndarray],
    cameras: dict[str, np.ndarray],
    frames: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    images = images_from_predictions(predictions)
    num_src = len(images)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    points = progressive["points"]
    colors = progressive["colors"]
    frame_ids = progressive["frame_ids"]
    _, radius = scene_center_radius(points, cameras["centers"])

    keep = floater_mask(points, radius)
    points, colors, frame_ids = points[keep], colors[keep], frame_ids[keep]
    print(f"[flythrough] kept {int(keep.sum()):,}/{len(keep):,} points after floater cull", flush=True)

    # Per-point splat cap from local density: dense surfaces splat solid, sparse points stay
    # small, lone points drop out. Keeps the flythrough smooth without snowing the void.
    counts = voxel_point_counts(points, voxel=radius * 0.02)
    splat_cap = np.select([counts >= 9, counts >= 4, counts >= 2], [3, 2, 1], default=0).astype(np.int32)
    print(f"[flythrough] splat caps: solid={int((splat_cap==3).sum()):,} mid={int((splat_cap==2).sum()):,} "
          f"thin={int((splat_cap==1).sum()):,} dropped={int((splat_cap==0).sum()):,}", flush=True)

    centers = smooth_track(cameras["centers"], window=9)
    fwd_axes, _ = camera_orientations(predictions)
    fwd_axes = smooth_track(fwd_axes, window=11)

    input_x, input_y = 32, 102
    input_w, input_h = 292, height - input_y - 64
    view_x, view_y = 360, 48
    view_w, view_h = width - view_x - 28, height - 92
    vfov = np.deg2rad(50.0)
    focal = 0.5 * view_h / np.tan(vfov / 2.0)
    near = max(radius * 0.01, 1e-3)

    for t in range(frames):
        alpha = t / max(frames - 1, 1)
        # Mostly-linear timeline with only a gentle ease at the very ends. A pure smoothstep
        # holds on the first source frame for many output frames, which reads as a stutter.
        eased = 0.82 * alpha + 0.18 * smoothstep(alpha)
        fidx = eased * (num_src - 1)
        source_idx = int(round(fidx))

        pos = interp_track(centers, fidx)
        forward = normalize_vector(interp_track(fwd_axes, fidx), DEFAULT_FORWARD)
        view = flythrough_view(pos, forward, radius)

        canvas = np.full((height, width, 3), 16, dtype=np.uint8)
        draw_input_video(canvas, images[source_idx], source_idx, num_src, input_x, input_y, input_w, input_h)

        viewport = canvas[view_y : view_y + view_h, view_x : view_x + view_w]
        viewport[:] = (8, 8, 8)
        active = frame_ids <= source_idx
        fresh = frame_ids == source_idx
        render_cloud_pov(viewport, points[active], colors[active], view, focal, near, splat_world=0.016, max_radius=4, radius_cap=splat_cap[active])
        if fresh.any():
            render_cloud_pov(viewport, points[fresh], brighten(colors[fresh], 1.1), view, focal, near, splat_world=0.016, max_radius=4, radius_cap=splat_cap[fresh])
        # Inpaint holes enclosed by geometry (e.g. the sparsely-sampled ceiling); keep the large
        # exterior void black. Then a light blur knits the splats together.
        fill_interior_holes(viewport)
        viewport[:] = cv2.GaussianBlur(viewport, (3, 3), 0)

        draw_text(canvas, "Input Video", (input_x, 48), scale=0.78, color=(255, 255, 255), thickness=2)
        draw_text(canvas, "selected key frames", (input_x, 76), scale=0.44, color=(190, 196, 205))
        draw_text(canvas, "Reconstruction", (view_x, 32), scale=0.78, color=(255, 255, 255), thickness=2)
        draw_text(canvas, "flythrough from the camera's point of view", (view_x + 184, 32), scale=0.44, color=(190, 196, 205))
        draw_progress(canvas, source_idx, num_src, points_added=int(active.sum()), origin=(view_x, height - 34), width=view_w)
        writer.write(canvas)

        if t == frames // 2:
            preview_path = out_path.with_name(out_path.stem + "_preview.jpg")
            cv2.imwrite(str(preview_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    writer.release()


def floater_mask(points: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask dropping low-confidence floaters / depth-noise fog from a dense cloud.

    Prefers Open3D statistical outlier removal (neighbour-distance based); falls back to a
    pure-numpy voxel density cull when Open3D is unavailable.
    """
    if len(points) == 0:
        return np.ones(0, dtype=bool)
    try:
        import open3d as o3d

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        # Splatting fattens any survivor, so cull harder than for plain point rendering.
        _, kept = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=0.85)
        mask = np.zeros(len(points), dtype=bool)
        mask[np.asarray(kept, dtype=np.int64)] = True
        return mask
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        print(f"[flythrough] open3d outlier removal unavailable ({exc}); using voxel cull", flush=True)
        return voxel_density_mask(points, voxel=radius * 0.012, min_count=3)


def voxel_point_counts(points: np.ndarray, voxel: float) -> np.ndarray:
    """Number of points sharing each point's voxel — a cheap local-density estimate."""
    if len(points) == 0 or voxel <= 0:
        return np.ones(len(points), dtype=np.int32)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.int32)


def voxel_density_mask(points: np.ndarray, voxel: float, min_count: int) -> np.ndarray:
    """Keep points whose occupied voxel holds at least `min_count` points (drops isolated floaters)."""
    if len(points) == 0 or voxel <= 0:
        return np.ones(len(points), dtype=bool)
    return voxel_point_counts(points, voxel) >= min_count


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def brighten(colors: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(colors.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def draw_frustum_perspective(
    canvas: np.ndarray,
    segments: np.ndarray,
    view: dict[str, np.ndarray],
    focal: float,
    near: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    h, w = canvas.shape[:2]
    flat = segments.reshape(-1, 3)
    x, y, _, inside = perspective_project(flat, view, focal, w, h, near)
    pts = np.stack([x, y], axis=1).reshape(-1, 2, 2)
    inside_pairs = inside.reshape(-1, 2)
    for pair, ok in zip(pts, inside_pairs, strict=False):
        if ok.all():
            cv2.line(canvas, tuple(int(v) for v in pair[0]), tuple(int(v) for v in pair[1]), color, thickness, cv2.LINE_AA)


def draw_path_perspective(
    canvas: np.ndarray,
    centers: np.ndarray,
    view: dict[str, np.ndarray],
    focal: float,
    near: float,
) -> None:
    if len(centers) < 2:
        return
    h, w = canvas.shape[:2]
    x, y, _, inside = perspective_project(centers, view, focal, w, h, near)
    pts = np.stack([x, y], axis=1)
    run = []
    for (px, py), ok in zip(pts, inside, strict=False):
        if ok:
            run.append((int(px), int(py)))
        elif len(run) > 1:
            cv2.polylines(canvas, [np.array(run, dtype=np.int32)], False, (235, 235, 235), 2, cv2.LINE_AA)
            run = []
    if len(run) > 1:
        cv2.polylines(canvas, [np.array(run, dtype=np.int32)], False, (235, 235, 235), 2, cv2.LINE_AA)


def scene_alignment(extrinsic: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_extrinsics_4x4(extrinsic)[0]) @ _opengl_conversion_matrix()


def normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def travel_axis_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    finite = np.isfinite(centers).all(axis=1)
    path = centers[finite]
    if len(path) < 2:
        return DEFAULT_FORWARD.copy()

    axis = path[-1] - path[0]
    axis = axis - WORLD_UP * float(axis @ WORLD_UP)
    if float(np.linalg.norm(axis)) < 1e-5:
        centered = path - path.mean(axis=0, keepdims=True)
        centered = centered - np.outer(centered @ WORLD_UP, WORLD_UP)
        if np.any(np.abs(centered) > 1e-6):
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0].astype(np.float32)
            if float(axis @ (path[-1] - path[0])) < 0:
                axis = -axis

    norm = float(np.linalg.norm(axis))
    if norm < 1e-5:
        return DEFAULT_FORWARD.copy()
    return (axis / norm).astype(np.float32)


def scene_center_radius(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points):
        merged = np.concatenate([points, centers], axis=0)
    else:
        merged = centers
    center = np.median(merged, axis=0).astype(np.float32)
    distances = np.linalg.norm(merged - center[None], axis=1)
    radius = float(np.percentile(distances, 94)) if len(distances) else 1.0
    return center, max(radius, 0.35)


def entrance_view(
    cameras: dict[str, np.ndarray],
    scene_center: np.ndarray,
    radius: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    centers = cameras["centers"]
    entry = centers[0].astype(np.float32) if len(centers) else np.zeros(3, dtype=np.float32)
    right = cameras["entrance_right"].astype(np.float32)
    up = cameras["entrance_up"].astype(np.float32)
    forward = cameras["entrance_forward"].astype(np.float32)

    eye = entry - forward * radius * 0.88 + up * radius * 0.18
    target = scene_center.astype(np.float32) + up * radius * 0.04
    view = make_view_with_basis(target, eye, right, up)
    return view, scene_center.astype(np.float32), max(radius * 1.18, 0.35)


def follow_path_view(
    centers: np.ndarray,
    frame_idx: int,
    radius: float,
    fallback_axis: np.ndarray,
    scene_center: np.ndarray | None = None,
    overview: float = 0.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    direction = np.asarray(fallback_axis, dtype=np.float32)
    direction = direction - WORLD_UP * float(direction @ WORLD_UP)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-5:
        direction = DEFAULT_FORWARD.copy()
    else:
        direction = direction / direction_norm

    if len(centers) == 0:
        current = np.zeros(3, dtype=np.float32)
    else:
        idx = min(max(frame_idx, 0), len(centers) - 1)
        current = centers[idx].astype(np.float32)

    if scene_center is None:
        scene_center = current
    scene_center = np.asarray(scene_center, dtype=np.float32)

    blend = float(np.clip(overview, 0.0, 1.0))
    blend = blend * blend * (3.0 - 2.0 * blend)

    path_focus = current + direction * radius * 0.28
    path_target = current + direction * radius * 0.72 - WORLD_UP * radius * 0.04
    path_eye = current - direction * radius * 1.15 + WORLD_UP * radius * 0.46

    overview_target = scene_center
    overview_eye = scene_center - direction * radius * 2.35 + WORLD_UP * radius * 0.78

    focus = (1.0 - blend) * path_focus + blend * scene_center
    target = (1.0 - blend) * path_target + blend * overview_target
    eye = (1.0 - blend) * path_eye + blend * overview_eye
    view_radius = (1.0 - blend) * max(radius * 0.82, 0.25) + blend * max(radius * 1.18, 0.35)
    return make_view(target, eye), focus.astype(np.float32), float(view_radius)


def make_view_with_basis(
    target: np.ndarray,
    eye: np.ndarray,
    right_hint: np.ndarray,
    up_hint: np.ndarray,
) -> dict[str, np.ndarray]:
    forward = target - eye
    forward = normalize_vector(forward, DEFAULT_FORWARD)
    right = np.asarray(right_hint, dtype=np.float32)
    right = right - forward * float(right @ forward)
    if float(np.linalg.norm(right)) < 1e-5:
        right = np.cross(forward, up_hint)
    right = normalize_vector(right, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up = np.cross(right, forward)
    up = normalize_vector(up, up_hint)
    if float(up @ up_hint) < 0.0:
        right = -right
        up = normalize_vector(np.cross(right, forward), up_hint)
    return {"eye": eye, "right": right, "up": up, "forward": forward}


def make_view(target: np.ndarray, eye: np.ndarray) -> dict[str, np.ndarray]:
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
    right = np.cross(forward, WORLD_UP)
    if np.linalg.norm(right) < 1e-5:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-6)
    return {"eye": eye, "right": right, "up": up, "forward": forward}


def project_points(
    points: np.ndarray,
    view: dict[str, np.ndarray],
    center: np.ndarray,
    radius: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty, np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool)
    rel = points - center[None]
    x3 = rel @ view["right"]
    y3 = rel @ view["up"]
    z3 = rel @ view["forward"]
    scale = 0.42 * min(width, height) / max(radius, 1e-6)
    x = (width * 0.5 + x3 * scale).astype(np.int32)
    y = (height * 0.54 - y3 * scale).astype(np.int32)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x, y, z3, inside


def render_cloud(
    canvas: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    view: dict[str, np.ndarray],
    center: np.ndarray,
    radius: float,
    point_radius: int = 1,
) -> None:
    if len(points) == 0:
        return
    h, w = canvas.shape[:2]
    x, y, z, inside = project_points(points, view, center, radius, w, h)
    x, y, z = x[inside], y[inside], z[inside]
    cols = colors[inside]
    if len(x) == 0:
        return
    order = np.argsort(z)
    if point_radius <= 1:
        canvas[y[order], x[order]] = cols[order][:, ::-1]
    else:
        for px, py, col in zip(x[order], y[order], cols[order], strict=False):
            cv2.circle(canvas, (int(px), int(py)), point_radius, tuple(int(v) for v in col[::-1]), -1, cv2.LINE_AA)


def draw_camera_path(
    canvas: np.ndarray,
    cameras: dict[str, np.ndarray],
    frame_idx: int,
    view: dict[str, np.ndarray],
    center: np.ndarray,
    radius: float,
) -> None:
    centers = cameras["centers"]
    if len(centers) < 2:
        return
    h, w = canvas.shape[:2]
    x, y, _, inside = project_points(centers[: frame_idx + 1], view, center, radius, w, h)
    pts = np.stack([x, y], axis=1)
    valid_pts = pts[inside]
    if len(valid_pts) > 1:
        cv2.polylines(canvas, [valid_pts.astype(np.int32)], False, (235, 235, 235), 2, cv2.LINE_AA)

    for past in range(0, frame_idx + 1, 8):
        draw_frustum(canvas, cameras["segments"][past], view, center, radius, color=(130, 130, 130), thickness=1)
    draw_frustum(canvas, cameras["segments"][frame_idx], view, center, radius, color=(100, 216, 146), thickness=2)


def draw_frustum(
    canvas: np.ndarray,
    segments: np.ndarray,
    view: dict[str, np.ndarray],
    center: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    h, w = canvas.shape[:2]
    flat = segments.reshape(-1, 3)
    x, y, _, inside = project_points(flat, view, center, radius, w, h)
    pts = np.stack([x, y], axis=1).reshape(-1, 2, 2)
    inside_pairs = inside.reshape(-1, 2)
    for pair, ok in zip(pts, inside_pairs, strict=False):
        if ok.all():
            cv2.line(canvas, tuple(pair[0]), tuple(pair[1]), color, thickness, cv2.LINE_AA)


def draw_input_video(
    canvas: np.ndarray,
    image_rgb: np.ndarray,
    frame_idx: int,
    num_frames: int,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    resized = cv2.resize(image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)
    panel = np.full((h, w, 3), 4, dtype=np.uint8)
    oy = (h - resized.shape[0]) // 2
    ox = (w - resized.shape[1]) // 2
    panel[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    canvas[y : y + h, x : x + w] = panel
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (82, 82, 82), 1)
    draw_text(canvas, f"frame {frame_idx + 1:02d}/{num_frames:02d}", (x, y + h + 28), scale=0.48, color=(220, 220, 220))


def draw_progress(canvas: np.ndarray, frame_idx: int, num_frames: int, points_added: int, origin: tuple[int, int], width: int) -> None:
    x, y = origin
    cv2.rectangle(canvas, (x, y), (x + width, y + 5), (54, 54, 54), -1)
    fill = int(width * (frame_idx + 1) / max(num_frames, 1))
    cv2.rectangle(canvas, (x, y), (x + fill, y + 5), (100, 216, 146), -1)
    draw_text(canvas, f"frame {frame_idx + 1:02d}/{num_frames:02d}   fused points {points_added:,}", (x, y - 12), scale=0.43, color=(220, 220, 220))


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
