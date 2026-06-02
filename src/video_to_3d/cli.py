from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from video_to_3d.export import (
    export_glb,
    export_ply,
    save_predictions,
    summarize_predictions,
    tensor_predictions_to_numpy,
    unproject_depth_map_to_point_map,
    write_camera_json,
    write_depth_preview,
)
from video_to_3d.report import write_run_report, write_scene_html
from video_to_3d.vggt_runner import VGGTRunConfig, add_vggt_root, cuda_environment, run_vggt_omega
from video_to_3d.video import extract_frames, write_contact_sheet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = Path(os.environ.get("VIDEO_PATH", str(PROJECT_ROOT / "data" / "kitchen_small_room.MOV")))
DEFAULT_VGGT_ROOT = Path(os.environ.get("VGGT_OMEGA_ROOT", str(PROJECT_ROOT.parent / "vggt-omega")))
DEFAULT_CHECKPOINT = Path(
    os.environ.get("VGGT_OMEGA_CHECKPOINT", str(PROJECT_ROOT / "checkpoints" / "vggt_omega" / "vggt_omega_1b_512.pt"))
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct a small indoor scene from phone video with VGGT-Omega.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs", help="Directory for run outputs.")
    parser.add_argument("--run-name", default=None, help="Stable output subdirectory name.")
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT, help="Local facebookresearch/vggt-omega repo.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="VGGT-Omega 1B-512 checkpoint path.")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Video sampling FPS. Default mirrors the demo.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for lower-memory experiments.")
    parser.add_argument("--image-resolution", type=int, default=512, help="VGGT-Omega image resolution.")
    parser.add_argument("--conf-thres", type=float, default=50.0, help="GLB confidence percentile for visualization.")
    parser.add_argument("--max-points", type=int, default=1_000_000, help="Maximum points exported to the GLB.")
    parser.add_argument("--ply-conf-thres", type=float, default=5.0, help="PLY confidence percentile. Lower is denser.")
    parser.add_argument("--ply-max-points", type=int, default=8_000_000, help="Maximum points exported to pointcloud.ply.")
    parser.add_argument(
        "--ply-filter-depth-edges",
        action="store_true",
        help="Apply VGGT-style depth-edge pruning to the PLY. Off by default for denser visualization.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="Inference device.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run directory.")
    parser.add_argument("--compressed-npz", action="store_true", help="Save predictions with np.savez_compressed.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/environment without running inference.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    run_name = args.run_name or f"{args.video.stem}_vggt_omega_1b_512"
    run_dir = args.output_dir / run_name
    frames_dir = run_dir / "frames"

    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    add_vggt_root(args.vggt_root)
    env = cuda_environment()
    print(f"[env] {json.dumps(env)}", flush=True)
    print(f"[run] output directory: {run_dir}", flush=True)

    if args.dry_run:
        print("[run] dry-run complete", flush=True)
        return

    print(f"[video] extracting frames from: {args.video}", flush=True)
    extraction = extract_frames(
        video_path=args.video,
        frames_dir=frames_dir,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    print(
        f"[video] extracted {len(extraction.frames)} frames "
        f"(interval={extraction.frame_interval}, source_fps={extraction.video_info.fps:.3f})",
        flush=True,
    )

    contact_sheet = write_contact_sheet(extraction.frames, run_dir / "contact_sheet.jpg")
    print(f"[preview] wrote {contact_sheet}", flush=True)

    vggt_config = VGGTRunConfig(
        vggt_root=args.vggt_root,
        checkpoint=args.checkpoint,
        image_resolution=args.image_resolution,
        device=args.device,
    )
    predictions_torch = run_vggt_omega(extraction.frames, vggt_config)

    predictions = tensor_predictions_to_numpy(predictions_torch)
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"],
        predictions["extrinsic"],
        predictions["intrinsic"],
    )

    predictions_path = save_predictions(predictions, run_dir / "predictions.npz", compressed=args.compressed_npz)
    cameras_path = write_camera_json(predictions["extrinsic"], predictions["intrinsic"], run_dir / "cameras.json")
    depth_preview = write_depth_preview(predictions, run_dir / "depth_preview.jpg")
    ply_path, ply_points = export_ply(
        predictions,
        output_path=run_dir / "pointcloud.ply",
        conf_thres=args.ply_conf_thres,
        max_points=args.ply_max_points,
        align_to_first_camera=True,
        filter_depth_edges=args.ply_filter_depth_edges,
    )
    glb_path = export_glb(
        predictions,
        output_path=run_dir / "scene.glb",
        target_dir=run_dir,
        conf_thres=args.conf_thres,
        max_points=args.max_points,
        show_cam=True,
    )
    html_path = write_scene_html(glb_path, run_dir / "scene.html", title=run_name)
    metrics = summarize_predictions(predictions)

    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "VGGT-Omega-1B-512 default model + depth unprojection + PLY/GLB export",
        "input_video": str(args.video),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(run_dir),
        "vggt": vggt_config.to_dict(),
        "video": extraction.to_dict(),
        "postprocess": {
            "conf_thres": args.conf_thres,
            "max_points": args.max_points,
            "show_cameras": True,
            "compressed_npz": args.compressed_npz,
            "ply_points": ply_points,
            "ply_conf_thres": args.ply_conf_thres,
            "ply_max_points": args.ply_max_points,
            "ply_filter_depth_edges": args.ply_filter_depth_edges,
            "ply_coordinate_frame": "aligned to first predicted camera, matching scene.glb visualization frame",
        },
        "environment": env,
        "artifacts": {
            "pointcloud_ply": str(ply_path),
            "scene_glb": str(glb_path),
            "scene_html": str(html_path),
            "predictions_npz": str(predictions_path),
            "cameras_json": str(cameras_path),
            "contact_sheet": str(contact_sheet),
            "depth_preview": str(depth_preview),
        },
        "metrics": metrics,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path = write_run_report(
        run_dir / "REPORT.md",
        run_name=run_name,
        artifact_paths={
            "ply": ply_path,
            "glb": glb_path,
            "html": html_path,
            "predictions": predictions_path,
            "cameras": cameras_path,
            "contact_sheet": contact_sheet,
            "depth_preview": depth_preview,
        },
        manifest=manifest,
        metrics=metrics,
    )

    print("[done] reconstruction complete", flush=True)
    print(f"[done] point cloud: {ply_path}", flush=True)
    print(f"[done] scene: {glb_path}", flush=True)
    print(f"[done] viewer: {html_path}", flush=True)
    print(f"[done] report: {report_path}", flush=True)
    print(f"[done] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
