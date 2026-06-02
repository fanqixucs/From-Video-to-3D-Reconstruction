from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from video_to_3d.export import (
    export_glb,
    export_ply,
    save_predictions,
    tensor_predictions_to_numpy,
    unproject_depth_map_to_point_map,
    write_camera_json,
    write_depth_preview,
)
from video_to_3d.metrics import (
    apply_sim3,
    camera_pose_metrics,
    chamfer_fscore,
    depth_maps_to_world_points,
    extrinsic_to_cam2world,
    predicted_world_points_for_eval,
    run_no_gt_evaluation,
    save_json,
    umeyama_align,
)
from video_to_3d.report import write_run_report, write_scene_html
from video_to_3d.vggt_runner import VGGTRunConfig, add_vggt_root, cuda_environment, run_vggt_omega
from video_to_3d.video import write_contact_sheet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCANNET_BUNDLE = Path(os.environ.get("SCANNET_BUNDLE", str(PROJECT_ROOT / "data" / "scannet_scene0000_00")))
DEFAULT_VGGT_ROOT = Path(os.environ.get("VGGT_OMEGA_ROOT", str(PROJECT_ROOT.parent / "vggt-omega")))
DEFAULT_CHECKPOINT = Path(
    os.environ.get("VGGT_OMEGA_CHECKPOINT", str(PROJECT_ROOT / "checkpoints" / "vggt_omega" / "vggt_omega_1b_512.pt"))
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VGGT-Omega on ScanNet scene0000_00 and evaluate against GT.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_SCANNET_BUNDLE, help="Pre-extracted ScanNet bundle.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--run-name", default="scannet_scene0000_00_vggt_omega_1b_512")
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf-thres", type=float, default=50.0)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--ply-conf-thres", type=float, default=5.0)
    parser.add_argument("--ply-max-points", type=int, default=8_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true", help="Skip VGGT if predictions.npz already exists.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = args.output_dir / args.run_name
    frames_out = run_dir / "frames"
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(exist_ok=True)

    add_vggt_root(args.vggt_root)
    env = cuda_environment()
    print(f"[env] {json.dumps(env)}", flush=True)

    frame_paths = sorted((args.bundle / "frames").glob("*.png"))
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise FileNotFoundError(f"No ScanNet frames found under {args.bundle / 'frames'}")

    copied_frames = []
    for frame_path in frame_paths:
        dst = frames_out / frame_path.name
        if not dst.exists():
            shutil.copy2(frame_path, dst)
        copied_frames.append(dst)
    contact_sheet = write_contact_sheet(copied_frames, run_dir / "contact_sheet.jpg")

    gt_pose_pack = np.load(args.bundle / "gt_poses_cam2world.npz")
    gt_c2w = gt_pose_pack["cam2world"][: len(copied_frames)]
    intrinsic_depth = gt_pose_pack["intrinsic_depth"]
    gt_depth = np.load(args.bundle / "gt_depth.npy")[: len(copied_frames)]

    predictions_path = run_dir / "predictions.npz"
    vggt_config = VGGTRunConfig(args.vggt_root, args.checkpoint, args.image_resolution, "cuda")
    if args.reuse_predictions and predictions_path.exists():
        with np.load(predictions_path) as loaded:
            predictions = {key: np.array(loaded[key]) for key in loaded.files}
    else:
        predictions_torch = run_vggt_omega(copied_frames, vggt_config)
        predictions = tensor_predictions_to_numpy(predictions_torch)
        predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
            predictions["depth"],
            predictions["extrinsic"],
            predictions["intrinsic"],
        )
        save_predictions(predictions, predictions_path)

    cameras_path = write_camera_json(predictions["extrinsic"], predictions["intrinsic"], run_dir / "cameras.json")
    depth_preview = write_depth_preview(predictions, run_dir / "depth_preview.jpg")
    ply_path, ply_points = export_ply(
        predictions,
        run_dir / "pointcloud.ply",
        conf_thres=args.ply_conf_thres,
        max_points=args.ply_max_points,
        align_to_first_camera=True,
        filter_depth_edges=False,
    )
    glb_path = export_glb(
        predictions,
        run_dir / "scene.glb",
        target_dir=run_dir,
        conf_thres=args.conf_thres,
        max_points=args.max_points,
        show_cam=True,
    )
    html_path = write_scene_html(glb_path, run_dir / "scene.html", title=args.run_name)

    pred_c2w = extrinsic_to_cam2world(predictions["extrinsic"])
    pose_metrics = camera_pose_metrics(pred_c2w, gt_c2w)
    scale, rotation, translation = umeyama_align(pred_c2w[:, :3, 3], gt_c2w[:, :3, 3])
    pred_eval_points = predicted_world_points_for_eval(predictions, max_points=500_000)
    pred_eval_points_aligned = apply_sim3(pred_eval_points, scale, rotation, translation)
    gt_eval_points = depth_maps_to_world_points(gt_depth, intrinsic_depth, gt_c2w, max_points=500_000)
    recon_metrics = chamfer_fscore(pred_eval_points_aligned, gt_eval_points)
    no_gt_metrics = run_no_gt_evaluation(predictions)

    metrics = {
        "dataset": "ScanNet scene0000_00",
        "num_frames": int(len(copied_frames)),
        "pose_vs_gt": pose_metrics,
        "reconstruction_vs_depth_fused_gt": recon_metrics,
        "no_gt_coherence": no_gt_metrics,
        "notes": [
            "Pose is Sim(3)-aligned before ATE because feed-forward monocular reconstruction has arbitrary global scale.",
            "3D metrics compare VGGT points to a GT point cloud fused from ScanNet depth maps and GT poses for the same 48 frames.",
            "This is an evidence run, not a leaderboard submission; frame sampling and GT construction differ from many published ScanNet tables.",
        ],
    }
    save_json(metrics, run_dir / "scannet_eval_metrics.json")

    manifest = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "VGGT-Omega-1B-512 on ScanNet scene0000_00 + GT/no-GT evaluation",
        "input_bundle": str(args.bundle),
        "vggt": vggt_config.to_dict(),
        "environment": env,
        "artifacts": {
            "pointcloud_ply": str(ply_path),
            "scene_glb": str(glb_path),
            "scene_html": str(html_path),
            "predictions_npz": str(predictions_path),
            "cameras_json": str(cameras_path),
            "contact_sheet": str(contact_sheet),
            "depth_preview": str(depth_preview),
            "metrics_json": str(run_dir / "scannet_eval_metrics.json"),
        },
        "postprocess": {
            "ply_points": ply_points,
            "ply_conf_thres": args.ply_conf_thres,
            "ply_max_points": args.ply_max_points,
        },
        "metrics": metrics,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_run_report(
        run_dir / "REPORT.md",
        run_name=args.run_name,
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

    print("[done] ScanNet evaluation complete", flush=True)
    print(f"[done] ATE RMSE: {pose_metrics['ate_rmse_m']:.4f} m", flush=True)
    print(f"[done] F1@10cm: {recon_metrics['f1@10cm']:.3f}", flush=True)
    print(f"[done] metrics: {run_dir / 'scannet_eval_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
