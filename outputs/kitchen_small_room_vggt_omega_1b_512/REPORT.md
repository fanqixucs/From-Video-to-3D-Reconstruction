# kitchen_small_room_vggt_omega_1b_512

Generated: `2026-05-28T00:07:06`

## Artifacts

- Point cloud PLY: `pointcloud.ply`
- Scene GLB: `scene.glb`
- Browser viewer: `scene.html`
- Raw predictions: `predictions.npz`
- Camera trajectory: `cameras.json`
- Input contact sheet: `contact_sheet.jpg`
- Depth preview: `depth_preview.jpg`

## Reconstruction Metrics

```json
{
  "num_frames": 80,
  "image_shape_hw": [
    688,
    384
  ],
  "raw_point_count": 21135360,
  "finite_point_count": 21135360,
  "depth_min_p02": 0.18772493571043014,
  "depth_median": 0.4072667360305786,
  "depth_max_p98": 1.144744851589203,
  "confidence_p10": 7.8054098129272464,
  "confidence_median": 13.040724754333496,
  "confidence_p90": 19.369572067260748,
  "camera_path_length_world_units": 3.9201385974884033
}
```

## Run Manifest

```json
{
  "run_name": "kitchen_small_room_vggt_omega_1b_512",
  "created_at": "2026-05-27T23:54:42",
  "pipeline": "VGGT-Omega-1B-512 default model + depth unprojection + dense PLY/GLB export",
  "input_video": "/home/fanqixu/experiments/datasets/humanoid_inter_challenge/kitchen_small_room.MOV",
  "project_root": "/home/fanqixu/experiments/video-to-3D",
  "output_dir": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512",
  "vggt": {
    "vggt_root": "/home/fanqixu/experiments/vggt-omega",
    "checkpoint": "/home/fanqixu/experiments/checkpoints/vggt_omega/vggt_omega_1b_512.pt",
    "image_resolution": 512,
    "device": "cuda"
  },
  "video": {
    "frames": [
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000000.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000001.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000002.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000003.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000004.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000005.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000006.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000007.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000008.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000009.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000010.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000011.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000012.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000013.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000014.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000015.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000016.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000017.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000018.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000019.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000020.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000021.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000022.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000023.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000024.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000025.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000026.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000027.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000028.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000029.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000030.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000031.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000032.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000033.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000034.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000035.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000036.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000037.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000038.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000039.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000040.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000041.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000042.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000043.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000044.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000045.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000046.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000047.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000048.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000049.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000050.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000051.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000052.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000053.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000054.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000055.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000056.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000057.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000058.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000059.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000060.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000061.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000062.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000063.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000064.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000065.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000066.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000067.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000068.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000069.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000070.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000071.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000072.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000073.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000074.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000075.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000076.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000077.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000078.jpg",
      "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/frames/000079.jpg"
    ],
    "num_frames": 80,
    "video_info": {
      "path": "/home/fanqixu/experiments/datasets/humanoid_inter_challenge/kitchen_small_room.MOV",
      "width": 1920,
      "height": 1080,
      "fps": 29.978204462895693,
      "frame_count": 2407,
      "duration_sec": 80.29166666666667
    },
    "sample_fps": 1.0,
    "frame_interval": 30,
    "backend": "ffmpeg"
  },
  "postprocess": {
    "conf_thres": 50.0,
    "max_points": 1000000,
    "show_cameras": true,
    "compressed_npz": false,
    "ply_points": 8000000,
    "ply_coordinate_frame": "aligned to first predicted camera, matching scene.glb visualization frame",
    "ply_conf_thres": 5.0,
    "ply_max_points": 8000000,
    "ply_filter_depth_edges": false,
    "pointcloud_full_points": 20712652
  },
  "environment": {
    "torch_version": "2.9.1+cu128",
    "cuda_available": true,
    "cuda_device_count": 1,
    "cuda_device_name": "NVIDIA RTX A6000"
  },
  "artifacts": {
    "scene_glb": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/scene.glb",
    "scene_html": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/scene.html",
    "predictions_npz": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/predictions.npz",
    "cameras_json": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/cameras.json",
    "contact_sheet": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/contact_sheet.jpg",
    "depth_preview": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/depth_preview.jpg",
    "pointcloud_ply": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/pointcloud.ply",
    "pointcloud_dense_ply": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/pointcloud_dense.ply",
    "pointcloud_sparse_1m_ply": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/pointcloud_sparse_1m.ply",
    "pointcloud_full_ply": "/home/fanqixu/experiments/video-to-3D/outputs/kitchen_small_room_vggt_omega_1b_512/pointcloud_full.ply"
  },
  "metrics": {
    "num_frames": 80,
    "image_shape_hw": [
      688,
      384
    ],
    "raw_point_count": 21135360,
    "finite_point_count": 21135360,
    "depth_min_p02": 0.18772493571043014,
    "depth_median": 0.4072667360305786,
    "depth_max_p98": 1.144744851589203,
    "confidence_p10": 7.8054098129272464,
    "confidence_median": 13.040724754333496,
    "confidence_p90": 19.369572067260748,
    "camera_path_length_world_units": 3.9201385974884033
  }
}
```
