"""Fast, dependency-light unit tests for the analysis/eval logic.

These exercise the same code paths as scripts/smoke_test.sh but on small synthetic inputs, so they
need no GPU, no VGGT-Omega checkpoint, and no large LFS data — suitable for CI. (Geometry/semantic
modules that need torch or Grounded-SAM are intentionally not imported here.)
"""
from __future__ import annotations

import numpy as np

from video_to_3d import export, novel_view, reconstruction_player as rp, scene_analysis as sa


# ---------------------------------------------------------------- reconstruction_player helpers
def test_smoothstep_monotonic_and_clamped():
    assert rp.smoothstep(-1.0) == 0.0
    assert rp.smoothstep(2.0) == 1.0
    assert rp.smoothstep(0.0) < rp.smoothstep(0.5) < rp.smoothstep(1.0)


def test_disk_offsets_centre_last_and_within_radius():
    offs = rp._disk_offsets(3)
    assert offs[-1][1:] == (0, 0)  # centre drawn last
    assert all(dx * dx + dy * dy <= 3 * 3 + 1 for _, dx, dy in offs)


def test_voxel_point_counts_and_density_mask():
    pts = np.array([[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0], [5, 5, 5]], dtype=np.float32)
    counts = rp.voxel_point_counts(pts, voxel=0.1)
    assert counts[0] >= 3 and counts[-1] == 1  # the lone point is isolated
    keep = rp.voxel_density_mask(pts, voxel=0.1, min_count=2)
    assert keep[:3].all() and not keep[3]


def test_make_view_orthonormal():
    view = rp.make_view(target=np.zeros(3, np.float32), eye=np.array([0, 0, 3], np.float32))
    r, u, f = view["right"], view["up"], view["forward"]
    for v in (r, u, f):
        assert abs(np.linalg.norm(v) - 1) < 1e-5
    assert abs(r @ u) < 1e-5 and abs(r @ f) < 1e-5 and abs(u @ f) < 1e-5


def test_perspective_project_in_front_is_inside():
    view = rp.make_view(np.zeros(3, np.float32), np.array([0, 0, 5], np.float32))
    pts = np.array([[0, 0, 0]], dtype=np.float32)  # straight ahead of the eye
    x, y, z, inside = rp.perspective_project(pts, view, focal=500.0, width=200, height=200, near=0.1)
    assert inside[0] and z[0] > 0


# ---------------------------------------------------------------- export round-trip
def test_ply_write_read_roundtrip(tmp_path):
    verts = np.random.RandomState(0).rand(500, 3).astype(np.float32)
    cols = (np.random.RandomState(1).rand(500, 3) * 255).astype(np.uint8)
    path = tmp_path / "cloud.ply"
    export._write_binary_ply(path, verts, cols)
    pts, rgb, idx, total = rp.load_ordered_ply_xyzrgb(path, max_points=0)
    assert total == 500 and len(pts) == 500 and rgb.shape == (500, 3)
    assert np.allclose(pts, verts, atol=1e-5)


def test_transform_points_identity():
    pts = np.random.RandomState(2).rand(10, 3)
    out = export._transform_points(np.eye(4), pts)
    assert np.allclose(out, pts)


# ---------------------------------------------------------------- novel_view
def test_ssim_identical_is_high_and_noise_lower():
    img = (np.random.RandomState(3).rand(64, 64, 3) * 255).astype(np.uint8)
    same = float(sa_mean(novel_view.ssim_map(img, img)))
    noisy = img.astype(np.int16) + np.random.RandomState(4).randint(-60, 60, img.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    diff = float(sa_mean(novel_view.ssim_map(img, noisy)))
    assert same > 0.99 and diff < same


def test_render_from_camera_covers_pixels():
    # a small frontal grid of points, identity camera looking down +Z
    g = np.linspace(-0.5, 0.5, 30)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 2.0)], axis=1).astype(np.float32)
    cols = np.full((len(pts), 3), 200, np.uint8)
    extr = np.eye(4)[:3]  # world == camera
    K = np.array([[400, 0, 100], [0, 400, 100], [0, 0, 1]], dtype=np.float64)
    img, cover = novel_view.render_from_camera(pts, cols, extr, K, height=200, width=200, splat=2)
    assert img.shape == (200, 200, 3) and cover.any()


def sa_mean(a):
    return np.mean(a)


# ---------------------------------------------------------------- scene_analysis (uses open3d)
def _synthetic_scene():
    """Floor at z=0, counter at z=0.3, cupboard band z=0.5..0.9, plus a small dust-bin cluster."""
    rng = np.random.RandomState(5)
    floor = np.column_stack([rng.uniform(-1, 1, 4000), rng.uniform(-1, 1, 4000), rng.normal(0, 0.002, 4000)])
    counter = np.column_stack([rng.uniform(-1, 1, 3000), rng.uniform(-1, 1, 3000), rng.normal(0.30, 0.003, 3000)])
    cup = np.column_stack([rng.uniform(-1, 1, 3000), rng.uniform(0.9, 1.0, 3000), rng.uniform(0.5, 0.9, 3000)])
    bin_ = np.column_stack([rng.normal(0.4, 0.05, 1500), rng.normal(-0.4, 0.05, 1500), rng.uniform(0.0, 0.15, 1500)])
    pts = np.vstack([floor, counter, cup, bin_]).astype(np.float64)
    labels = np.concatenate([np.full(4000, 3), np.full(3000, 1), np.full(3000, 2), np.full(1500, 4)]).astype(np.int32)
    name2id = {"countertop": 1, "cupboard": 2, "floor": 3, "dust bin": 4}
    summary = {"1": {"name": "countertop"}, "2": {"name": "cupboard"}, "3": {"name": "floor"}, "4": {"name": "dust bin"}}
    return pts, labels, name2id, summary


def test_fit_plane_recovers_horizontal_normal():
    rng = np.random.RandomState(6)
    plane = np.column_stack([rng.uniform(-1, 1, 3000), rng.uniform(-1, 1, 3000), rng.normal(0, 0.002, 3000)])
    n, d, inl = sa.fit_plane(plane, threshold=0.02)
    assert abs(abs(n[2]) - 1.0) < 0.05 and len(inl) > 2000


def test_gravity_frame_up_points_toward_cameras():
    pts, labels, name2id, _ = _synthetic_scene()
    floor = pts[labels == name2id["floor"]]
    cams = np.array([[0, 0, 1.4], [0.3, 0.2, 1.4]], dtype=np.float64)  # above the floor
    frame = sa.estimate_gravity_frame(floor, cams, scene_diag=3.0)
    assert frame["up"][2] > 0.9  # up ~ +z


def test_scale_anchor_counter_height():
    pts, labels, name2id, _ = _synthetic_scene()
    # already gravity-aligned (z=height); counter at 0.30 -> 0.90 m => ~3.0 m/unit
    info = sa.compute_scale_anchor(pts, labels, name2id)
    assert 2.6 < info["meters_per_scene_unit"] < 3.4


def test_object_anchors_and_bev_shapes():
    pts, labels, name2id, summary = _synthetic_scene()
    scale = sa.compute_scale_anchor(pts, labels, name2id)["meters_per_scene_unit"]
    anchors = sa.object_anchors(pts, labels, summary, scale)
    names = {a["name"] for a in anchors}
    assert "dust bin" in names
    bin_anchor = next(a for a in anchors if a["name"] == "dust bin")
    assert bin_anchor["kind"] == "object" and bin_anchor["size_m"]["height"] > 0
    bev = sa.render_bev(pts, np.full((len(pts), 3), 180, np.uint8), labels, name2id, anchors, scale)
    assert bev.ndim == 3 and bev.shape[2] == 3 and bev.size > 0
