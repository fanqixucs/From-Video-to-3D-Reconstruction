"""Held-out novel-view rendering test.

The strongest no-ground-truth check of a reconstruction is: hide a camera, rebuild the scene
from every *other* frame, render the hidden camera's view, and compare to the real photo. If the
geometry and poses are right, the re-rendered image matches the held-out frame.

For each evaluated frame i we gather the confidence-filtered 3D points contributed by all frames
j != i, project them through camera i's estimated pose/intrinsics with a z-buffer, and score the
rendered RGB against the real frame i using PSNR and SSIM over the covered pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from video_to_3d.export import _images_to_rgb


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM map (averaged over colour channels), 11x11 Gaussian windows.

    Self-contained (no scikit-image dependency) so the analysis runs in the geometry env.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win, sigma = (11, 11), 1.5
    maps = []
    for ch in range(a.shape[2]):
        x, y = a[:, :, ch], b[:, :, ch]
        mux = cv2.GaussianBlur(x, win, sigma)
        muy = cv2.GaussianBlur(y, win, sigma)
        mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
        sx = cv2.GaussianBlur(x * x, win, sigma) - mux2
        sy = cv2.GaussianBlur(y * y, win, sigma) - muy2
        sxy = cv2.GaussianBlur(x * y, win, sigma) - muxy
        maps.append(((2 * muxy + c1) * (2 * sxy + c2)) / ((mux2 + muy2 + c1) * (sx + sy + c2)))
    return np.mean(maps, axis=0)


def render_from_camera(points_w, colors, extrinsic_i, intrinsic_i, height, width, splat=2):
    """Z-buffer project world points into camera i with a small splat. Returns (image, mask)."""
    R = extrinsic_i[:3, :3]
    t = extrinsic_i[:3, 3]
    cam = points_w @ R.T + t  # world -> camera
    z = cam[:, 2]
    front = z > 1e-3
    cam, colz, z = cam[front], colors[front], z[front]
    fx, fy = intrinsic_i[0, 0], intrinsic_i[1, 1]
    cx, cy = intrinsic_i[0, 2], intrinsic_i[1, 2]
    u = (fx * cam[:, 0] / cam[:, 2] + cx).astype(np.int32)
    v = (fy * cam[:, 1] / cam[:, 2] + cy).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, colz = u[inside], v[inside], z[inside], colz[inside]
    order = np.argsort(z)[::-1]  # far first so nearer points overwrite
    u, v, colz = u[order], v[order], colz[order]
    img = np.zeros((height, width, 3), np.uint8)
    cover = np.zeros((height, width), bool)
    offsets = [(dx, dy) for dy in range(-splat, splat + 1) for dx in range(-splat, splat + 1) if dx * dx + dy * dy <= splat * splat]
    offsets.sort(key=lambda o: -(o[0] ** 2 + o[1] ** 2))  # outer first, centre last
    for dx, dy in offsets:
        tu, tv = u + dx, v + dy
        ok = (tu >= 0) & (tu < width) & (tv >= 0) & (tv < height)
        img[tv[ok], tu[ok]] = colz[ok]
        cover[tv[ok], tu[ok]] = True
    return img, cover


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Held-out novel-view rendering test (PSNR/SSIM).")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-views", type=int, default=12, help="held-out frames to evaluate")
    parser.add_argument("--budget", type=int, default=2_500_000, help="max points projected per view")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or (args.run_dir / "novel_view")
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.run_dir / "predictions.npz") as loaded:
        preds = {k: np.array(loaded[k]) for k in loaded.files}
    P = preds["world_points_from_depth"].astype(np.float32)       # (N,H,W,3)
    conf = preds["depth_conf"].astype(np.float32)                 # (N,H,W)
    rgb = (_images_to_rgb(preds["images"]) * 255).clip(0, 255).astype(np.uint8)  # (N,H,W,3)
    extr = preds["extrinsic"].astype(np.float64)
    intr = preds["intrinsic"].astype(np.float64)
    n, H, W = conf.shape

    # pre-flatten per-frame point/colour/conf for fast gather
    pts_f = P.reshape(n, -1, 3)
    col_f = rgb.reshape(n, -1, 3)
    cnf_f = conf.reshape(n, -1)
    finite = np.isfinite(pts_f).all(-1) & np.isfinite(cnf_f) & (cnf_f > 1e-5)
    thr = np.array([np.percentile(cnf_f[i][finite[i]], args.conf_percentile) if finite[i].any() else 0.0 for i in range(n)])

    eval_idx = np.linspace(0, n - 1, args.num_views).astype(int)
    rng = np.random.default_rng(0)
    rows = []
    sample_pair = None
    for i in eval_idx:
        # gather points from all other frames
        keep_pts, keep_col = [], []
        for j in range(n):
            if j == i:
                continue
            m = finite[j] & (cnf_f[j] >= thr[j])
            keep_pts.append(pts_f[j][m])
            keep_col.append(col_f[j][m])
        allp = np.concatenate(keep_pts)
        allc = np.concatenate(keep_col)
        if len(allp) > args.budget:
            sel = rng.choice(len(allp), args.budget, replace=False)
            allp, allc = allp[sel], allc[sel]
        rendered, cover = render_from_camera(allp, allc, extr[i], intr[i], H, W)
        real = rgb[i]

        cov = float(cover.mean())
        m = cover
        mse = float(((rendered[m].astype(np.float32) - real[m].astype(np.float32)) ** 2).mean())
        psnr = 10 * np.log10(255.0**2 / max(mse, 1e-6))
        # SSIM over covered pixels: fill holes with the real frame so window stats are local,
        # then average the SSIM map only where we actually rendered.
        filled = rendered.copy()
        filled[~cover] = real[~cover]
        smap = ssim_map(real, filled)
        ssim = float(smap[cover].mean())
        rows.append({"frame": int(i), "coverage": round(cov, 3), "psnr_db": round(float(psnr), 2), "ssim": round(ssim, 3)})
        if sample_pair is None and cov > 0.5:
            sample_pair = (real, rendered, cover, int(i))

    psnrs = np.array([r["psnr_db"] for r in rows])
    ssims = np.array([r["ssim"] for r in rows])
    covs = np.array([r["coverage"] for r in rows])
    summary = {
        "protocol": "leave-one-out: render each held-out camera from all OTHER frames' confidence-filtered points; compare to the real frame over covered pixels",
        "num_views": len(rows),
        "points_per_view_budget": args.budget,
        "conf_percentile": args.conf_percentile,
        "psnr_db_mean": round(float(psnrs.mean()), 2),
        "psnr_db_median": round(float(np.median(psnrs)), 2),
        "ssim_mean": round(float(ssims.mean()), 3),
        "ssim_median": round(float(np.median(ssims)), 3),
        "coverage_mean": round(float(covs.mean()), 3),
        "per_view": rows,
    }
    (out_dir / "novel_view_metrics.json").write_text(json.dumps(summary, indent=2))

    if sample_pair is not None:
        real, rendered, cover, fi = sample_pair
        real_b = cv2.cvtColor(real, cv2.COLOR_RGB2BGR)
        rend_b = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        gap = np.full((real.shape[0], 8, 3), 255, np.uint8)
        panel = np.hstack([real_b, gap, rend_b])
        cv2.putText(panel, f"real frame {fi}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "rendered from OTHER frames", (real.shape[1] + 16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "novel_view_compare.jpg"), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_dir}/novel_view_metrics.json and novel_view_compare.jpg")


if __name__ == "__main__":
    main()
