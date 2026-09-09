from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from aero3d.geo import geodetic_to_enu, rotation_zyx
from aero3d.types import CameraIntrinsics


def generate_synthetic_mission(out_dir: str | Path, seconds: float = 8.0, fps: int = 12) -> dict:
    """Render a single-pass flyover of terrain + buildings; write mp4 + GPS telemetry."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width, height = 960, 540
    cam = CameraIntrinsics.from_size_hfov(width, height, 65.0)

    lat0, lon0, alt0 = 28.6139, 77.2090, 210.0
    n_frames = int(seconds * fps)
    xs = np.linspace(-40, 40, n_frames)
    ys = np.full(n_frames, 0.0)
    zs = np.full(n_frames, 55.0)
    yaw = np.full(n_frames, 90.0)  # heading east
    pitch = np.full(n_frames, -55.0)
    roll = np.zeros(n_frames)

    buildings = [
        {"c": np.array([5.0, 4.0, 0.0]), "s": np.array([8.0, 6.0, 14.0]), "color": (40, 90, 180)},
        {"c": np.array([18.0, -6.0, 0.0]), "s": np.array([6.0, 6.0, 10.0]), "color": (30, 140, 90)},
        {"c": np.array([-12.0, 2.0, 0.0]), "s": np.array([10.0, 5.0, 8.0]), "color": (160, 80, 50)},
    ]

    raw_dir = out / "raw_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    video_path = out / "single_pass.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        video_path = out / "single_pass.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open a VideoWriter for the synthetic mission")
    samples = []
    gt_pts = _ground_truth_points(buildings)

    for i in range(n_frames):
        t_enu = np.array([xs[i], ys[i], zs[i]], dtype=np.float64)
        R_body = rotation_zyx(yaw[i], pitch[i], roll[i])
        R_cam = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        R = R_body @ R_cam
        img = _render_scene(cam, R, t_enu, buildings)
        writer.write(img)
        cv2.imwrite(str(raw_dir / f"f{i:04d}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        lat = lat0 + ys[i] / 111132.92
        lon = lon0 + xs[i] / (111412.84 * np.cos(np.deg2rad(lat0)))
        alt = alt0 + zs[i]
        samples.append(
            {
                "t": i / fps,
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "yaw": float(yaw[i]),
                "pitch": float(pitch[i]),
                "roll": float(roll[i]),
            }
        )
    writer.release()

    telemetry = {
        "origin": {"lat": lat0, "lon": lon0, "alt": alt0},
        "camera": {
            "fx": cam.fx,
            "fy": cam.fy,
            "cx": cam.cx,
            "cy": cam.cy,
            "width": width,
            "height": height,
        },
        "samples": samples,
    }
    tel_path = out / "telemetry.json"
    tel_path.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    np.save(out / "gt_points.npy", gt_pts)
    return {
        "video": str(raw_dir),
        "video_file": str(video_path),
        "telemetry": str(tel_path),
        "frames": n_frames,
        "gt_points": str(out / "gt_points.npy"),
    }


def _ground_truth_points(buildings) -> np.ndarray:
    pts = []
    for b in buildings:
        c, s = b["c"], b["s"]
        for x in np.linspace(c[0] - s[0] / 2, c[0] + s[0] / 2, 12):
            for y in np.linspace(c[1] - s[1] / 2, c[1] + s[1] / 2, 10):
                pts.append([x, y, 0.0])
                pts.append([x, y, s[2]])
        for z in np.linspace(0, s[2], 8):
            for x in (c[0] - s[0] / 2, c[0] + s[0] / 2):
                for y in np.linspace(c[1] - s[1] / 2, c[1] + s[1] / 2, 8):
                    pts.append([x, y, z])
    # terrain grid
    for x in np.linspace(-30, 35, 40):
        for y in np.linspace(-20, 20, 30):
            h = 0.6 * np.sin(x / 8.0) + 0.4 * np.cos(y / 6.0)
            pts.append([x, y, h])
    return np.asarray(pts, dtype=np.float32)


def _render_scene(cam: CameraIntrinsics, R: np.ndarray, t: np.ndarray, buildings) -> np.ndarray:
    h, w = cam.height, cam.width
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (210, 200, 170)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)

    xs = np.linspace(-35, 40, 140)
    ys = np.linspace(-22, 22, 90)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.6 * np.sin(xx / 8.0) + 0.4 * np.cos(yy / 6.0)
    terrain = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    checker = (((xx.ravel() * 2).astype(int) + (yy.ravel() * 2).astype(int)) % 2)[:, None]
    tcol = np.where(checker == 0, np.array([50, 130, 55]), np.array([90, 170, 80]))
    _splat_colored(img, zbuf, cam, R, t, terrain, tcol, size=2)

    for b in buildings:
        c, s, col = b["c"], b["s"], np.array(b["color"], dtype=np.float64)
        corners = []
        colors = []
        for dx in np.linspace(-s[0] / 2, s[0] / 2, 28):
            for dy in np.linspace(-s[1] / 2, s[1] / 2, 22):
                window = int((dx * 3) % 2 == 0) * int((dy * 3) % 2 == 0)
                shade = 0.55 + 0.45 * window
                corners.append(c + np.array([dx, dy, 0.0]))
                colors.append(col * 0.35)
                corners.append(c + np.array([dx, dy, s[2]]))
                colors.append(col * shade)
            for z in np.linspace(0, s[2], 20):
                window = int((dx * 2) % 2 == 0) * int((z * 1.2) % 2 == 0)
                shade = 0.4 + 0.6 * window
                corners.append(c + np.array([dx, -s[1] / 2, z]))
                colors.append(col * shade)
                corners.append(c + np.array([dx, s[1] / 2, z]))
                colors.append(col * shade)
        _splat_colored(img, zbuf, cam, R, t, np.asarray(corners), np.asarray(colors), size=2)

    road = []
    rcol = []
    for x in np.linspace(-35, 40, 160):
        for y in np.linspace(-1.8, 1.8, 10):
            road.append([x, y, 0.05])
            stripe = 220 if abs(y) < 0.15 and int(x) % 4 < 2 else 55
            rcol.append([stripe, stripe, stripe + 10])
    _splat_colored(img, zbuf, cam, R, t, np.asarray(road), np.asarray(rcol), size=2)
    noise = np.random.default_rng(3).integers(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _splat_colored(img, zbuf, cam, R, t, pts, colors, size=2):
    pts = np.asarray(pts, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim == 1:
        colors = np.repeat(colors.reshape(1, 3), len(pts), axis=0)
    Xc = (R.T @ (pts.T - t.reshape(3, 1))).T
    z = Xc[:, 2]
    vis = z > 0.5
    if not np.any(vis):
        return
    Xc, colors = Xc[vis], colors[vis]
    z = Xc[:, 2]
    u = cam.fx * (Xc[:, 0] / z) + cam.cx
    v = cam.fy * (Xc[:, 1] / z) + cam.cy
    h, w = img.shape[:2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, z, colors = ui[inb], vi[inb], z[inb], colors[inb]
    for x, y, depth, col in zip(ui, vi, z, colors):
        if depth < zbuf[y, x]:
            y0, y1 = max(y - size, 0), min(y + size + 1, h)
            x0, x1 = max(x - size, 0), min(x + size + 1, w)
            img[y0:y1, x0:x1] = np.clip(col, 0, 255).astype(np.uint8)
            zbuf[y0:y1, x0:x1] = np.minimum(zbuf[y0:y1, x0:x1], depth)
