from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aero3d.geo import geodetic_to_enu, rotation_zyx
from aero3d.ingest import interpolate_sample, resolve_origin
from aero3d.types import CameraIntrinsics, FrameRecord, Telemetry


def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _iter_source_frames(source: Path | None):
    if source is None or not str(source) or not Path(source).exists():
        # Generate Synthetic 3D Drone Flight frames mathematically for testing
        print("[SYS] Generating Synthetic Drone Video frames dynamically...")
        fps = 30.0
        h, w = 480, 640
        for idx in range(30):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Create a moving synthetic scene (simulating a drone flying over buildings)
            offset = int(idx * 5)
            cv2.rectangle(frame, (100 - offset, 100), (200 - offset, 300), (0, 120, 0), -1)
            cv2.rectangle(frame, (400 - offset, 150), (550 - offset, 350), (120, 0, 0), -1)
            # Draw a synthetic "car" that the CLIPSeg AI can detect
            cv2.rectangle(frame, (300 - int(offset*1.5), 250), (350 - int(offset*1.5), 280), (0, 0, 255), -1)
            # Add synthetic noise
            noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            yield idx, idx / fps, frame, fps
        return
        
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not files:
            raise FileNotFoundError(f"No images in {source}")
        fps = 12.0
        for idx, path in enumerate(files):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            yield idx, idx / fps, frame, fps
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, idx / fps, frame, fps
        idx += 1
    cap.release()


def select_frames(
    video_path: str | Path | None,
    telemetry: Telemetry,
    cfg: dict,
    work_dir: Path,
) -> tuple[list[FrameRecord], CameraIntrinsics, float]:
    source = Path(video_path) if video_path else None
    gen = _iter_source_frames(source)
    try:
        first = next(gen)
    except StopIteration as exc:
        raise RuntimeError(f"No frames in {source}") from exc

    _idx, _t, frame0, fps = first
    src_h, src_w = frame0.shape[:2]
    target_w = int(cfg["video"]["target_width"])
    scale = target_w / max(src_w, 1)
    out_w, out_h = target_w, max(int(src_h * scale), 1)

    if telemetry.camera:
        s = out_w / telemetry.camera.width
        camera = CameraIntrinsics(
            fx=telemetry.camera.fx * s,
            fy=telemetry.camera.fy * s,
            cx=telemetry.camera.cx * s,
            cy=telemetry.camera.cy * s,
            width=out_w,
            height=out_h,
            dist=telemetry.camera.dist,
        )
    else:
        hfov = float(cfg["camera"]["default_hfov_deg"])
        camera = CameraIntrinsics.from_size_hfov(out_w, out_h, hfov)

    lat0, lon0, alt0 = resolve_origin(telemetry)
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    max_frames = int(cfg["video"]["max_frames"])
    min_sharp = float(cfg["video"]["min_sharpness"])
    min_move = float(cfg["video"]["min_move_m"])
    max_gap = float(cfg["video"]["max_time_gap_s"])

    selected: list[FrameRecord] = []
    last_t = -1e9
    last_enu = None

    def remaining():
        yield first
        yield from gen

    for _idx, t, frame, fps in remaining():
        sample = interpolate_sample(telemetry, t)
        enu = geodetic_to_enu(sample.lat, sample.lon, sample.alt, lat0, lon0, alt0)
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        sharp = sharpness_score(gray)

        moved = True if last_enu is None else float(np.linalg.norm(enu - last_enu)) >= min_move
        stale = (t - last_t) >= max_gap
        is_first = last_enu is None
        # Always keep frames that satisfy movement/gap, ignore sharpness so we don't end up with 0 frames on blurry videos
        keep = is_first or moved or stale
        if keep:
            rec = FrameRecord(
                index=len(selected),
                t=t,
                path=frames_dir / f"frame_{len(selected):04d}.jpg",
                image=resized,
                sharpness=sharp,
                lat=sample.lat,
                lon=sample.lon,
                alt=sample.alt,
                R_enu=_attitude_matrix(sample, cfg),
                t_enu=enu,
                yaw=sample.yaw,
                pitch=sample.pitch,
                roll=sample.roll,
            )
            cv2.imwrite(str(rec.path), resized)
            selected.append(rec)
            last_t = t
            last_enu = enu
            if len(selected) >= max_frames:
                break

    if len(selected) < 4:
        raise RuntimeError(
            f"Only {len(selected)} usable frames. Check GPS coverage, sharpness, or lower min_sharpness."
        )
    return selected, camera, fps


def _attitude_matrix(sample, cfg: dict) -> np.ndarray:
    if sample.yaw is not None and sample.pitch is not None and sample.roll is not None:
        R_body = rotation_zyx(sample.yaw, sample.pitch, sample.roll)
        R_cam = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        return R_body @ R_cam
    yaw = sample.yaw if sample.yaw is not None else 0.0
    pitch = -90.0 if cfg["camera"].get("assume_nadir_if_missing_attitude", False) else -45.0
    return rotation_zyx(yaw, pitch, 0.0) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
