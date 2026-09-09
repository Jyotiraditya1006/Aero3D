from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from aero3d.geo import lerp
from aero3d.types import CameraIntrinsics, GeoSample, Telemetry


def load_telemetry(path: str | Path) -> Telemetry:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return _from_json(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return _from_csv(p)
    if p.suffix.lower() in {".mp4", ".mov", ".avi"}:
        return _from_video_srt(p)
    raise ValueError(f"Unsupported telemetry format: {p.suffix}")

def _from_video_srt(video_path: Path) -> Telemetry:
    import subprocess
    import imageio_ffmpeg
    import re
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_srt = video_path.with_suffix(".srt")
    subprocess.run([ffmpeg_exe, "-y", "-i", str(video_path), "-map", "0:s:0", str(out_srt)], capture_output=True)
    
    def _dummy_telemetry():
        # Creates a basic 1-minute dummy flight starting at 0,0,0
        return Telemetry(samples=[
            GeoSample(t=0.0, lat=0.0, lon=0.0, alt=50.0, yaw=0.0, pitch=-90.0, roll=0.0),
            GeoSample(t=60.0, lat=0.0000001, lon=0.0, alt=50.0, yaw=0.0, pitch=-90.0, roll=0.0)
        ])

    if not out_srt.exists() or out_srt.stat().st_size == 0:
        print("Warning: No embedded telemetry found in video. Falling back to pure Visual Odometry.")
        return _dummy_telemetry()
    
    content = out_srt.read_text(encoding="utf-8", errors="replace")
    samples = []
    
    # Parse SRT blocks
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 3: continue
        
        # 00:00:00,000 --> 00:00:00,033
        time_match = re.search(r"(\d+):(\d+):(\d+),(\d+)\s*-->", lines[1])
        if not time_match: continue
        
        h, m, s, ms = map(int, time_match.groups())
        t = h * 3600 + m * 60 + s + ms / 1000.0
        
        text = " ".join(lines[2:])
        
        lat_m = re.search(r"\[latitude:\s*([\-\.\d]+)\]", text)
        lon_m = re.search(r"\[longitude:\s*([\-\.\d]+)\]", text)
        alt_m = re.search(r"\[rel_alt:\s*([\-\.\d]+)", text) or re.search(r"\[altitude:\s*([\-\.\d]+)", text)
        if not (lat_m and lon_m):
            continue
            
        samples.append(
            GeoSample(
                t=t,
                lat=float(lat_m.group(1)),
                lon=float(lon_m.group(1)),
                alt=float(alt_m.group(1)) if alt_m else 50.0,
                yaw=None, pitch=None, roll=None
            )
        )
        
    if not samples:
        print("Warning: Could not parse GPS from embedded subtitles. Falling back to pure Visual Odometry.")
        return _dummy_telemetry()
        
    samples.sort(key=lambda s: s.t)
    return Telemetry(samples=samples)


def _camera_from_dict(d: dict | None) -> CameraIntrinsics | None:
    if not d:
        return None
    if "fx" in d and "width" in d:
        return CameraIntrinsics(
            fx=float(d["fx"]),
            fy=float(d.get("fy", d["fx"])),
            cx=float(d.get("cx", d["width"] / 2)),
            cy=float(d.get("cy", d["height"] / 2)),
            width=int(d["width"]),
            height=int(d["height"]),
        )
    if "hfov_deg" in d and "width" in d:
        return CameraIntrinsics.from_size_hfov(int(d["width"]), int(d["height"]), float(d["hfov_deg"]))
    return None


def _from_json(path: Path) -> Telemetry:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    
    # Check for Google Earth Studio format
    ge_frames = None
    if isinstance(data, list) and len(data) > 0 and "coordinate" in data[0]:
        ge_frames = data
    elif isinstance(data, dict) and "cameras" in data and isinstance(data["cameras"], list) and len(data["cameras"]) > 0 and "coordinate" in data["cameras"][0]:
        ge_frames = data["cameras"]
        
    if ge_frames is not None:
        fps = data.get("fps", 30.0) if isinstance(data, dict) else 30.0
        for i, row in enumerate(ge_frames):
            coord = row.get("coordinate", {})
            rot = row.get("rotation", {})
            samples.append(
                GeoSample(
                    t=float(row.get("time", row.get("t", i / fps))),
                    lat=float(coord.get("latitude", 0.0)),
                    lon=float(coord.get("longitude", 0.0)),
                    alt=float(coord.get("altitude", 0.0)),
                    yaw=float(rot.get("z", 0.0)) if "z" in rot else None,
                    pitch=float(rot.get("x", 0.0)) if "x" in rot else None,
                    roll=float(rot.get("y", 0.0)) if "y" in rot else None,
                )
            )
        samples.sort(key=lambda s: s.t)
        return Telemetry(samples=samples)

    # Standard format
    for row in data.get("samples", data.get("telemetry", [])):
        samples.append(
            GeoSample(
                t=float(row.get("t", row.get("time", row.get("timestamp", 0.0)))),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                alt=float(row.get("alt", row.get("altitude", row.get("rel_alt", 0.0)))),
                yaw=_opt(row, "yaw", "heading"),
                pitch=_opt(row, "pitch"),
                roll=_opt(row, "roll"),
            )
        )
    samples.sort(key=lambda s: s.t)
    cam = _camera_from_dict(data.get("camera"))
    origin = data.get("origin", {})
    return Telemetry(
        samples=samples,
        camera=cam,
        origin_lat=origin.get("lat"),
        origin_lon=origin.get("lon"),
        origin_alt=origin.get("alt"),
        extra={k: v for k, v in data.items() if k not in {"samples", "telemetry", "camera", "origin"}},
    )


def _opt(row: dict, *keys: str) -> float | None:
    for k in keys:
        if k in row and row[k] is not None:
            return float(row[k])
    return None


def _from_csv(path: Path) -> Telemetry:
    samples: list[GeoSample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lower = {k.lower().strip(): v for k, v in row.items() if k}
            t = float(lower.get("t", lower.get("time", lower.get("timestamp", len(samples)))))
            lat = float(lower["lat"])
            lon = float(lower["lon"])
            alt = float(lower.get("alt", lower.get("altitude", 0.0)))
            samples.append(
                GeoSample(
                    t=t,
                    lat=lat,
                    lon=lon,
                    alt=alt,
                    yaw=_csv_opt(lower, "yaw", "heading"),
                    pitch=_csv_opt(lower, "pitch"),
                    roll=_csv_opt(lower, "roll"),
                )
            )
    samples.sort(key=lambda s: s.t)
    return Telemetry(samples=samples)


def _csv_opt(row: dict, *keys: str) -> float | None:
    for k in keys:
        if k in row and row[k] not in ("", None):
            return float(row[k])
    return None


def interpolate_sample(telemetry: Telemetry, t: float) -> GeoSample:
    samples = telemetry.samples
    if not samples:
        raise ValueError("Telemetry is empty")
    if t <= samples[0].t:
        return samples[0]
    if t >= samples[-1].t:
        return samples[-1]
    lo, hi = 0, len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    span = max(b.t - a.t, 1e-9)
    u = (t - a.t) / span
    yaw = pitch = roll = None
    if a.yaw is not None and b.yaw is not None:
        from aero3d.geo import interp_angle_deg

        yaw = interp_angle_deg(a.yaw, b.yaw, u)
    if a.pitch is not None and b.pitch is not None:
        pitch = lerp(a.pitch, b.pitch, u)
    if a.roll is not None and b.roll is not None:
        roll = lerp(a.roll, b.roll, u)
    return GeoSample(
        t=t,
        lat=lerp(a.lat, b.lat, u),
        lon=lerp(a.lon, b.lon, u),
        alt=lerp(a.alt, b.alt, u),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def resolve_origin(telemetry: Telemetry) -> tuple[float, float, float]:
    if telemetry.origin_lat is not None:
        return (
            float(telemetry.origin_lat),
            float(telemetry.origin_lon or 0.0),
            float(telemetry.origin_alt or 0.0),
        )
    s0 = telemetry.samples[0]
    telemetry.origin_lat, telemetry.origin_lon, telemetry.origin_alt = s0.lat, s0.lon, s0.alt
    return s0.lat, s0.lon, s0.alt


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))
