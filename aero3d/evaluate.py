from __future__ import annotations

from typing import Any

import numpy as np

from aero3d.ingest import haversine_m
from aero3d.types import FrameRecord


def evaluate_reconstruction(
    points: np.ndarray,
    conf: np.ndarray,
    trajectory: np.ndarray,
    frames: list[FrameRecord],
    elapsed_s: float,
    video_duration_s: float,
    used_pairs: int,
    skipped_dynamic: int,
    vo_stats: dict,
    dsm: np.ndarray | None,
) -> dict[str, Any]:
    n = int(points.shape[0])
    area = _footprint_area_m2(points) if n else 0.0
    density = n / max(area, 1e-3)
    gps_span = 0.0
    if len(frames) >= 2:
        gps_span = haversine_m(frames[0].lat, frames[0].lon, frames[-1].lat, frames[-1].lon)
    traj_len = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))) if len(trajectory) > 1 else 0.0
    dsm_fill = 0.0
    if dsm is not None and dsm.size:
        dsm_fill = float(np.isfinite(dsm).mean())
    z_rng = float(points[:, 2].max() - points[:, 2].min()) if n else 0.0
    realtime_ratio = elapsed_s / max(video_duration_s, 1e-3)

    score_density = float(np.clip(density / 8.0, 0, 1))
    score_fill = dsm_fill
    score_speed = float(np.clip(2.0 / max(realtime_ratio, 1e-3), 0, 1))
    score_geo = float(np.clip(1.0 - abs(traj_len - gps_span) / max(gps_span, 1.0), 0, 1))
    overall = 100.0 * (0.35 * score_density + 0.25 * score_fill + 0.2 * score_geo + 0.2 * score_speed)

    return {
        "point_count": n,
        "mean_confidence": float(conf.mean()) if n else 0.0,
        "footprint_area_m2": area,
        "density_per_m2": density,
        "height_range_m": z_rng,
        "dsm_fill_ratio": dsm_fill,
        "gps_path_m": gps_span,
        "trajectory_length_m": traj_len,
        "scale_mismatch_m": abs(traj_len - gps_span),
        "selected_frames": len(frames),
        "stereo_pairs_used": used_pairs,
        "pairs_skipped_dynamic": skipped_dynamic,
        "elapsed_s": elapsed_s,
        "video_span_s": video_duration_s,
        "realtime_factor": realtime_ratio,
        "vo": vo_stats,
        "scores": {
            "density": score_density,
            "completeness": score_fill,
            "georef": score_geo,
            "speed": score_speed,
            "overall_0_100": overall,
        },
        "notes": _notes(n, density, dsm_fill, skipped_dynamic, vo_stats),
    }


def _footprint_area_m2(points: np.ndarray) -> float:
    e = points[:, 0]
    n = points[:, 1]
    return float(max(e.max() - e.min(), 0.5) * max(n.max() - n.min(), 0.5))


def _notes(n, density, fill, skipped, vo_stats) -> list[str]:
    notes = []
    if n < 2000:
        notes.append("Sparse cloud — increase frames or check stereo baseline / camera attitude.")
    if density < 4:
        notes.append("Low point density for fine facade detail.")
    if fill < 0.4:
        notes.append("DSM has large holes (single-pass occlusions or failed matches).")
        if skipped:
            notes.append(
                f"{skipped} pairs had high optical flow (expected on a moving UAV; used as a confidence downweight, not a hard drop)."
            )
    if not vo_stats.get("aligned"):
        notes.append("Visual odometry did not align; GPS/IMU translations used as-is.")
    if not notes:
        notes.append("Reconstruction completed with usable coverage for visualization and coarse measurement.")
    return notes
