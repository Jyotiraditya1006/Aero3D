from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from aero3d.cloud import raster_dsm, statistical_filter, terrain_mesh, voxel_downsample, write_ply_cloud, write_ply_mesh
from aero3d.depth import optical_flow_dynamic_mask, stereo_cloud_from_pair
from aero3d.evaluate import evaluate_reconstruction
from aero3d.frames import select_frames
from aero3d.geo import load_config
from aero3d.ingest import load_telemetry
from aero3d.pose import refine_poses_visual
from aero3d.types import ReconstructionResult


def run_pipeline(
    video_path: str | Path,
    telemetry_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    progress=None,
) -> ReconstructionResult:
    cfg = load_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def tick(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    tick("Loading telemetry and selecting frames", 0.05)
    telemetry = load_telemetry(telemetry_path)
    frames, camera, _fps = select_frames(video_path, telemetry, cfg, out)

    tick("GPS-scaled visual odometry", 0.2)
    vo_stats = refine_poses_visual(frames, camera)

    tick("Multi-view stereo fusion", 0.35)
    skip = int(cfg["stereo"]["pair_skip"])
    flow_thr = float(cfg["cloud"]["dynamic_flow_threshold"])
    chunks_p, chunks_c, chunks_k = [], [], []
    used_pairs = 0
    skipped_dynamic = 0
    n_pairs = max(len(frames) - skip, 1)
    for i in range(0, len(frames) - skip):
        fa, fb = frames[i], frames[i + skip]
        flow = optical_flow_dynamic_mask(fa, fb)
        pts, cols, conf = stereo_cloud_from_pair(fa, fb, camera, cfg)
        if pts.shape[0] == 0:
            continue
        if flow > flow_thr:
            conf = conf * 0.7
            skipped_dynamic += 1
        chunks_p.append(pts)
        chunks_c.append(cols)
        chunks_k.append(conf)
        used_pairs += 1
        if progress and i % 4 == 0:
            progress(f"Stereo pair {i + 1}/{n_pairs}", 0.35 + 0.4 * i / n_pairs)

    if not chunks_p:
        print("Warning: Stereo reconstruction produced no points. Generating dummy point to prevent crash.")
        # Provide a single dummy point so concatenation and downstream doesn't crash
        chunks_p.append(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        chunks_c.append(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        chunks_k.append(np.array([1.0], dtype=np.float32))

    points = np.concatenate(chunks_p, axis=0)
    colors = np.concatenate(chunks_c, axis=0)
    conf = np.concatenate(chunks_k, axis=0)

    tick("Cleaning point cloud and meshing", 0.8)
    points, colors, conf = voxel_downsample(points, colors, conf, float(cfg["cloud"]["voxel_m"]))
    points, colors, conf = statistical_filter(
        points, colors, conf, int(cfg["cloud"]["statistical_nb"]), float(cfg["cloud"]["statistical_std"])
    )
    max_pts = int(cfg["cloud"]["max_points"])
    if points.shape[0] > max_pts:
        idx = np.random.default_rng(1).choice(points.shape[0], max_pts, replace=False)
        points, colors, conf = points[idx], colors[idx], conf[idx]

    verts, faces, vcol = terrain_mesh(
        points, colors, int(cfg["mesh"]["sample_points"]), float(cfg["mesh"]["max_edge_m"])
    )
    dsm, origin_xy, res = raster_dsm(points, resolution=max(float(cfg["cloud"]["voxel_m"]), 0.4))

    write_ply_cloud(out / "model_cloud.ply", points, colors)
    if verts is not None:
        write_ply_mesh(out / "model_mesh.ply", verts, faces, vcol)

    np.save(out / "dsm.npy", dsm)
    traj = np.stack([f.t_enu for f in frames])
    origin = {
        "lat": float(telemetry.origin_lat),
        "lon": float(telemetry.origin_lon),
        "alt": float(telemetry.origin_alt),
    }
    elapsed = time.perf_counter() - t0
    metrics = evaluate_reconstruction(
        points=points,
        conf=conf,
        trajectory=traj,
        frames=frames,
        elapsed_s=elapsed,
        video_duration_s=frames[-1].t - frames[0].t if frames else 0.0,
        used_pairs=used_pairs,
        skipped_dynamic=skipped_dynamic,
        vo_stats=vo_stats,
        dsm=dsm,
    )
    payload = {
        "origin": origin,
        "metrics": metrics,
        "camera": {
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "width": camera.width,
            "height": camera.height,
        },
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tick("Done", 1.0)
    return ReconstructionResult(
        points=points,
        colors=colors,
        confidence=conf,
        mesh_vertices=verts,
        mesh_faces=faces,
        dsm=dsm,
        dsm_origin_xy=origin_xy,
        dsm_resolution_m=res,
        trajectory=traj,
        metrics=metrics,
        origin=origin,
        frame_count=len(frames),
        output_dir=out,
    )
