# Aero3D

SIH 2026 **SIH26158** — *Single-Pass Drone Video to Accurate 3D Model Generation System* (NTRO).

Turns **one UAV video pass** plus **GPS / flight metadata** into a georeferenced, metrically scaled 3D model: colored point cloud, surface mesh, DSM, trajectory, and a quality report. Optional IMU, baro, camera intrinsics, and RTK/PPK improve pose and scale; dense GCPs are not required.

The official problem text includes the placeholder *“Add Desired Output and Evaluation Criteria table here”*. Those tables are filled in [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md).

## Pipeline

```
Video 1080p/4K + GPS (+ IMU/intrinsics)
        │
        ▼
 Sharp / spaced frame pick
        │
        ▼
 Visual odometry  ×  GPS scale  →  ENU poses
        │
        ▼
 Pose-rectified SGBM stereo  →  fuse clouds
        │
        ▼
 Drop high-flow pairs (movers / blur)
        │
        ▼
 Voxel + statistical clean  →  PLY cloud / mesh / DSM / JSON report
```

Single-pass limits (one side of buildings, occlusions) are treated as **low confidence**, not invented geometry.

## Quick start

```bash
cd aero3d
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**UI**

```bash
streamlit run app.py
```

Use *Synthetic demo* if the organisers have not issued video yet. Or upload `mp4` + `telemetry.json`.

**CLI**

```bash
python -m aero3d demo --out outputs/demo
python -m aero3d reconstruct --video path\to\flight.mp4 --telemetry path\to\telemetry.json --out outputs/run
```

## Telemetry schema

Mandatory: per-sample `t` (seconds), `lat`, `lon`, `alt` (metres). Optional: `yaw`, `pitch`, `roll` (degrees), `camera` intrinsics.

```json
{
  "origin": { "lat": 28.6139, "lon": 77.2090, "alt": 210.0 },
  "camera": { "fx": 1100, "fy": 1100, "cx": 480, "cy": 270, "width": 960, "height": 540 },
  "samples": [
    { "t": 0.0, "lat": 28.6139, "lon": 77.2090, "alt": 265.0, "yaw": 90, "pitch": -55, "roll": 0 }
  ]
}
```

CSV with the same column names is also accepted.

## Outputs

| File | Role |
| --- | --- |
| `model_cloud.ply` | Georeferenced XYZ+RGB (local ENU metres; origin WGS84 in `report.json`) |
| `model_mesh.ply` | Delaunay surface for visualization / coarse measurement |
| `dsm.npy` | Grid DSM |
| `report.json` | Density, fill, GPS vs trajectory, timing, scores |

## Mapping to NTRO challenges

| Challenge | Approach in this repo |
| --- | --- |
| One flight path / missing facades | Stereo only where baseline and overlap exist; DSM holes reported |
| Blur / compression | Laplacian frame gate; speckle-filtered SGBM |
| Illumination | Local stereo (not global photometric SfM) |
| Dynamic objects | Optical-flow pair rejection |
| GPS noise | VO path blended with GPS; RTK/PPK consumed as better `lat/lon/alt` |
| Near real-time | Frame cap + voxel cap in `configs/default.yaml` |
| Occlusions | No backside hallucination; completeness via DSM fill |
| Metric accuracy without GCPs | GPS/RTK scale; IMU attitudes when present |

When the live dataset arrives, drop video + metadata into the UI; tune `video.max_frames` and `stereo.pair_skip` for 4K length vs speed.
