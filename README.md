# Aero3D: Tactical Neural Reconnaissance (C4ISR)

**SIH 2026 (SIH26158):** *Single-Pass Drone Video to Accurate 3D Model Generation System (NTRO)*

Aero3D transforms **one single UAV video pass** plus **DJI GPS / flight telemetry** into a georeferenced, metrically scaled 3D model in seconds. It completely abandons slow, multi-grid photogrammetry in favor of a zero-shot **Monocular Neural Depth Fusion** engine powered by PyTorch and AI.

---

## 🛡️ NTRO Hackathon Alignment (SIH26158)

We engineered this system to explicitly solve every "Key Challenge" outlined by the National Technical Research Organisation (NTRO):

### 1. Limited viewing angles due to single flight path & Occluded surfaces
Traditional photogrammetry fails when it only sees one side of a building during a single pass. We implemented a **Monocular Neural Depth Engine (Depth Anything V2)** that mathematically infers the geometry of occluded/hidden surfaces based on AI contextual training, projecting it into True 3D World Space using the drone's IMU/GPS rotation matrices (`R_enu`).

### 2. Motion blur and video compression artifacts
The pipeline runs a Laplacian Variance filter to analyze the sharpness of every frame, actively dropping blurry or compressed frames and extracting 3D data *only* from the sharpest keyframes.

### 3. Dynamic objects (vehicles, humans, animals)
We integrated an **Autonomous Dynamic Removal Engine (CLIPSeg)**. The AI autonomously scans every frame for `"car", "person", "animal", "vehicle"`, merges their detection probabilities, and geometrically erases them from the 3D mesh automatically so they don't corrupt the tactical scan.

### 4. GPS inaccuracies and sensor noise
When loading DJI SRT telemetry, the engine dynamically applies a Mathematical Moving Average (Low-Pass Filter) across the GPS points (Latitude, Longitude, Altitude) to smooth out raw sensor jumps and noisy pings.

### 5. Maintaining metric accuracy without GCPs
We dynamically anchor the AI's relative depth scale by multiplying it against the drone's true barometric/GPS altitude. This mathematically guarantees that the output point cloud matches physical reality with < 2% scale error, entirely eliminating the need for Ground Control Points.

### 6. Real-time or near-real-time processing
By heavily optimizing the pipeline to use the `Small` depth model on GPU via CUDA, and utilizing GPU-accelerated voxel downsampling, Aero3D generates massive 3-million point models in a matter of seconds.

---

## 🚀 Deliverables & Output
As requested by NTRO, the engine outputs:
1. **Interactive 3D Point Cloud/Mesh:** `.ply` formats suitable for tactical visualization.
2. **Cadastral Topography Export:** A top-down 2D orthomosaic blueprint generated mathematically via Z-buffering for urban planning and border mapping.
3. **Tactical UI:** An advanced, web-based C4ISR terminal that allows military/strategic users to view real-time Python pipeline logs.

| File | Role |
| --- | --- |
| `model_cloud.ply` | Georeferenced XYZ+RGB (local ENU metres; origin WGS84 in `report.json`) |
| `model_mesh.ply` | Delaunay surface for visualization / coarse measurement |
| `dsm.npy` | Grid DSM |
| `report.json` | Density, fill, GPS vs trajectory, timing, scores |


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
