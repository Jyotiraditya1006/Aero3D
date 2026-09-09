# SIH26158 — Single-Pass Drone Video to Accurate 3D Model Generation System

**Organization:** National Technical Research Organisation (NTRO)  
**Category:** Software · **Theme:** Robotics and Drones  
**ID:** 26158 / SIH26158

The official listing leaves a placeholder (“Add Desired Output and Evaluation Criteria table here”). The tables below are derived from the stated reconstruction targets, key challenges, mandatory inputs, and the requirement that models support visualization, measurement, and analysis.

## Desired output

| ID | Deliverable | Format / CRS | What “good” looks like |
| --- | --- | --- | --- |
| D1 | Georeferenced dense point cloud | PLY (XYZ+RGB+confidence); optional LAS/LAZ · WGS84 / local ENU | Terrain, structures, roads, vegetation visible from the single pass; metric coordinates |
| D2 | Textured / vertex-colored 3D mesh | PLY / OBJ · same CRS | Facades and rooftops where viewing geometry allows; holes tagged rather than hallucinated |
| D3 | Digital surface model | GeoTIFF or raster grid + world file | Elevation suitable for measurements and overlay |
| D4 | Recovered camera trajectory | JSON/CSV (timestamp, lat/lon/alt, R, t) | GPS-scaled visual odometry; IMU/RTK used when present |
| D5 | Quality & accuracy report | JSON (+ on-screen dashboard) | RMSE, density, completeness, timing, warnings (blur, GPS gaps, occlusions) |
| D6 | Interactive viewer | Web app | Orbit/inspect, measure 3D distance, download artefacts |
| D7 | Optional layers | GeoJSON footprints, dynamic-object mask, occlusion/confidence map | Supports analysis without claiming unseen surfaces as truth |

## Evaluation criteria

| Criterion | Weight | How it is scored |
| --- | --- | --- |
| Metric / geometric accuracy | 25% | Horizontal & vertical RMSE vs sparse GCPs or a reference cloud; scale error after GPS/RTK alignment; ASPRS-style check if GCPs exist |
| Completeness of scene classes | 20% | Terrain, rooftops, visible facades, roads/infrastructure, vegetation/obstacles recovered from **one** path (not multi-orbit coverage) |
| Georeferencing without dense GCPs | 15% | Trajectory vs flight GPS; ENU consistency; RTK/PPK/IMU improves score when provided |
| Robustness to operational video | 15% | Motion blur, compression, illumination, and moving objects (vehicles/people) do not destroy the static model |
| Processing time | 10% | Near-real-time or bounded offline time vs video duration (throughput, not just accuracy) |
| Measurement & visualization | 10% | Distances/heights in metres; textured model inspectable by a non-expert operator |
| Occlusion honesty & AI value | 5% | Unobserved surfaces marked low-confidence; optional completion clearly labelled |

**Dataset note:** organisers will provide video + metadata in real time. The repo includes a synthetic single-pass generator so the pipeline can be demonstrated before that drop.
