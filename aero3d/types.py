from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_size_hfov(cls, width: int, height: int, hfov_deg: float = 70.0) -> CameraIntrinsics:
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        fy = fx
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


@dataclass
class GeoSample:
    t: float
    lat: float
    lon: float
    alt: float
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    vx: float | None = None
    vy: float | None = None
    vz: float | None = None


@dataclass
class Telemetry:
    samples: list[GeoSample]
    camera: CameraIntrinsics | None = None
    origin_lat: float | None = None
    origin_lon: float | None = None
    origin_alt: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameRecord:
    index: int
    t: float
    path: Path | None
    image: np.ndarray
    sharpness: float
    lat: float
    lon: float
    alt: float
    R_enu: np.ndarray
    t_enu: np.ndarray
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


@dataclass
class ReconstructionResult:
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    mesh_vertices: np.ndarray | None
    mesh_faces: np.ndarray | None
    dsm: np.ndarray | None
    dsm_origin_xy: tuple[float, float] | None
    dsm_resolution_m: float | None
    trajectory: np.ndarray
    metrics: dict[str, Any]
    origin: dict[str, float]
    frame_count: int
    output_dir: Path
