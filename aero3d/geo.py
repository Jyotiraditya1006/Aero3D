from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path | None = None) -> dict:
    default = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg_path = Path(path) if path else default
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def interp_angle_deg(a: float, b: float, u: float) -> float:
    da = ((b - a + 180.0) % 360.0) - 180.0
    return (a + da * u) % 360.0


def rotation_zyx(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Body-to-ENU using yaw (heading), pitch, roll in degrees."""
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    return Rz @ Ry @ Rx


def wgs84_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat_r) ** 2)
    x = (n + alt) * np.cos(lat_r) * np.cos(lon_r)
    y = (n + alt) * np.cos(lat_r) * np.sin(lon_r)
    z = (n * (1.0 - e2) + alt) * np.sin(lat_r)
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu_matrix(lat0: float, lon0: float) -> np.ndarray:
    lat_r, lon_r = np.deg2rad(lat0), np.deg2rad(lon0)
    sl, cl = np.sin(lat_r), np.cos(lat_r)
    so, co = np.sin(lon_r), np.cos(lon_r)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def geodetic_to_enu(
    lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float
) -> np.ndarray:
    r = wgs84_to_ecef(lat, lon, alt)
    r0 = wgs84_to_ecef(lat0, lon0, alt0)
    return ecef_to_enu_matrix(lat0, lon0) @ (r - r0)


def enu_to_geodetic(
    e: float, n: float, u: float, lat0: float, lon0: float, alt0: float
) -> tuple[float, float, float]:
    """Small-area inverse: ENU metres back to approximate lat/lon/alt."""
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * np.deg2rad(lat0))
    m_per_deg_lon = 111412.84 * np.cos(np.deg2rad(lat0))
    lat = lat0 + n / m_per_deg_lat
    lon = lon0 + e / max(m_per_deg_lon, 1e-6)
    alt = alt0 + u
    return float(lat), float(lon), float(alt)
