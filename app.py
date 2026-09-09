from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aero3d.pipeline import run_pipeline
from aero3d.synthetic import generate_synthetic_mission

ROOT = Path(__file__).resolve().parent


def main() -> None:
    st.set_page_config(page_title="Aero3D — Single-pass 3D", layout="wide", page_icon="🛩️")
    st.title("Aero3D")
    st.caption(
        "SIH26158 · NTRO — georeferenced 3D from a **single UAV video pass** "
        "(GPS-scaled VO + multi-view stereo)."
    )

    with st.sidebar:
        st.header("Inputs")
        mode = st.radio("Source", ["Synthetic demo", "Upload mission"], index=0)
        run = st.button("Generate 3D model", type="primary", use_container_width=True)

    video_file = tel_file = None
    if mode == "Upload mission":
        c1, c2 = st.columns(2)
        with c1:
            video_file = st.file_uploader("Drone video (1080p/4K)", type=["mp4", "avi", "mov", "mkv"])
        with c2:
            tel_file = st.file_uploader("Telemetry JSON or CSV (GPS + optional IMU/intrinsics)", type=["json", "csv"])
        st.markdown(
            "Optional fields in JSON: `camera` (fx, fy, cx, cy, width, height), "
            "`yaw`/`pitch`/`roll`, RTK-corrected `lat`/`lon`/`alt`."
        )

    if run:
        work = ROOT / "outputs" / "ui"
        work.mkdir(parents=True, exist_ok=True)
        bar = st.progress(0, text="Starting…")

        def progress(msg: str, frac: float) -> None:
            bar.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            if mode == "Synthetic demo":
                info = generate_synthetic_mission(work / "mission")
                video_path, tel_path = info["video"], info["telemetry"]
            else:
                if not video_file or not tel_file:
                    st.error("Upload both a video and telemetry file.")
                    st.stop()
                video_path = work / video_file.name
                tel_path = work / tel_file.name
                video_path.write_bytes(video_file.getbuffer())
                tel_path.write_bytes(tel_file.getbuffer())

            result = run_pipeline(video_path, tel_path, work / "model", progress=progress)
            st.session_state["result"] = result
            bar.progress(1.0, text="Done")
        except Exception as exc:
            st.exception(exc)
            st.stop()

    result = st.session_state.get("result")
    if result is None:
        _landing()
        return

    m = result.metrics
    scores = m["scores"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Points", f"{m['point_count']:,}")
    k2.metric("Density", f"{m['density_per_m2']:.1f} / m²")
    k3.metric("Height range", f"{m['height_range_m']:.1f} m")
    k4.metric("Realtime factor", f"{m['realtime_factor']:.1f}×")
    k5.metric("Overall score", f"{scores['overall_0_100']:.0f} / 100")

    tab_view, tab_dsm, tab_report, tab_eval = st.tabs(
        ["3D model & measure", "DSM", "Quality report", "Evaluation criteria"]
    )
    with tab_view:
        fig = _cloud_figure(result.points, result.colors, result.trajectory)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Measure (ENU metres from mission origin)")
        a1, a2 = st.columns(2)
        with a1:
            p0 = st.text_input("Point A (e, n, u)", value=_fmt_pt(result.points[0] if len(result.points) else None))
        with a2:
            p1 = st.text_input(
                "Point B (e, n, u)",
                value=_fmt_pt(result.points[min(len(result.points) - 1, 50)] if len(result.points) else None),
            )
        try:
            A = np.fromstring(p0, sep=",")
            B = np.fromstring(p1, sep=",")
            if A.size == 3 and B.size == 3:
                dist = float(np.linalg.norm(A - B))
                st.success(f"3D distance **{dist:.2f} m** · Δh = **{abs(A[2] - B[2]):.2f} m**")
        except Exception:
            st.warning("Enter three comma-separated numbers per point.")

        cld = result.output_dir / "model_cloud.ply"
        msh = result.output_dir / "model_mesh.ply"
        dl1, dl2, dl3 = st.columns(3)
        if cld.exists():
            dl1.download_button("Download point cloud (PLY)", data=cld.read_bytes(), file_name="model_cloud.ply")
        if msh.exists():
            dl2.download_button("Download mesh (PLY)", data=msh.read_bytes(), file_name="model_mesh.ply")
        rpt = result.output_dir / "report.json"
        if rpt.exists():
            dl3.download_button("Download report (JSON)", data=rpt.read_bytes(), file_name="report.json")

    with tab_dsm:
        if result.dsm is not None:
            st.image(_dsm_preview(result.dsm), caption="Digital surface model (local ENU up / metres)", use_container_width=True)
        origin = result.origin
        st.write(
            f"Origin WGS84: lat **{origin['lat']:.6f}**, lon **{origin['lon']:.6f}**, alt **{origin['alt']:.1f} m**"
        )

    with tab_report:
        for note in m.get("notes", []):
            st.info(note)
        st.json(m)

    with tab_eval:
        st.markdown(Path(ROOT / "docs" / "PROBLEM_STATEMENT.md").read_text(encoding="utf-8"))


def _landing() -> None:
    st.markdown(
        """
        ### What this system does
        1. Selects sharp, spatially spaced frames from **one** flight-line video.
        2. Fuses **GPS** (and IMU when present) with visual odometry for metric scale — no dense GCPs required.
        3. Runs pose-aware **stereo matching** to recover terrain, roofs, visible facades, roads, and vegetation.
        4. Filters high-flow pairs (movers / blur), builds a colored cloud, DSM, and surface mesh.
        5. Scores accuracy proxies: density, DSM fill, trajectory vs GPS, throughput.

        Use **Synthetic demo** if the NTRO dataset has not been issued yet.
        """
    )
    st.markdown("See `docs/PROBLEM_STATEMENT.md` for Desired Output and Evaluation Criteria tables.")


def _cloud_figure(points: np.ndarray, colors: np.ndarray, traj: np.ndarray):
    rng = np.random.default_rng(2)
    if points.shape[0] > 12000:
        idx = rng.choice(points.shape[0], 12000, replace=False)
        pts, cols = points[idx], colors[idx]
    else:
        pts, cols = points, colors
    rgb = np.clip(cols * 255, 0, 255).astype(int)
    color_str = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.8, color=color_str),
                name="Scene",
            ),
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines+markers",
                marker=dict(size=3, color="#ffdd55"),
                line=dict(color="#ffdd55", width=4),
                name="UAV path",
            ),
        ]
    )
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="data",
            bgcolor="#0e1117",
        },
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h"),
    )
    return fig


def _dsm_preview(dsm: np.ndarray) -> np.ndarray:
    vis = dsm.copy()
    finite = np.isfinite(vis)
    if not finite.any():
        return np.zeros((*dsm.shape, 3), dtype=np.uint8)
    lo, hi = np.nanpercentile(vis[finite], [2, 98])
    norm = np.clip((vis - lo) / max(hi - lo, 1e-6), 0, 1)
    norm[~finite] = 0
    # simple height colormap
    r = np.clip(1.5 * norm, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(norm - 0.45)), 0, 1)
    b = np.clip(1.2 * (1 - norm), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~finite] = 0.08
    return (rgb * 255).astype(np.uint8)


def _fmt_pt(p) -> str:
    if p is None:
        return "0, 0, 0"
    return f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}"


if __name__ == "__main__":
    main()
