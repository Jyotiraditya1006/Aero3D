from __future__ import annotations

import cv2
import numpy as np

from aero3d.types import CameraIntrinsics, FrameRecord


def stereo_cloud_from_pair(
    fa: FrameRecord,
    fb: FrameRecord,
    camera: CameraIntrinsics,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense cloud from optical-flow correspondences + known metric poses."""
    baseline = float(np.linalg.norm(fa.t_enu - fb.t_enu))
    if baseline < 0.25:
        return _empty()

    pts, cols, conf = _flow_triangulate(fa, fb, camera, cfg)
    extra, ecol, econf = _sgbm_cloud(fa, fb, camera, cfg)
    if extra.shape[0]:
        pts = np.concatenate([pts, extra], axis=0) if pts.shape[0] else extra
        cols = np.concatenate([cols, ecol], axis=0) if cols.shape[0] else ecol
        conf = np.concatenate([conf, econf], axis=0) if conf.shape[0] else econf
    return pts, cols, conf


def _projection(K: np.ndarray, R_cw: np.ndarray, t_w: np.ndarray) -> np.ndarray:
    Rt = np.hstack([R_cw.T, -(R_cw.T @ t_w.reshape(3, 1))])
    return K @ Rt


def _flow_triangulate(fa, fb, camera, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g0 = cv2.cvtColor(fa.image, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(fb.image, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    h, w = g0.shape
    stride = 4
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    dx = flow[ys, xs, 0]
    dy = flow[ys, xs, 1]
    mag = np.hypot(dx, dy)
    keep = mag > 0.8
    u1 = xs[keep].astype(np.float32)
    v1 = ys[keep].astype(np.float32)
    u2 = u1 + dx[keep]
    v2 = v1 + dy[keep]
    inb = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)
    u1, v1, u2, v2 = u1[inb], v1[inb], u2[inb], v2[inb]
    if u1.size < 80:
        return _empty()

    K = camera.K
    Pa = _projection(K, fa.R_enu, fa.t_enu)
    Pb = _projection(K, fb.R_enu, fb.t_enu)
    pts4 = cv2.triangulatePoints(Pa, Pb, np.vstack([u1, v1]), np.vstack([u2, v2]))
    pts = (pts4[:3] / np.clip(pts4[3], 1e-8, None)).T

    Xc = (fa.R_enu.T @ (pts.T - fa.t_enu.reshape(3, 1))).T
    zmin, zmax = float(cfg["stereo"]["min_depth_m"]), float(cfg["stereo"]["max_depth_m"])
    ok = (Xc[:, 2] > zmin) & (Xc[:, 2] < zmax) & np.isfinite(pts).all(axis=1)
    # reprojection sanity
    proj = Pa @ np.vstack([pts.T, np.ones((1, len(pts)))])
    proj = proj[:2] / np.clip(proj[2], 1e-6, None)
    err = np.hypot(proj[0] - u1, proj[1] - v1)
    ok &= err < 6.0
    pts, u1, v1, Xc = pts[ok], u1[ok], v1[ok], Xc[ok]
    if pts.shape[0] == 0:
        return _empty()
    ui = np.clip(np.round(u1).astype(int), 0, w - 1)
    vi = np.clip(np.round(v1).astype(int), 0, h - 1)
    colors = fa.image[vi, ui][:, ::-1] / 255.0
    conf = np.clip(1.0 / (1.0 + err[ok] / 2.0), 0.1, 1.0).astype(np.float32)
    return pts.astype(np.float32), colors.astype(np.float32), conf


def _sgbm_cloud(fa, fb, camera, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = camera.K.astype(np.float64)
    dist = camera.dist.astype(np.float64)
    size = (camera.width, camera.height)
    R_rel = fb.R_enu.T @ fa.R_enu
    t_rel = fb.R_enu.T @ (fa.t_enu - fb.t_enu)
    if float(np.linalg.norm(t_rel)) < 0.3:
        return _empty()
    try:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, dist, K, dist, size, R_rel, t_rel.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        m1l, m1r = cv2.initUndistortRectifyMap(K, dist, R1, P1, size, cv2.CV_32FC1)
        m2l, m2r = cv2.initUndistortRectifyMap(K, dist, R2, P2, size, cv2.CV_32FC1)
        img_l = cv2.remap(fa.image, m1l, m1r, cv2.INTER_LINEAR)
        img_r = cv2.remap(fb.image, m2l, m2r, cv2.INTER_LINEAR)
    except cv2.error:
        return _empty()

    nd = int(cfg["stereo"]["num_disparities"])
    nd = max(16, nd - (nd % 16))
    bs = int(cfg["stereo"]["block_size"])
    if bs % 2 == 0:
        bs += 1
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=nd,
        blockSize=bs,
        P1=8 * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        uniquenessRatio=int(cfg["stereo"]["uniqueness_ratio"]),
        speckleWindowSize=int(cfg["stereo"]["speckle_window_size"]),
        speckleRange=int(cfg["stereo"]["speckle_range"]),
        disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = matcher.compute(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY))
    disp = disp.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    pts_cam = cv2.reprojectImageTo3D(np.nan_to_num(disp, nan=0.0), Q)
    z = pts_cam[:, :, 2]
    zmin, zmax = float(cfg["stereo"]["min_depth_m"]), float(cfg["stereo"]["max_depth_m"])
    valid = np.isfinite(disp) & (disp > 0) & (z > zmin) & (z < zmax)
    ys, xs = np.where(valid)
    if ys.size == 0:
        return _empty()
    step = max(1, ys.size // 20000)
    ys, xs = ys[::step], xs[::step]
    cam_pts = pts_cam[ys, xs]
    colors = img_l[ys, xs][:, ::-1] / 255.0
    R_rect = fa.R_enu @ R1.T
    world = (R_rect @ cam_pts.T).T + fa.t_enu
    conf = np.full((world.shape[0],), 0.45, dtype=np.float32)
    return world.astype(np.float32), colors.astype(np.float32), conf


def _empty():
    z = np.zeros((0, 3), np.float32)
    return z, z, np.zeros((0,), np.float32)


def optical_flow_dynamic_mask(fa: FrameRecord, fb: FrameRecord) -> float:
    g0 = cv2.cvtColor(fa.image, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(fb.image, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    return float(np.median(mag))
