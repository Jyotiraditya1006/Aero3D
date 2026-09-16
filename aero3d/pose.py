from __future__ import annotations

import cv2
import numpy as np

from aero3d.types import CameraIntrinsics, FrameRecord


def refine_poses_visual(
    frames: list[FrameRecord],
    camera: CameraIntrinsics,
) -> dict:
    """Scale-correct visual odometry, then rigidly align to GPS ENU translations."""
    if len(frames) < 3:
        return {"vo_inliers_mean": 0.0, "aligned": False}

    K = camera.K
    # We will track the Camera Center in World coordinates (C_w)
    # and the Camera Rotation (R_cw = R_{world->cam})
    C_w = [np.zeros(3)]
    R_cw = [np.eye(3)]
    
    inliers = []
    orb = cv2.ORB_create(nfeatures=2500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    prev_gray = cv2.cvtColor(frames[0].image, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        
        R_inc = np.eye(3)
        t_inc = np.array([0.0, 0.0, 1.0])
        n_inl = 0
        
        if prev_des is not None and des is not None and len(prev_kp) >= 8 and len(kp) >= 8:
            matches = bf.knnMatch(prev_des, des, k=2)
            good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
            
            if len(good) >= 12:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good])
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R_inc, t_inc, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
                    n_inl = int(mask_pose.sum()) if mask_pose is not None else 0
                    t_inc = t_inc.reshape(3)
                    
        inliers.append(n_inl)
        
        # R_inc is R_{prev->curr}. t_inc is t_{prev->curr}
        # C_curr = C_prev - R_prev.T @ (R_inc.T @ t_inc)
        R_curr = R_inc @ R_cw[-1]
        C_curr = C_w[-1] - R_curr.T @ t_inc
        
        R_cw.append(R_curr)
        C_w.append(C_curr)
        
        prev_kp, prev_des = kp, des

    vo = np.stack(C_w)
    gps = np.stack([f.t_enu for f in frames])
    vo_len = np.linalg.norm(vo[-1] - vo[0])
    gps_len = np.linalg.norm(gps[-1] - gps[0])
    
    if vo_len < 1e-6:
        return {"vo_inliers_mean": float(np.mean(inliers) if inliers else 0), "aligned": False}

    if gps_len < 1e-6:
        # Fallback: We have no GPS! Use the true visual tracking (ORB VO) to assemble the video.
        time_span = frames[-1].t - frames[0].t if frames else 1.0
        vo_speed = vo_len / max(time_span, 1.0)
        # Scale VO up to a reasonable drone flight speed (e.g., 5m/s)
        scale = 5.0 / max(vo_speed, 1e-3)
        vo_s = vo * scale
        
        for i, f in enumerate(frames):
            f.t_enu = vo_s[i] + np.array([0.0, 0.0, 50.0])
            f.R_enu = R_cw[i]
                
        return {
            "vo_inliers_mean": float(np.mean(inliers) if inliers else 0),
            "vo_scale": float(scale),
            "gps_path_m": float(5.0 * time_span),
            "aligned": True,
            "has_gps": False,
        }

    scale = gps_len / vo_len
    vo_s = vo * scale
    R_align, t_align = _umeyama(vo_s, gps)
    aligned = (R_align @ vo_s.T).T + t_align

    # Blend GPS translation (metric / georef anchor) with VO-smoothed path.
    alpha = 0.35
    blended = alpha * aligned + (1.0 - alpha) * gps
    for i, f in enumerate(frames):
        f.t_enu = blended[i]
        # Align the local VO camera rotation to the global GPS ENU coordinate system!
        # R_cw is World->Cam. R_align maps VO-World to ENU-World.
        f.R_enu = R_cw[i] @ R_align.T

    return {
        "vo_inliers_mean": float(np.mean(inliers) if inliers else 0),
        "vo_scale": float(scale),
        "gps_path_m": float(gps_len),
        "aligned": True,
        "has_gps": True,
    }


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    cov = (Y.T @ X) / src.shape[0]
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_d - R @ mu_s
    return R, t
