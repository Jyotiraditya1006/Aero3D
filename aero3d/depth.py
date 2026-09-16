from __future__ import annotations

import cv2
import numpy as np

from aero3d.types import CameraIntrinsics, FrameRecord


def stereo_cloud_from_pair(
    fa: FrameRecord, fb: FrameRecord, camera: CameraIntrinsics, cfg: dict, ai_models=None, target_object=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense cloud from optical-flow correspondences + known metric poses."""
    baseline = float(np.linalg.norm(fa.t_enu - fb.t_enu))
    if baseline < 0.25:
        return _empty()

    pts, cols, conf = _flow_triangulate(fa, fb, camera, cfg)
    extra, ecol, econf = _sgbm_cloud(fa, fb, camera, cfg, ai_models=ai_models, target_object=target_object)
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


def _sgbm_cloud(fa, fb, camera, cfg, ai_models=None, target_object=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        
    # AI Target Isolation (CLIPSeg)
    if ai_models is not None and target_object is not None:
        from PIL import Image
        import torch
        import torch.nn.functional as F
        
        img_rgb = cv2.cvtColor(fa.image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        proc = ai_models["clipseg_proc"]
        model = ai_models["clipseg_model"]
        device_str = "cuda" if ai_models["device"] == 0 else "cpu"
        
        inputs = proc(text=[target_object], images=[pil_img], padding="max_length", return_tensors="pt").to(device_str)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.unsqueeze(1)
            
        h, w = fa.image.shape[:2]
        mask_tensor = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
        mask = torch.sigmoid(mask_tensor[0, 0]).cpu().numpy()
        
        # Filter valid ys, xs against the AI mask
        ai_mask = mask[ys, xs] > 0.4
        ys, xs = ys[ai_mask], xs[ai_mask]
        
    if ys.size == 0:
        return _empty()
    # High density sampling! (step=2 gives ~25% of all pixels, millions of points!)
    step = 2
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

def init_ai_depth_pipeline():
    """Initializes the Depth Anything V2 model via HuggingFace transformers."""
    from transformers import pipeline, CLIPSegProcessor, CLIPSegForImageSegmentation
    import torch
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading Depth Anything V2 on {'GPU' if device == 0 else 'CPU'}...")
    # Using the tiny/small model which takes ~100MB VRAM and runs instantly
    depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    
    print("Loading AI Target Isolation Engine (CLIPSeg)...")
    clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    if device == 0:
        clipseg_model.to("cuda")
        
    return {
        "depth": depth_pipe,
        "clipseg_proc": clipseg_processor,
        "clipseg_model": clipseg_model,
        "device": device
    }

def ai_depth_cloud(
    fa: FrameRecord, 
    camera: CameraIntrinsics, 
    depth_models: dict,
    target_object: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a dense, photorealistic point cloud from a single frame using AI Depth."""
    from PIL import Image
    import torch
    
    # Run the image through the Neural Network
    img_rgb = cv2.cvtColor(fa.image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    result = depth_models["depth"](pil_img)
    
    # The model outputs a high-precision float tensor. We use this instead of the 8-bit quantized PIL image 
    # to avoid the "sliced pyramid" terracing artifacts!
    depth_map = result["predicted_depth"].cpu().numpy()
    
    # Normalize disparity (larger is closer) to 0.0 - 1.0
    depth_map = depth_map - depth_map.min()
    depth_map = depth_map / (depth_map.max() + 1e-6)
    
    # Add a realistic offset so the ratio of Furthest/Closest is 2:1
    # instead of 100:1. This stops the Z-axis from stretching into a mountain!
    inv_depth = depth_map + 1.0 
    raw_Z = 1.0 / inv_depth
    
    # Dynamically anchor the Neural Depth scale to the true physical GPS altitude!
    # This mathematically prevents the "deck of cards" smearing artifact across frames.
    dist_to_ground = max(15.0, fa.t_enu[2])
    current_median = float(np.median(raw_Z))
    scale_factor = dist_to_ground / max(current_median, 1e-3)
    Z = raw_Z * scale_factor
    
    # Create pixel grid
    h, w = Z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply camera intrinsics to unproject rays
    cx, cy = camera.cx, camera.cy
    fx, fy = camera.fx, camera.fy
    
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Optional AI Target Isolation using CLIPSeg!
    valid_mask = np.ones((h, w), dtype=bool)
    if target_object:
        print(f"Isolating target object: '{target_object}' using CLIPSeg...")
        proc = depth_models["clipseg_proc"]
        model = depth_models["clipseg_model"]
        device_str = "cuda" if depth_models["device"] == 0 else "cpu"
        
        inputs = proc(text=[target_object], images=[pil_img], padding="max_length", return_tensors="pt").to(device_str)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.unsqueeze(1)
            
        # CLIPSeg outputs 352x352 usually, resize back to original image
        import torch.nn.functional as F
        mask_tensor = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
        mask = torch.sigmoid(mask_tensor[0, 0]).cpu().numpy()
        
        # Threshold the mask (e.g., > 0.4 probability)
        valid_mask = mask > 0.4
    
    # Flatten into 3D points in Camera Space
    pts_cam = np.stack((X, Y, Z), axis=-1)[valid_mask].reshape(-1, 3)
    colors = img_rgb[valid_mask].reshape(-1, 3) / 255.0
    
    # CRITICAL: Transform points from Local Camera Space to Global World Space!
    # Without this, all frames overlap at origin (0,0,0) creating a glitchy mess.
    world_pts = (fa.R_enu @ pts_cam.T).T + fa.t_enu
    
    # Sample down slightly to prevent crashing RAM when merging infinite frames
    step = 2 
    world_pts = world_pts[::step]
    colors = colors[::step]
    
    # Assume high confidence for AI depth
    conf = np.full((world_pts.shape[0],), 0.9, dtype=np.float32)
    
    return world_pts.astype(np.float32), colors.astype(np.float32), conf
