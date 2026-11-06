import numpy as np
import cv2
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

from core.processing_base import ProcessingBase
from norfair import Tracker, Detection


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing and Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _check_opencl() -> bool:
    """Test if OpenCL is available via cv2.UMat."""
    try:
        cv2.UMat(np.zeros((4, 4), dtype=np.uint8))
        return cv2.ocl.useOpenCL()
    except Exception:
        return False


def _denseflow_compute(prev_gray: np.ndarray,
                       gray: np.ndarray,
                       last_umat_flow,
                       fb_params: dict,
                       use_opencl: bool):
    """
    Computes dense Farneback optical flow.
    """
    if use_opencl:
        prev_u = cv2.UMat(prev_gray)
        cur_u  = cv2.UMat(gray)
        flags  = cv2.OPTFLOW_USE_INITIAL_FLOW if last_umat_flow is not None else 0
        last_umat_flow = cv2.calcOpticalFlowFarneback(
            prev_u, cur_u, last_umat_flow, flags=flags, **fb_params)
        fx_u, fy_u = cv2.split(last_umat_flow)
        mag_u, ang_u = cv2.cartToPolar(fx_u, fy_u)
        tmp = cv2.merge((mag_u, ang_u)).get()
        mag = tmp[..., 0]
        ang = tmp[..., 1]
    else:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, **fb_params)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        last_umat_flow = None
    return mag, ang, last_umat_flow


def _denseflow_visualize(mag: np.ndarray,
                         ang: np.ndarray,
                         binary_mask: np.ndarray,
                         supersat: bool,
                         use_opencl: bool) -> np.ndarray:
    """
    Creates an HSV->BGR representation of flow.
    """
    H, W = mag.shape
    hue = (ang * (90.0 / np.pi)).astype(np.uint8)

    if supersat:
        val = np.full((H, W), 255, np.uint8)
    else:
        m_min, m_max = float(mag.min()), float(mag.max())
        if m_max > m_min:
            val = ((mag - m_min) * (255.0 / (m_max - m_min))).astype(np.uint8)
        else:
            val = np.zeros((H, W), np.uint8)

    sat = np.full((H, W), 255, np.uint8)
    hsv = cv2.merge((hue, sat, val))

    if use_opencl:
        try:
            bgr_u = cv2.cvtColor(cv2.UMat(hsv), cv2.COLOR_HSV2BGR)
            bgr = bgr_u.get()
        except Exception:
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out = cv2.bitwise_and(bgr, bgr, mask=binary_mask)
    return out


def _apply_monochrome_mode(frame: np.ndarray, mode: str) -> np.ndarray:
    """Isolates the Red, Green, or Blue channel, or converts to standard Grayscale."""
    if len(frame.shape) != 3:
        return frame.copy()
    if mode == "Red Channel":
        return frame[:, :, 2].copy()
    elif mode == "Green Channel":
        return frame[:, :, 1].copy()
    elif mode == "Blue Channel":
        return frame[:, :, 0].copy()
    else:  # Grayscale
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Applies gamma correction using lookup tables (LUT)."""
    if abs(gamma - 1.0) < 1e-3:
        return image
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def _apply_circular_crop(gray: np.ndarray, crop_radius: int, center: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Applies circular crop to grayscale image using a custom center, returns (cropped_gray, crop_mask)."""
    H, W = gray.shape[:2]
    cx, cy = center
    crop_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(crop_mask, (cx, cy), int(crop_radius), 255, -1)
    cropped_gray = cv2.bitwise_and(gray, crop_mask)
    return cropped_gray, crop_mask


def _apply_rectangular_crop(gray: np.ndarray, crop_width: int, crop_height: int, center: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Applies rectangular crop to grayscale image using a custom center, returns (cropped_gray, crop_mask)."""
    H, W = gray.shape[:2]
    cx, cy = center
    crop_mask = np.zeros((H, W), dtype=np.uint8)
    x1 = max(0, cx - int(crop_width) // 2)
    y1 = max(0, cy - int(crop_height) // 2)
    x2 = min(W, cx + int(crop_width) // 2)
    y2 = min(H, cy + int(crop_height) // 2)
    cv2.rectangle(crop_mask, (x1, y1), (x2, y2), 255, -1)
    cropped_gray = cv2.bitwise_and(gray, crop_mask)
    return cropped_gray, crop_mask


def _get_crop_center(W: int, H: int, p: dict) -> Tuple[int, int]:
    """Returns crop center coordinates (crop_cx, crop_cy)."""
    if p.get('crop_use_abs_center', p.get('use_absolute_center', False)):
        cx = int(p.get('crop_center_x', p.get('center_x', W // 2)))
        cy = int(p.get('crop_center_y', p.get('center_y', H // 2)))
    else:
        cx = W // 2 + int(p.get('crop_offset_x', p.get('offset_x', 0)))
        cy = H // 2 + int(p.get('crop_offset_y', p.get('offset_y', 0)))
    return cx, cy


def _get_pellet_center(W: int, H: int, p: dict) -> Tuple[int, int]:
    """Returns attractant pellet center coordinates (pellet_cx, pellet_cy)."""
    if p.get('pellet_use_abs_center', p.get('use_absolute_center', False)):
        cx = int(p.get('pellet_center_x', p.get('center_x', W // 2)))
        cy = int(p.get('pellet_center_y', p.get('center_y', H // 2)))
    else:
        cx = W // 2 + int(p.get('pellet_offset_x', p.get('offset_x', 0)))
        cy = H // 2 + int(p.get('pellet_offset_y', p.get('offset_y', 0)))
    return cx, cy


def _get_center(W: int, H: int, p: dict) -> Tuple[int, int]:
    """Compatibility alias for _get_crop_center."""
    return _get_crop_center(W, H, p)


def _apply_crop_mode(gray: np.ndarray, p: dict, cx: int, cy: int, pellet_center: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """Applies selected crop mode (None, Circular, Rectangle) and excludes inner pellet area if pellet_radius > 0.
    Returns (cropped_gray, crop_mask, is_circular, crop_radius)."""
    H, W = gray.shape[:2]
    crop_type = p.get('crop_type', 'Circular' if p.get('circular_crop', False) else 'None')
    if crop_type == 'Circular':
        crop_radius = int(p.get('crop_radius', 150))
        gray_cropped, crop_mask = _apply_circular_crop(gray, crop_radius, (cx, cy))
        is_circ = True
    elif crop_type == 'Rectangle':
        crop_w = int(p.get('crop_width', 300))
        crop_h = int(p.get('crop_height', 300))
        gray_cropped, crop_mask = _apply_rectangular_crop(gray, crop_w, crop_h, (cx, cy))
        is_circ = False
        crop_radius = max(crop_w, crop_h) // 2
    else:
        crop_mask = np.full((H, W), 255, dtype=np.uint8)
        gray_cropped = gray
        is_circ = False
        crop_radius = 150

    pellet_radius = int(p.get('pellet_radius', 0))
    if pellet_radius > 0:
        if pellet_center is not None:
            pellet_cx, pellet_cy = pellet_center
        else:
            pellet_cx, pellet_cy = _get_pellet_center(W, H, p)
        cv2.circle(crop_mask, (pellet_cx, pellet_cy), pellet_radius, 0, -1)
        gray_cropped = cv2.bitwise_and(gray_cropped, crop_mask)

    return gray_cropped, crop_mask, is_circ, crop_radius


def _draw_distance_circles(frame: np.ndarray, num_circles: int, circular_crop: bool, crop_radius: int, pellet_center: Tuple[int, int], crop_center: Tuple[int, int], spatial_scale: float = 1.0, spatial_unit: str = "px", pellet_radius: int = 0, p: dict = None):
    """Draws crop contour outline centered at crop_center, attractant pellet and concentric distance reticules centered at pellet_center."""
    H, W = frame.shape[:2]
    pellet_cx, pellet_cy = pellet_center
    crop_cx, crop_cy = crop_center
    max_r = crop_radius if circular_crop else int(np.sqrt((W // 2) ** 2 + (H // 2) ** 2))
    
    crop_type = p.get('crop_type', 'Circular' if circular_crop else 'None') if p else ('Circular' if circular_crop else 'None')
    
    # Draw Crop Outline post-processing centered at crop_center (does not affect binarization)
    if p and p.get('show_crop_outline', True):
        if crop_type == 'Circular':
            cv2.circle(frame, (crop_cx, crop_cy), crop_radius, (255, 255, 0), 2, cv2.LINE_AA)
        elif crop_type == 'Rectangle':
            crop_w = int(p.get('crop_width', 300))
            crop_h = int(p.get('crop_height', 300))
            x1 = crop_cx - crop_w // 2
            y1 = crop_cy - crop_h // 2
            x2 = crop_cx + crop_w // 2
            y2 = crop_cy + crop_h // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2, cv2.LINE_AA)

    # Draw Attractant Pellet circle centered at pellet_center
    if pellet_radius > 0:
        cv2.circle(frame, (pellet_cx, pellet_cy), pellet_radius, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (pellet_cx, pellet_cy), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "Attractant Pellet", (pellet_cx + 4, pellet_cy - pellet_radius - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    if num_circles > 0:
        start_r = pellet_radius if pellet_radius > 0 else 0
        if crop_type == 'Circular':
            d_pc = float(np.hypot(pellet_cx - crop_cx, pellet_cy - crop_cy))
            max_dist = d_pc + float(crop_radius)
        elif crop_type == 'Rectangle':
            crop_w = int(p.get('crop_width', 300)) if p else 300
            crop_h = int(p.get('crop_height', 300)) if p else 300
            x1 = crop_cx - crop_w // 2
            y1 = crop_cy - crop_h // 2
            x2 = crop_cx + crop_w // 2
            y2 = crop_cy + crop_h // 2
            c1 = np.hypot(x1 - pellet_cx, y1 - pellet_cy)
            c2 = np.hypot(x2 - pellet_cx, y1 - pellet_cy)
            c3 = np.hypot(x1 - pellet_cx, y2 - pellet_cy)
            c4 = np.hypot(x2 - pellet_cx, y2 - pellet_cy)
            max_dist = max(c1, c2, c3, c4)
        else:
            d1 = np.hypot(pellet_cx, pellet_cy)
            d2 = np.hypot(W - pellet_cx, pellet_cy)
            d3 = np.hypot(pellet_cx, H - pellet_cy)
            d4 = np.hypot(W - pellet_cx, H - pellet_cy)
            max_dist = max(d1, d2, d3, d4)

        usable_r = max(1.0, float(max_dist - start_r))
        step_r = usable_r / float(num_circles)

        for i in range(1, num_circles + 1):
            r_edge = i * step_r
            r_abs = int(round(start_r + r_edge))
            cv2.circle(frame, (pellet_cx, pellet_cy), r_abs, (120, 120, 120), 1, cv2.LINE_AA)
            if spatial_scale != 1.0 or spatial_unit != "px":
                r_unit = r_edge * spatial_scale
                txt = f"{int(round(r_edge))}px ({r_unit:.1f}{spatial_unit})"
            else:
                txt = f"{int(round(r_edge))}px"
def _get_cv2_colormap(palette_name: str) -> int:
    """Maps palette string to corresponding OpenCV colormap enum."""
    match palette_name:
        case 'Inferno': return cv2.COLORMAP_INFERNO
        case 'Jet': return cv2.COLORMAP_JET
        case 'Turbo': return cv2.COLORMAP_TURBO
        case 'Viridis': return cv2.COLORMAP_VIRIDIS
        case 'Plasma': return cv2.COLORMAP_PLASMA
        case 'Hot': return cv2.COLORMAP_HOT
        case 'HSV': return cv2.COLORMAP_HSV
        case _: return cv2.COLORMAP_INFERNO


def _draw_proximity_density(frame: np.ndarray,
                            num_circles: int,
                            circular_crop: bool,
                            crop_radius: int,
                            pellet_center: Tuple[int, int],
                            crop_center: Tuple[int, int],
                            crop_mask: Optional[np.ndarray],
                            centroids: List[Tuple[int, int]],
                            mask: Optional[np.ndarray],
                            p: dict) -> np.ndarray:
    """
    Renders proximity density heatmap on concentric rings around the attractant.
    Supports Centroid mode and Surface mode with Absolute or Relative (%) scaling.
    Optimized for high-framerate realtime video processing.
    """
    if not p or not p.get('enable_proximity', False) or num_circles <= 0:
        return frame

    opacity = float(np.clip(float(p.get('proximity_opacity', 0.5)), 0.0, 1.0))
    if opacity <= 0.0:
        return frame

    H, W = frame.shape[:2]
    pellet_cx, pellet_cy = pellet_center
    crop_cx, crop_cy = crop_center
    pellet_radius = int(p.get('pellet_radius', 0))
    start_r = pellet_radius if pellet_radius > 0 else 0

    crop_type = p.get('crop_type', 'Circular' if circular_crop else 'None')
    if crop_type == 'Circular':
        d_pc = float(np.hypot(pellet_cx - crop_cx, pellet_cy - crop_cy))
        max_dist = d_pc + float(crop_radius)
    elif crop_type == 'Rectangle':
        crop_w = int(p.get('crop_width', 300))
        crop_h = int(p.get('crop_height', 300))
        x1 = crop_cx - crop_w // 2
        y1 = crop_cy - crop_h // 2
        x2 = crop_cx + crop_w // 2
        y2 = crop_cy + crop_h // 2
        c1 = np.hypot(x1 - pellet_cx, y1 - pellet_cy)
        c2 = np.hypot(x2 - pellet_cx, y1 - pellet_cy)
        c3 = np.hypot(x1 - pellet_cx, y2 - pellet_cy)
        c4 = np.hypot(x2 - pellet_cx, y2 - pellet_cy)
        max_dist = max(c1, c2, c3, c4)
    else:
        d1 = np.hypot(pellet_cx, pellet_cy)
        d2 = np.hypot(W - pellet_cx, pellet_cy)
        d3 = np.hypot(pellet_cx, H - pellet_cy)
        d4 = np.hypot(W - pellet_cx, H - pellet_cy)
        max_dist = max(d1, d2, d3, d4)

    usable_r = max(1.0, float(max_dist - start_r))
    step_r = usable_r / float(num_circles)

    radii = [start_r + i * step_r for i in range(num_circles + 1)]
    radii_int = [int(round(r)) for r in radii]

    density_mode = p.get('proximity_density_mode', 'Centroid')
    scale_mode = p.get('proximity_scale_mode', 'Absolute')
    palette_name = p.get('proximity_palette', 'Inferno')
    show_scale = p.get('proximity_show_scale', True)

    values = [0.0] * num_circles
    total_val = 0.0

    # Vectorized fast histogram computation
    if density_mode == 'Centroid':
        if centroids:
            pts = np.asarray(centroids, dtype=np.float32)
            dists = np.hypot(pts[:, 0] - pellet_cx, pts[:, 1] - pellet_cy)
            counts, _ = np.histogram(dists, bins=radii)
            values = [float(c) for c in counts]
            total_val = float(len(centroids))
    else:  # Surface mode
        if mask is not None and np.any(mask > 0):
            y_idx, x_idx = np.nonzero(mask)
            if len(x_idx) > 0:
                dists = np.hypot(x_idx - pellet_cx, y_idx - pellet_cy)
                counts, _ = np.histogram(dists, bins=radii)
                values = [float(c) for c in counts]
                total_val = float(len(x_idx))

    if scale_mode == 'Relative (%)':
        if total_val > 0:
            values = [(v / total_val) * 100.0 for v in values]
        else:
            values = [0.0] * num_circles

    v_min = 0.0
    v_max = max(values) if values else 1.0
    if v_max <= 0:
        v_max = 1.0

    cv2_cmap = _get_cv2_colormap(palette_name)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2_cmap).reshape(256, 3)

    # Render color rings into a dedicated overlay
    overlay = np.zeros_like(frame)
    for k in reversed(range(num_circles)):
        norm_v = np.clip((values[k] - v_min) / (v_max - v_min), 0.0, 1.0)
        lut_idx = int(norm_v * 255)
        color = tuple(int(c) for c in lut[lut_idx])
        cv2.circle(overlay, (pellet_cx, pellet_cy), radii_int[k + 1], color, -1)

    if start_r > 0:
        cv2.circle(overlay, (pellet_cx, pellet_cy), start_r, (0, 0, 0), -1)

    if crop_mask is not None:
        overlay[crop_mask == 0] = 0

    # Fast in-place alpha blend on active pixels only
    active_idx = np.nonzero(overlay.any(axis=2))
    if len(active_idx[0]) > 0:
        alpha = float(opacity)
        beta = 1.0 - alpha
        f_sub = frame[active_idx]
        o_sub = overlay[active_idx]
        frame[active_idx] = cv2.addWeighted(o_sub, alpha, f_sub, beta, 0.0)

    if show_scale:
        scale_pos = str(p.get('proximity_scale_position', 'Top Right'))
        custom_size = int(p.get('proximity_scale_size', 150))
        bar_w = 16
        bar_h = int(np.clip(custom_size, 40, H - 70))
        margin = 15

        if "Right" in scale_pos:
            bar_x = W - bar_w - margin
            panel_x1 = max(0, bar_x - 48)
            panel_x2 = min(W - 1, bar_x + bar_w + 6)
            text_x = bar_x - 44
        else:
            bar_x = margin + 48
            panel_x1 = max(0, margin)
            panel_x2 = min(W - 1, bar_x + bar_w + 6)
            text_x = margin + 4

        if "Top" in scale_pos:
            bar_y = 35
        else:
            bar_y = max(35, H - bar_h - 25)

        panel_y1 = max(0, bar_y - 20)
        panel_y2 = min(H - 1, bar_y + bar_h + 15)

        if bar_x >= 0 and bar_y + bar_h <= H:
            grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape((bar_h, 1))
            grad_bgr = cv2.applyColorMap(grad, cv2_cmap)
            grad_rect = np.repeat(grad_bgr, bar_w, axis=1)

            cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), -1)
            cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (100, 100, 100), 1)

            frame[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = grad_rect
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (220, 220, 220), 1)

            title = "Proximity" if density_mode == "Centroid" else "Surface"
            cv2.putText(frame, title, (text_x, bar_y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

            if scale_mode == 'Relative (%)':
                txt_max = f"{v_max:.1f}%"
                txt_min = f"{v_min:.1f}%"
            else:
                txt_max = f"{int(v_max)}" if density_mode == "Centroid" else f"{int(v_max)}px"
                txt_min = f"{int(v_min)}" if density_mode == "Centroid" else f"{int(v_min)}px"

            cv2.putText(frame, txt_max, (text_x, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, txt_min, (text_x, bar_y + bar_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def render_processed_frame(raw_frame: np.ndarray,
                           params: dict,
                           is_crop_to_selection: bool = False,
                           tracker: Optional[Any] = None,
                           tracking_dict: Optional[dict] = None) -> np.ndarray:
    """
    Renders a single frame through the complete visual processing pipeline
    (monochrome, gamma, crop, binarization, blobs, contours, tracking, proximity heatmap, reticules).
    Optionally crops output to the bounding box of the active ROI crop.
    """
    if raw_frame is None or not isinstance(raw_frame, np.ndarray):
        return None

    H, W = raw_frame.shape[:2]
    crop_cx, crop_cy = _get_crop_center(W, H, params)
    pellet_cx, pellet_cy = _get_pellet_center(W, H, params)
    num_circles = int(params.get('num_distance_circles', 5))

    # 1. Preprocessing
    if params.get('enable_preprocessing', True):
        gray = _apply_monochrome_mode(raw_frame, params.get('monochrome_mode', 'Grayscale'))
        if params.get('invert_image', False):
            gray = cv2.bitwise_not(gray)
        gamma = float(params.get('gamma', 1.0))
        if abs(gamma - 1.0) > 1e-3:
            gray = _adjust_gamma(gray, gamma)
        gray, crop_mask, is_circ, crop_radius = _apply_crop_mode(gray, params, crop_cx, crop_cy, (pellet_cx, pellet_cy))

        r1 = params.get("_ref_frame_1", None)
        r2 = params.get("_ref_frame_2", None)
        if params.get('remove_static', False) and r1 is not None and r2 is not None:
            if r1.shape == gray.shape and r2.shape == gray.shape:
                similarity_threshold = int(params.get('similarity_threshold', 15))
                diff_ref = cv2.absdiff(r1, r2)
                diff_cur = cv2.absdiff(gray, r1)
                _, mask_ref = cv2.threshold(diff_ref, similarity_threshold, 255, cv2.THRESH_BINARY_INV)
                _, mask_cur = cv2.threshold(diff_cur, similarity_threshold, 255, cv2.THRESH_BINARY_INV)
                static_mask = cv2.bitwise_and(mask_ref, mask_cur)
                gray[static_mask > 0] = 0
    else:
        gray = _apply_monochrome_mode(raw_frame, 'Grayscale')
        crop_mask = np.full((H, W), 255, dtype=np.uint8)
        is_circ = False
        crop_radius = 150

    # 2. Binarization
    blur_k = int(params.get('blur', 3))
    if blur_k > 0:
        blur_k = blur_k | 1
        gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    else:
        gray_blurred = gray

    block_size = int(params.get('block_size', 25))
    if block_size < 3: block_size = 3
    block_size = block_size | 1
    bin_thresh = int(params.get('bin_thresh', 15))

    mask = cv2.adaptiveThreshold(
        gray_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, bin_thresh
    )
    mask = cv2.bitwise_and(mask, crop_mask)

    morph_close = int(params.get('morph_close', 0))
    if morph_close > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close)

    if params.get('fill_contours', False):
        cnts_fill = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.drawContours(mask, cnts_fill, -1, 255, -1)

    # 3. Canvas
    if params.get('monochrome_mode') == 'RGB (Color)' and len(raw_frame.shape) == 3:
        frame_out = raw_frame.copy()
    else:
        frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    enable_contour_detection = params.get('enable_contour_detection', True)
    if not enable_contour_detection:
        if params.get('show_blobs', True):
            frame_out[mask > 0] = (0, 0, 255)
        if params.get('enable_proximity', False):
            frame_out = _draw_proximity_density(frame_out, num_circles, is_circ, crop_radius, (pellet_cx, pellet_cy), (crop_cx, crop_cy), crop_mask, [], mask, params)
        if params.get('show_distance_circles', False) or params.get('show_crop_outline', True) or int(params.get('pellet_radius', 0)) > 0:
            _draw_distance_circles(frame_out, num_circles, is_circ, crop_radius, (pellet_cx, pellet_cy), (crop_cx, crop_cy), float(params.get('spatial_scale', 1.0)), str(params.get('spatial_unit', 'px')), int(params.get('pellet_radius', 0)), params)
    else:
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        areas = [cv2.contourArea(c) for c in cnts]
        lower_thresh = float(params.get('lower_surface_thresh', 25))
        upper_thresh = float(params.get('upper_surface_thresh', 1000))
        thresh_mode = params.get('area_thresh_mode', 'Absolute (px^2)')

        filtered_cnts = []
        if areas:
            if thresh_mode == 'Relative (%)':
                lo_pct = np.clip(lower_thresh, 0.0, 100.0)
                hi_pct = np.clip(100.0 - upper_thresh, 0.0, 100.0)
                lo_bound = np.percentile(areas, lo_pct)
                hi_bound = np.percentile(areas, hi_pct)
                filtered_cnts = [c for c, a in zip(cnts, areas) if lo_bound <= a <= hi_bound]
            else:
                filtered_cnts = [c for c, a in zip(cnts, areas) if lower_thresh <= a <= upper_thresh]

        filtered_mask = np.zeros_like(mask)
        if filtered_cnts:
            cv2.drawContours(filtered_mask, filtered_cnts, -1, 255, -1)

        visu_format = params.get('visu_format', 'Frame')
        if visu_format == 'Mask':
            frame_out = np.zeros_like(raw_frame, dtype=np.uint8)
            if is_circ:
                frame_out[crop_mask == 0] = 0

        if params.get('show_blobs', True):
            frame_out[filtered_mask > 0] = (0, 0, 255)

        isolate_selection = params.get('isolate_selection', False)
        selection_mask = np.zeros_like(mask)

        centroids = []
        for c in filtered_cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx_obj, cy_obj = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                rect = cv2.minAreaRect(c)
                cx_obj, cy_obj = int(rect[0][0]), int(rect[0][1])
            centroids.append((cx_obj, cy_obj))

            if isolate_selection:
                cv2.drawContours(selection_mask, [c], -1, 255, -1)

            if params.get('show_boxes', True):
                rect = cv2.minAreaRect(c)
                box_pts = np.array(cv2.boxPoints(rect), dtype='int')
                cv2.drawContours(frame_out, [box_pts], -1, (0, 255, 0), 1, cv2.LINE_AA)

            if params.get('show_centroids', True):
                cv2.circle(frame_out, (cx_obj, cy_obj), 2, (0, 0, 255), -1, cv2.LINE_AA)

        if isolate_selection:
            if visu_format == 'Mask':
                frame_out[selection_mask == 255] = raw_frame[selection_mask == 255]
            else:
                frame_out[selection_mask != 255] = 0

        # Tracking & Trails rendering
        enable_tracking = params.get('enable_tracking', True)
        if tracker is not None and enable_tracking and centroids:
            try:
                from norfair import Detection
                norfair_detections = [Detection(np.array(pt)) for pt in centroids]
                tracked_objects = tracker.update(norfair_detections)
            except Exception:
                tracked_objects = []

            if tracking_dict is None:
                tracking_dict = {}

            trail_length = int(params.get("trail_length", 30))
            trail_thickness = max(1, int(params.get("trail_thickness", 2)))
            show_trail = params.get("show_trail", False)
            show_id = params.get("show_id", False)
            show_age = params.get("show_age", False)
            show_distance = params.get("show_distance", False)
            show_speed = params.get("show_speed", False)
            show_proximity = params.get("show_proximity", True)
            pellet_radius = float(params.get('pellet_radius', 0))

            for obj in tracked_objects:
                cx_obj, cy_obj = map(int, obj.estimate[0])
                dist_center = float(np.sqrt((cx_obj - pellet_cx) ** 2 + (cy_obj - pellet_cy) ** 2))
                dist = max(0.0, dist_center - pellet_radius)

                if obj.id not in tracking_dict:
                    tracking_dict[obj.id] = {
                        "color": tuple(int(x) for x in np.random.randint(50, 255, size=3)),
                        "points": [],
                        "distance": 0.0,
                        "age": obj.age
                    }

                tracking_dict[obj.id]["points"].append((cx_obj, cy_obj))
                tracking_dict[obj.id]["distance"] += getattr(obj, "last_distance", 0.0)
                tracking_dict[obj.id]["age"] = obj.age

                if show_trail:
                    trail = tracking_dict[obj.id]["points"][-trail_length:]
                    if len(trail) >= 2:
                        for i in range(1, len(trail)):
                            cv2.line(frame_out, trail[i - 1], trail[i],
                                     tracking_dict[obj.id]["color"], trail_thickness, cv2.LINE_AA)

                text_lines = []
                if show_id:        text_lines.append(f"ID:{obj.id}")
                if show_age:       text_lines.append(f"Age:{obj.age}")
                if show_distance:  text_lines.append(f"Dist:{tracking_dict[obj.id]['distance']:.1f}")
                if show_speed:
                    pts = tracking_dict[obj.id]["points"][-trail_length:]
                    if len(pts) >= 2:
                        path_len = sum(
                            float(np.sqrt((pts[k][0] - pts[k-1][0])**2 + (pts[k][1] - pts[k-1][1])**2))
                            for k in range(1, len(pts))
                        )
                        calib = float(params.get('spatial_scale', params.get('df_calibration', 1.0)))
                        fps = float(params.get('video_fps', params.get('df_fps', 30.0)))
                        speed = (path_len * calib * fps) / (len(pts) - 1)
                        text_lines.append(f"Speed:{speed:.2f}")
                if show_proximity: text_lines.append(f"Prox:{dist:.1f}")

                for i, line in enumerate(text_lines):
                    cv2.putText(
                        frame_out, line, (cx_obj + 4, cy_obj + 12 + i * 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
                    )

        if params.get('enable_proximity', False):
            frame_out = _draw_proximity_density(frame_out, num_circles, is_circ, crop_radius, (pellet_cx, pellet_cy), (crop_cx, crop_cy), crop_mask, centroids, mask, params)

        if params.get('show_distance_circles', False) or params.get('show_crop_outline', True) or int(params.get('pellet_radius', 0)) > 0:
            _draw_distance_circles(frame_out, num_circles, is_circ, crop_radius, (pellet_cx, pellet_cy), (crop_cx, crop_cy), float(params.get('spatial_scale', 1.0)), str(params.get('spatial_unit', 'px')), int(params.get('pellet_radius', 0)), params)

    if is_crop_to_selection:
        crop_type = params.get('crop_type', 'Circular' if is_circ else 'None')
        if crop_type == 'Circular':
            r = int(params.get('crop_radius', 150))
            x1, x2 = max(0, crop_cx - r), min(W, crop_cx + r)
            y1, y2 = max(0, crop_cy - r), min(H, crop_cy + r)
        elif crop_type == 'Rectangle':
            cw = int(params.get('crop_width', 300))
            ch = int(params.get('crop_height', 300))
            x1, x2 = max(0, crop_cx - cw // 2), min(W, crop_cx + cw // 2)
            y1, y2 = max(0, crop_cy - ch // 2), min(H, crop_cy + ch // 2)
        else:
            x1, x2, y1, y2 = 0, W, 0, H

        out_w = (x2 - x1) & ~1
        out_h = (y2 - y1) & ~1
        if out_w > 0 and out_h > 0:
            frame_out = frame_out[y1:y1 + out_h, x1:x1 + out_w]

    return frame_out


def _compute_denseflow_metrics_and_vis(mag: np.ndarray,
                                       ang: np.ndarray,
                                       p: dict,
                                       use_opencl: bool,
                                       crop_mask: np.ndarray) -> Tuple[np.ndarray, float, list]:
    lower_mag  = float(p.get('df_lower_mag_thresh', 1.5))
    erosion_it = int(p.get('df_erosion', 3))
    supersat   = p.get('df_supersat', True)
    calib = float(p.get('spatial_scale', p.get('df_calibration', 1.0)))
    fps = float(p.get('df_fps', 30.0))
    scale = calib * fps  # px/frame -> unit/s
    use_contours = p.get('df_use_contours', False)
    isolate = p.get('df_isolate_selection', False)
    show_arrows = p.get('df_show_arrows', True)
    lo_thresh = int(p.get('df_lower_surface_thresh', 25))
    hi_thresh = int(p.get('df_upper_surface_thresh', 100))

    binary = (mag >= lower_mag).astype(np.uint8) * 255
    if erosion_it > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.erode(binary, k, iterations=erosion_it)

    binary = cv2.bitwise_and(binary, crop_mask)

    mag_filt = mag.copy()
    mag_filt[binary == 0] = 0.0

    if not use_contours:
        valid_mags = mag_filt[mag_filt > 0]
        if len(valid_mags) > 0:
            mean_speed_px = float(np.mean(valid_mags))
        else:
            mean_speed_px = 0.0
        metric_val = round(mean_speed_px * scale, 3)
        raw_speeds_um_s = (valid_mags * scale).tolist() if p.get('collect_raw_speeds', False) else []
        flow_vis = _denseflow_visualize(mag_filt, ang, binary, supersat, use_opencl)
        return flow_vis, metric_val, raw_speeds_um_s
    else:
        cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        filtered_cnts = []
        if cnts:
            bboxes = [cv2.minAreaRect(c) for c in cnts]
            areas = [(b[1][0] * b[1][1]) for b in bboxes]
            if areas:
                lo = np.percentile(areas, lo_thresh)
                hi = np.percentile(areas, hi_thresh)
                filtered_cnts = [c for c, a in zip(cnts, areas) if lo <= a <= hi]
        
        selection_mask = np.zeros_like(binary)
        if filtered_cnts:
            cv2.fillPoly(selection_mask, filtered_cnts, 255)

        if isolate:
            binary = selection_mask.copy()
            mag_filt[binary == 0] = 0.0

        flow_vis = _denseflow_visualize(mag_filt, ang, binary, supersat, use_opencl)

        pixel_mags = mag[selection_mask > 0]
        raw_speeds_um_s = (pixel_mags * scale).tolist() if p.get('collect_raw_speeds', False) else []

        speeds = []
        if filtered_cnts:
            for c in filtered_cnts:
                c_mask = np.zeros_like(binary)
                cv2.fillPoly(c_mask, [c], 255)
                c_mags = mag[c_mask > 0]
                mean_mag = float(np.mean(c_mags)) if len(c_mags) > 0 else 0.0
                
                if mean_mag > 0:
                    speeds.append(mean_mag)
                
                if show_arrows and mean_mag > 0:
                    pts = c[:, 0, :]
                    ax = np.clip(pts[:, 0], 0, mag.shape[1]-1)
                    ay = np.clip(pts[:, 1], 0, mag.shape[0]-1)
                    angs_c = ang[ay, ax]
                    mean_ang = np.arctan2(np.mean(np.sin(angs_c)), np.mean(np.cos(angs_c)))
                    if mean_ang < 0: mean_ang += 2 * np.pi
                    circle = cv2.minEnclosingCircle(c)
                    cx, cy = map(int, circle[0])
                    scale_arrow = 5.0
                    end_x = int(cx + mean_mag * scale_arrow * np.cos(mean_ang))
                    end_y = int(cy + mean_mag * scale_arrow * np.sin(mean_ang))
                    cv2.arrowedLine(flow_vis, (cx, cy), (end_x, end_y), (0, 255, 0), 2, cv2.LINE_AA)

        if speeds:
            mean_speed_px = float(np.mean(speeds))
        else:
            mean_speed_px = 0.0
        
        metric_val = round(mean_speed_px * scale, 3)
        return flow_vis, metric_val, raw_speeds_um_s


def process_frame_denseflow_stateless(prev_gray: np.ndarray,
                                      gray: np.ndarray,
                                      params: dict,
                                      use_opencl: bool = False,
                                      crop_mask: np.ndarray = None):
    fb_params = {
        'pyr_scale':  params.get('df_pyr_scale',  0.5),
        'levels':     params.get('df_levels',      2),
        'winsize':    params.get('df_winsize',      9),
        'iterations': params.get('df_iterations',  2),
        'poly_n':     params.get('df_poly_n',       5),
        'poly_sigma': params.get('df_poly_sigma',   1.1),
    }
    mag, ang, _ = _denseflow_compute(prev_gray, gray, None, fb_params, use_opencl)
    if crop_mask is None:
        crop_mask = np.full(gray.shape[:2], 255, dtype=np.uint8)
    flow_vis, metric_val, mags_list = _compute_denseflow_metrics_and_vis(mag, ang, params, use_opencl, crop_mask)
    return flow_vis, metric_val, mags_list


# ─────────────────────────────────────────────────────────────────────────────
# Batch workers for ProcessPoolExecutor
# ─────────────────────────────────────────────────────────────────────────────

def batch_worker_process_chunk(video_path: str, start: int, end: int, params: dict):
    import cv2 as _cv2
    cap_local = _cv2.VideoCapture(video_path)
    cap_local.set(_cv2.CAP_PROP_POS_FRAMES, start)
    results = []
    try:
        for _ in range(end - start):
            ret, frame = cap_local.read()
            if not ret:
                break
            centroids = RadialContourProcessor.process_frame_stateless(frame, params)
            results.append(centroids)
    finally:
        cap_local.release()
    return results


def batch_worker_denseflow_chunk(video_path: str, start: int, end: int, params: dict):
    import cv2 as _cv2
    use_opencl = _check_opencl()

    cap = _cv2.VideoCapture(video_path)
    cap.set(_cv2.CAP_PROP_POS_FRAMES, start)

    results = []
    prev_gray = None

    try:
        for fi in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            
            H, W = frame.shape[:2]
            crop_cx, crop_cy = _get_crop_center(W, H, params)
            pellet_cx, pellet_cy = _get_pellet_center(W, H, params)

            if params.get('enable_preprocessing', True):
                gray = _apply_monochrome_mode(frame, params.get('monochrome_mode', 'Grayscale'))
                if params.get('invert_image', False):
                    gray = _cv2.bitwise_not(gray)
                gamma = float(params.get('gamma', 1.0))
                if abs(gamma - 1.0) > 1e-3:
                    gray = _adjust_gamma(gray, gamma)
                gray, crop_mask, _, _ = _apply_crop_mode(gray, params, crop_cx, crop_cy, (pellet_cx, pellet_cy))

                # Static removal filter in batch worker
                r1 = params.get("_ref_frame_1", None)
                r2 = params.get("_ref_frame_2", None)
                if params.get('remove_static', False) and r1 is not None and r2 is not None:
                    if r1.shape == gray.shape and r2.shape == gray.shape:
                        similarity_threshold = int(params.get('similarity_threshold', 15))
                        diff_ref = _cv2.absdiff(r1, r2)
                        diff_cur = _cv2.absdiff(gray, r1)
                        _, mask_ref = _cv2.threshold(diff_ref, similarity_threshold, 255, _cv2.THRESH_BINARY_INV)
                        _, mask_cur = _cv2.threshold(diff_cur, similarity_threshold, 255, _cv2.THRESH_BINARY_INV)
                        static_mask = _cv2.bitwise_and(mask_ref, mask_cur)
                        gray[static_mask > 0] = 0
            else:
                gray = _apply_monochrome_mode(frame, 'Grayscale')
                crop_mask = np.full((H, W), 255, dtype=np.uint8)

            if prev_gray is None:
                prev_gray = gray
                continue

            flow_vis, metric_val, mags = process_frame_denseflow_stateless(
                prev_gray, gray, params, use_opencl, crop_mask)
            results.append((start + fi, metric_val, mags))
            prev_gray = gray
    finally:
        cap.release()
    return results


def batch_worker_both_chunk(video_path: str, start: int, end: int, params: dict):
    import cv2 as _cv2
    use_opencl = _check_opencl()

    cap = _cv2.VideoCapture(video_path)
    cap.set(_cv2.CAP_PROP_POS_FRAMES, start)

    results = []
    prev_gray = None

    try:
        for fi in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            
            H, W = frame.shape[:2]
            crop_cx, crop_cy = _get_crop_center(W, H, params)
            pellet_cx, pellet_cy = _get_pellet_center(W, H, params)

            if params.get('enable_preprocessing', True):
                gray = _apply_monochrome_mode(frame, params.get('monochrome_mode', 'Grayscale'))
                if params.get('invert_image', False):
                    gray = _cv2.bitwise_not(gray)
                gamma = float(params.get('gamma', 1.0))
                if abs(gamma - 1.0) > 1e-3:
                    gray = _adjust_gamma(gray, gamma)
                gray, crop_mask, _, _ = _apply_crop_mode(gray, params, crop_cx, crop_cy, (pellet_cx, pellet_cy))

                # Static removal filter in batch worker
                r1 = params.get("_ref_frame_1", None)
                r2 = params.get("_ref_frame_2", None)
                if params.get('remove_static', False) and r1 is not None and r2 is not None:
                    if r1.shape == gray.shape and r2.shape == gray.shape:
                        similarity_threshold = int(params.get('similarity_threshold', 15))
                        diff_ref = _cv2.absdiff(r1, r2)
                        diff_cur = _cv2.absdiff(gray, r1)
                        _, mask_ref = _cv2.threshold(diff_ref, similarity_threshold, 255, _cv2.THRESH_BINARY_INV)
                        _, mask_cur = _cv2.threshold(diff_cur, similarity_threshold, 255, _cv2.THRESH_BINARY_INV)
                        static_mask = _cv2.bitwise_and(mask_ref, mask_cur)
                        gray[static_mask > 0] = 0
            else:
                gray = _apply_monochrome_mode(frame, 'Grayscale')
                crop_mask = np.full((H, W), 255, dtype=np.uint8)

            if prev_gray is None:
                prev_gray = gray
                continue

            fb_params = {
                'pyr_scale':  params.get('df_pyr_scale',  0.5),
                'levels':     params.get('df_levels',      2),
                'winsize':    params.get('df_winsize',      9),
                'iterations': params.get('df_iterations',  2),
                'poly_n':     params.get('df_poly_n',       5),
                'poly_sigma': params.get('df_poly_sigma',   1.1),
            }
            mag, ang, _ = _denseflow_compute(prev_gray, gray, None, fb_params, use_opencl)
            
            centroids = RadialContourProcessor.process_frame_stateless(frame, params)
            
            calib = float(params.get('df_calibration', 1.0))
            fps = float(params.get('df_fps', 30.0))
            scale = calib * fps
            
            speeds = []
            objects_data = []
            pellet_radius = float(params.get('pellet_radius', 0))
            for cx_obj, cy_obj in centroids:
                dist_center = float(np.sqrt((cx_obj - pellet_cx) ** 2 + (cy_obj - pellet_cy) ** 2))
                dist = max(0.0, dist_center - pellet_radius)
                if 0 <= cx_obj < mag.shape[1] and 0 <= cy_obj < mag.shape[0]:
                    spd = float(mag[cy_obj, cx_obj]) * scale
                else:
                    spd = 0.0
                speeds.append(spd)
                objects_data.append((cx_obj, cy_obj, dist, spd))
                    
            mean_speed = round(float(np.mean(speeds)), 3) if speeds else 0.0
            results.append((start + fi, mean_speed, speeds, objects_data))
            prev_gray = gray
    finally:
        cap.release()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Radial Contour Processor Class
# ─────────────────────────────────────────────────────────────────────────────

class RadialContourProcessor(ProcessingBase):
    """
    Processing engine for the Radial Contour Tracker.
    Handles color isolation, cropping, offset, gamma correction, separate
    preprocessing / processing flow, contour extraction, reticule drawing, and tracking.
    """

    def __init__(self,
                 params: Optional[Dict[str, Any]] = None,
                 *,
                 buffer_size: int = 10,
                 drop_policy: str = 'drop_new',
                 daemon: bool = True) -> None:
        super().__init__(params=params, buffer_size=buffer_size,
                         drop_policy=drop_policy, daemon=daemon)

        dist_func = params.get('distance_function', 'euclidean') if params else 'euclidean'
        dist_thresh = params.get('distance_threshold', 30) if params else 30

        self.tracker = Tracker(
            distance_function=dist_func,
            distance_threshold=dist_thresh,
        )
        self.tracking: Dict[int, Any] = {}
        self._last_reset_trigger: int = 0

        # Reference frames triggers for static removal
        self._ref_frame_1: Optional[np.ndarray] = None
        self._ref_frame_2: Optional[np.ndarray] = None
        self._last_r1_trigger = 0
        self._last_r2_trigger = 0
        self._last_clear_ref_trigger = 0

        self._opencl_available = _check_opencl()
        if not self._opencl_available:
            warnings.warn(
                "OpenCL not available – Optical flow running in CPU mode.",
                RuntimeWarning
            )
        self._df_prev_gray: Optional[np.ndarray] = None
        self._df_last_umat_flow = None

    def _get_random_color(self):
        return tuple(random.randint(50, 255) for _ in range(3))

    def _reset_denseflow_state(self):
        self._df_prev_gray = None
        self._df_last_umat_flow = None

    def _process_data(self, data: Any, p: Dict[str, Any]) -> Any:
        frame_idx = None
        if isinstance(data, tuple) and len(data) == 2:
            frame, frame_idx = data
        else:
            frame = data

        if not isinstance(frame, np.ndarray):
            return (frame, {}, [], None, []), frame_idx

        # Retrieve Center settings
        H, W = frame.shape[:2]
        crop_cx, crop_cy = _get_crop_center(W, H, p)
        pellet_cx, pellet_cy = _get_pellet_center(W, H, p)
        crop_center = (crop_cx, crop_cy)
        pellet_center = (pellet_cx, pellet_cy)

        enable_preprocessing = p.get('enable_preprocessing', True)
        active_processing = p.get('active', True)

        # 1. Preprocessing Stage (Monochrome conversion, Invert, Gamma correction, Crop & Pellet exclusion)
        if enable_preprocessing:
            gray = _apply_monochrome_mode(frame, p.get('monochrome_mode', 'Grayscale'))
            
            if p.get('invert_image', False):
                gray = cv2.bitwise_not(gray)

            gamma = float(p.get('gamma', 1.0))
            if abs(gamma - 1.0) > 1e-3:
                gray = _adjust_gamma(gray, gamma)
                
            gray, crop_mask, circular_crop, crop_radius = _apply_crop_mode(gray, p, crop_cx, crop_cy, pellet_center)

            # --- Reference frames capture triggers ---
            r1_trig = p.get('capture_r1_trigger', 0)
            if r1_trig > self._last_r1_trigger:
                self._ref_frame_1 = gray.copy()
                self._last_r1_trigger = r1_trig

            r2_trig = p.get('capture_r2_trigger', 0)
            if r2_trig > self._last_r2_trigger:
                self._ref_frame_2 = gray.copy()
                self._last_r2_trigger = r2_trig

            clear_trig = p.get('clear_ref_trigger', 0)
            if clear_trig > self._last_clear_ref_trigger:
                self._ref_frame_1 = None
                self._ref_frame_2 = None
                self._last_clear_ref_trigger = clear_trig

            # Ensure reference frame dimensions remain identical to frame shape
            if self._ref_frame_1 is not None and self._ref_frame_1.shape != gray.shape:
                self._ref_frame_1 = None
            if self._ref_frame_2 is not None and self._ref_frame_2.shape != gray.shape:
                self._ref_frame_2 = None

            # --- Static parasite removal filter ---
            if p.get('remove_static', False) and self._ref_frame_1 is not None and self._ref_frame_2 is not None:
                similarity_threshold = int(p.get('similarity_threshold', 15))
                diff_ref = cv2.absdiff(self._ref_frame_1, self._ref_frame_2)
                diff_cur = cv2.absdiff(gray, self._ref_frame_1)
                _, mask_ref = cv2.threshold(diff_ref, similarity_threshold, 255, cv2.THRESH_BINARY_INV)
                _, mask_cur = cv2.threshold(diff_cur, similarity_threshold, 255, cv2.THRESH_BINARY_INV)
                static_mask = cv2.bitwise_and(mask_ref, mask_cur)
                gray[static_mask > 0] = 0

        else:
            # Bypass preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            crop_mask = np.full((H, W), 255, dtype=np.uint8)
            circular_crop = False
            crop_radius = 150

        # 2. Skip Processing Stage if active is False
        if not active_processing:
            if p.get('monochrome_mode') == 'RGB (Color)' and len(frame.shape) == 3:
                frame_out = frame.copy()
            else:
                frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if p.get('show_distance_circles', False) or p.get('show_crop_outline', True) or int(p.get('pellet_radius', 0)) > 0:
                _draw_distance_circles(frame_out, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, float(p.get('spatial_scale', 1.0)), str(p.get('spatial_unit', 'px')), int(p.get('pellet_radius', 0)), p)
            return (frame_out, {}, [], None, []), frame_idx

        # 3. Processing Stage
        mode = p.get('processing_mode', 'contour')
        if mode == 'denseflow':
            res = self._process_denseflow_preprocessed(frame, gray, crop_mask, circular_crop, crop_radius, pellet_center, crop_center, p)
        elif mode == 'both':
            res = self._process_both_preprocessed(frame, gray, crop_mask, circular_crop, crop_radius, pellet_center, crop_center, p)
        else:
            res = self._process_contour_preprocessed(frame, gray, crop_mask, circular_crop, crop_radius, pellet_center, crop_center, p)

        return res, frame_idx

    # ── Preprocessed Processing Handlers ──────────────────────────────

    def _process_contour_preprocessed(self,
                                      frame: np.ndarray,
                                      gray: np.ndarray,
                                      crop_mask: np.ndarray,
                                      circular_crop: bool,
                                      crop_radius: int,
                                      pellet_center: Tuple[int, int],
                                      crop_center: Tuple[int, int],
                                      p: Dict[str, Any]) -> Tuple[np.ndarray, dict, list, Optional[np.ndarray], list]:
        """
        Contours extraction & tracking run on top of preprocessed grayscale canvas.
        """
        pellet_cx, pellet_cy = pellet_center
        crop_cx, crop_cy = crop_center
        H, W = gray.shape[:2]

        # Reset tracker check
        current_reset_trigger = p.get('reset_trigger', 0)
        if current_reset_trigger > self._last_reset_trigger:
            self.tracker = Tracker(
                distance_function=p.get('distance_function', 'euclidean'),
                distance_threshold=p.get('distance_threshold', 30),
            )
            self.tracking = {}
            self._last_reset_trigger = current_reset_trigger

        # Binarization
        blur_k = int(p.get('blur', 3))
        if blur_k > 0:
            blur_k = blur_k | 1
            gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        else:
            gray_blurred = gray

        block_size = int(p.get('block_size', 25))
        if block_size < 3: block_size = 3
        block_size = block_size | 1

        bin_thresh = int(p.get('bin_thresh', 15))

        mask = cv2.adaptiveThreshold(
            gray_blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, bin_thresh
        )

        mask = cv2.bitwise_and(mask, crop_mask)

        morph_close = int(p.get('morph_close', 0))
        if morph_close > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close)

        if p.get('fill_contours', False):
            cnts_fill = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cv2.drawContours(mask, cnts_fill, -1, 255, -1)

        # Prepare 3-channel background (RGB Color or 3-channel grayscale)
        if p.get('monochrome_mode') == 'RGB (Color)' and len(frame.shape) == 3:
            frame_out = frame.copy()
        else:
            frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        enable_contour_detection = p.get('enable_contour_detection', True)
        if not enable_contour_detection:
            if p.get('show_blobs', True):
                frame_out[mask > 0] = (0, 0, 255)
            if p.get('enable_proximity', False):
                frame_out = _draw_proximity_density(frame_out, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, crop_mask, [], mask, p)
            if p.get('show_distance_circles', False) or p.get('show_crop_outline', True):
                _draw_distance_circles(frame_out, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, float(p.get('spatial_scale', 1.0)), str(p.get('spatial_unit', 'px')), int(p.get('pellet_radius', 0)), p)
            return frame_out, {}, [], mask, []

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        areas = [cv2.contourArea(c) for c in cnts]

        lower_thresh = float(p.get('lower_surface_thresh', 25))
        upper_thresh = float(p.get('upper_surface_thresh', 1000))
        thresh_mode = p.get('area_thresh_mode', 'Absolute (px^2)')

        filtered_cnts = []
        if areas:
            if thresh_mode == 'Relative (%)':
                lo_pct = np.clip(lower_thresh, 0.0, 100.0)
                hi_pct = np.clip(100.0 - upper_thresh, 0.0, 100.0)
                lo_bound = np.percentile(areas, lo_pct)
                hi_bound = np.percentile(areas, hi_pct)
                filtered_cnts = [c for c, a in zip(cnts, areas) if lo_bound <= a <= hi_bound]
            else:  # Absolute
                filtered_cnts = [c for c, a in zip(cnts, areas) if lower_thresh <= a <= upper_thresh]

        filtered_mask = np.zeros_like(mask)
        if filtered_cnts:
            cv2.drawContours(filtered_mask, filtered_cnts, -1, 255, -1)

        show_blobs = p.get('show_blobs', True)
        show_boxes = p.get('show_boxes', True)
        show_centroids = p.get('show_centroids', True)
        isolate_selection = p.get('isolate_selection', False)
        visu_format = p.get('visu_format', 'Frame')

        if visu_format == 'Mask':
            frame_out_contours = np.zeros_like(frame, dtype=np.uint8)
            if circular_crop:
                frame_out_contours[crop_mask == 0] = 0
        else:
            frame_out_contours = frame_out

        if show_blobs:
            frame_out_contours[filtered_mask > 0] = (0, 0, 255)

        centroids = []
        selection_mask = np.zeros_like(mask)

        for c in filtered_cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx_obj = int(M["m10"] / M["m00"])
                cy_obj = int(M["m01"] / M["m00"])
            else:
                rect = cv2.minAreaRect(c)
                cx_obj, cy_obj = int(rect[0][0]), int(rect[0][1])
            
            centroids.append((cx_obj, cy_obj))

            if isolate_selection:
                cv2.drawContours(selection_mask, [c], -1, 255, -1)

            if show_boxes:
                rect = cv2.minAreaRect(c)
                box_pts = np.array(cv2.boxPoints(rect), dtype='int')
                cv2.drawContours(frame_out_contours, [box_pts], -1, (0, 255, 0), 1, cv2.LINE_AA)

            if show_centroids:
                cv2.circle(frame_out_contours, (cx_obj, cy_obj), 2, (0, 0, 255), -1, cv2.LINE_AA)

        if isolate_selection:
            if visu_format == 'Mask':
                frame_out_contours[selection_mask == 255] = frame[selection_mask == 255]
            else:
                frame_out_contours[selection_mask != 255] = 0

        # Tracking
        enable_tracking = p.get('enable_tracking', True)
        distances = []
        pellet_radius = float(p.get('pellet_radius', 0))

        if enable_tracking:
            norfair_detections = [Detection(np.array(pt)) for pt in centroids]
            tracked_objects = self.tracker.update(norfair_detections)
        else:
            tracked_objects = []
            self.tracking.clear()
            for idx, pt in enumerate(centroids):
                dist_center = float(np.sqrt((pt[0] - pellet_cx) ** 2 + (pt[1] - pellet_cy) ** 2))
                dist = max(0.0, dist_center - pellet_radius)
                distances.append((idx, dist))

        trail_length = int(p.get("trail_length", 30))
        trail_thickness = max(1, int(p.get("trail_thickness", 2)))
        show_trail = p.get("show_trail", False)
        show_id = p.get("show_id", False)
        show_age = p.get("show_age", False)
        show_distance = p.get("show_distance", False)
        show_speed = p.get("show_speed", False)
        show_proximity = p.get("show_proximity", True)

        for obj in tracked_objects:
            cx_obj, cy_obj = map(int, obj.estimate[0])
            dist_center = float(np.sqrt((cx_obj - pellet_cx) ** 2 + (cy_obj - pellet_cy) ** 2))
            dist = max(0.0, dist_center - pellet_radius)
            distances.append((obj.id, dist))

            if obj.id not in self.tracking:
                self.tracking[obj.id] = {
                    "color":    self._get_random_color(),
                    "points":   [],
                    "distance": 0.0,
                    "age":      obj.age
                }

            self.tracking[obj.id]["points"].append((cx_obj, cy_obj))
            self.tracking[obj.id]["distance"] += getattr(obj, "last_distance", 0.0)
            self.tracking[obj.id]["age"] = obj.age

            if show_trail:
                trail = self.tracking[obj.id]["points"][-trail_length:]
                if len(trail) >= 2:
                    for i in range(1, len(trail)):
                        cv2.line(frame_out_contours, trail[i - 1], trail[i],
                                 self.tracking[obj.id]["color"], trail_thickness, cv2.LINE_AA)

            text_lines = []
            if show_id:        text_lines.append(f"ID:{obj.id}")
            if show_age:       text_lines.append(f"Age:{obj.age}")
            if show_distance:  text_lines.append(f"Dist:{self.tracking[obj.id]['distance']:.1f}")
            if show_speed:
                pts = self.tracking[obj.id]["points"][-trail_length:]
                if len(pts) >= 2:
                    path_len = sum(
                        float(np.sqrt((pts[k][0] - pts[k-1][0])**2 + (pts[k][1] - pts[k-1][1])**2))
                        for k in range(1, len(pts))
                    )
                    calib = float(p.get('spatial_scale', p.get('df_calibration', 1.0)))
                    fps = float(p.get('video_fps', p.get('df_fps', 30.0)))
                    speed = (path_len * calib * fps) / (len(pts) - 1)
                    text_lines.append(f"Speed:{speed:.2f}")
            if show_proximity: text_lines.append(f"Prox:{dist:.1f}")

            for i, line in enumerate(text_lines):
                cv2.putText(
                    frame_out_contours, line, (cx_obj + 4, cy_obj + 12 + i * 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
                )

        if p.get('enable_proximity', False):
            frame_out_contours = _draw_proximity_density(frame_out_contours, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, crop_mask, centroids, mask, p)

        if p.get('show_distance_circles', False) or p.get('show_crop_outline', True):
            _draw_distance_circles(frame_out_contours, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, float(p.get('spatial_scale', 1.0)), str(p.get('spatial_unit', 'px')), int(p.get('pellet_radius', 0)), p)

        return frame_out_contours, self.tracking, filtered_cnts, filtered_mask, distances

    def _process_denseflow_preprocessed(self,
                                        frame: np.ndarray,
                                        gray: np.ndarray,
                                        crop_mask: np.ndarray,
                                        circular_crop: bool,
                                        crop_radius: int,
                                        pellet_center: Tuple[int, int],
                                        crop_center: Tuple[int, int],
                                        p: Dict[str, Any]) -> Tuple[np.ndarray, dict, list, Optional[np.ndarray], list]:
        """
        Optical flow calculated and overlayed on top of the preprocessed monochrome frame.
        """
        pellet_cx, pellet_cy = pellet_center
        crop_cx, crop_cy = crop_center
        if self._df_prev_gray is None:
            self._df_prev_gray = gray
            frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if p.get('show_distance_circles', False) or p.get('show_crop_outline', True):
                _draw_distance_circles(frame_out, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, float(p.get('spatial_scale', 1.0)), str(p.get('spatial_unit', 'px')), int(p.get('pellet_radius', 0)), p)
            return frame_out, {}, [], None, []

        fb_params = {
            'pyr_scale':  p.get('df_pyr_scale',  0.5),
            'levels':     p.get('df_levels',      2),
            'winsize':    p.get('df_winsize',      9),
            'iterations': p.get('df_iterations',  2),
            'poly_n':     p.get('df_poly_n',       5),
            'poly_sigma': p.get('df_poly_sigma',   1.1),
        }
        mag, ang, self._df_last_umat_flow = _denseflow_compute(
            self._df_prev_gray, gray,
            self._df_last_umat_flow,
            fb_params,
            self._opencl_available
        )
        self._df_prev_gray = gray

        mag[crop_mask == 0] = 0.0

        flow_vis, metric_val, raw_speeds = _compute_denseflow_metrics_and_vis(mag, ang, p, self._opencl_available, crop_mask)

        mono_bgr = frame.copy() if (p.get('monochrome_mode') == 'RGB (Color)' and len(frame.shape) == 3) else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        frame_out = mono_bgr.copy()
        active_flow = np.any(flow_vis > 0, axis=2)
        if np.any(active_flow):
            frame_out[active_flow] = flow_vis[active_flow]

        if p.get('show_distance_circles', False) or p.get('show_crop_outline', True):
            _draw_distance_circles(frame_out, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, float(p.get('spatial_scale', 1.0)), str(p.get('spatial_unit', 'px')), int(p.get('pellet_radius', 0)), p)

        distances = [(-1, metric_val)]
        return frame_out, {}, [], flow_vis, distances

    def _process_both_preprocessed(self,
                                   frame: np.ndarray,
                                   gray: np.ndarray,
                                   crop_mask: np.ndarray,
                                   circular_crop: bool,
                                   crop_radius: int,
                                   pellet_center: Tuple[int, int],
                                   crop_center: Tuple[int, int],
                                   p: Dict[str, Any]) -> Tuple[np.ndarray, dict, list, Optional[np.ndarray], list]:
        """
        Combines optical flow speed and contours from the preprocessed grayscale image.
        """
        pellet_cx, pellet_cy = pellet_center
        crop_cx, crop_cy = crop_center
        if self._df_prev_gray is None:
            self._df_prev_gray = gray
            frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if p.get('show_distance_circles', False) or p.get('show_crop_outline', True):
                _draw_distance_circles(frame_out, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, float(p.get('spatial_scale', 1.0)), str(p.get('spatial_unit', 'px')), int(p.get('pellet_radius', 0)), p)
            return frame_out, {}, [], None, []

        fb_params = {
            'pyr_scale':  p.get('df_pyr_scale',  0.5),
            'levels':     p.get('df_levels',      2),
            'winsize':    p.get('df_winsize',      9),
            'iterations': p.get('df_iterations',  2),
            'poly_n':     p.get('df_poly_n',       5),
            'poly_sigma': p.get('df_poly_sigma',   1.1),
        }
        mag, ang, self._df_last_umat_flow = _denseflow_compute(
            self._df_prev_gray, gray,
            self._df_last_umat_flow,
            fb_params,
            self._opencl_available
        )
        self._df_prev_gray = gray

        mag[crop_mask == 0] = 0.0

        frame_vis, tracking, filtered_cnts, mask, distances = self._process_contour_preprocessed(
            frame, gray, crop_mask, circular_crop, crop_radius, pellet_center, crop_center, p)

        vis_mode = p.get('both_vis_mode', 'Surface')
        if vis_mode == 'Speed':
            flow_vis = _denseflow_visualize(mag, ang, crop_mask, p.get('df_supersat', True), self._opencl_available)
            active_f = np.any(flow_vis > 0, axis=2)
            if np.any(active_f):
                frame_vis[active_f] = flow_vis[active_f]
        elif vis_mode == 'Selection':
            flow_vis = _denseflow_visualize(mag, ang, mask, p.get('df_supersat', True), self._opencl_available)
            active_f = np.any(flow_vis > 0, axis=2)
            if np.any(active_f):
                frame_vis[active_f] = flow_vis[active_f]

        if p.get('show_distance_circles', False) or p.get('show_crop_outline', True) or int(p.get('pellet_radius', 0)) > 0:
            _draw_distance_circles(frame_vis, int(p.get('num_distance_circles', 5)), circular_crop, crop_radius, pellet_center, crop_center, float(p.get('spatial_scale', 1.0)), str(p.get('spatial_unit', 'px')), int(p.get('pellet_radius', 0)), p)

        return frame_vis, tracking, filtered_cnts, mask, distances

    # ── Stateless frame processor ─────────────────────────────────────

    @staticmethod
    def process_frame_stateless(frame: np.ndarray, params: dict) -> List[Tuple[int, int]]:
        """
        Binarizes a frame and extracts its contour centroids, stateless.
        Applies Monochrome, Gamma and Circular Crop centering offset settings if enabled.
        """
        if not isinstance(frame, np.ndarray):
            return []

        H, W = frame.shape[:2]
        crop_cx, crop_cy = _get_crop_center(W, H, params)
        pellet_cx, pellet_cy = _get_pellet_center(W, H, params)

        enable_preprocessing = params.get('enable_preprocessing', True)
        active_processing = params.get('active', True)

        if not active_processing:
            return []

        # 1. Preprocess
        if enable_preprocessing:
            gray = _apply_monochrome_mode(frame, params.get('monochrome_mode', 'Grayscale'))
            
            if params.get('invert_image', False):
                gray = cv2.bitwise_not(gray)

            gamma = float(params.get('gamma', 1.0))
            if abs(gamma - 1.0) > 1e-3:
                gray = _adjust_gamma(gray, gamma)
                
            gray, crop_mask, _, _ = _apply_crop_mode(gray, params, crop_cx, crop_cy, (pellet_cx, pellet_cy))

            # Static removal filter in stateless batch mode
            r1 = params.get("_ref_frame_1", None)
            r2 = params.get("_ref_frame_2", None)
            if params.get('remove_static', False) and r1 is not None and r2 is not None:
                if r1.shape == gray.shape and r2.shape == gray.shape:
                    similarity_threshold = int(params.get('similarity_threshold', 15))
                    diff_ref = cv2.absdiff(r1, r2)
                    diff_cur = cv2.absdiff(gray, r1)
                    _, mask_ref = cv2.threshold(diff_ref, similarity_threshold, 255, cv2.THRESH_BINARY_INV)
                    _, mask_cur = cv2.threshold(diff_cur, similarity_threshold, 255, cv2.THRESH_BINARY_INV)
                    static_mask = cv2.bitwise_and(mask_ref, mask_cur)
                    gray[static_mask > 0] = 0
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            crop_mask = np.full((H, W), 255, dtype=np.uint8)

        # 2. Binarize
        blur_k = int(params.get('blur', 3))
        if blur_k > 0:
            blur_k = blur_k | 1
            gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        else:
            gray_blurred = gray

        block_size = int(params.get('block_size', 25))
        if block_size < 3: block_size = 3
        block_size = block_size | 1

        bin_thresh = int(params.get('bin_thresh', 15))

        mask = cv2.adaptiveThreshold(
            gray_blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, bin_thresh
        )

        mask = cv2.bitwise_and(mask, crop_mask)

        morph_close = int(params.get('morph_close', 0))
        if morph_close > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close)

        if params.get('fill_contours', False):
            cnts_fill = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cv2.drawContours(mask, cnts_fill, -1, 255, -1)

        # 3. Find Contours
        if not params.get('enable_contour_detection', True):
            return []

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        areas = [cv2.contourArea(c) for c in cnts]

        lower_thresh = float(params.get('lower_surface_thresh', 25))
        upper_thresh = float(params.get('upper_surface_thresh', 1000))
        thresh_mode = params.get('area_thresh_mode', 'Absolute (px^2)')

        filtered_cnts = []
        if areas:
            if thresh_mode == 'Relative (%)':
                lo_pct = np.clip(lower_thresh, 0.0, 100.0)
                hi_pct = np.clip(100.0 - upper_thresh, 0.0, 100.0)
                lo_bound = np.percentile(areas, lo_pct)
                hi_bound = np.percentile(areas, hi_pct)
                filtered_cnts = [c for c, a in zip(cnts, areas) if lo_bound <= a <= hi_bound]
            else:  # Absolute
                filtered_cnts = [c for c, a in zip(cnts, areas) if lower_thresh <= a <= upper_thresh]

        # Extract centroids
        centroids = []
        for c in filtered_cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx_obj = int(M["m10"] / M["m00"])
                cy_obj = int(M["m01"] / M["m00"])
            else:
                rect = cv2.minAreaRect(c)
                cx_obj, cy_obj = int(rect[0][0]), int(rect[0][1])
            centroids.append((cx_obj, cy_obj))

        return centroids
