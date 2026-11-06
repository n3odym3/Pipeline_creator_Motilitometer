"""
Dense Optical Flow Engine for Motilitometer Counter Tracker.

Encapsulates:
- GPU OpenCL / CPU Farneback dense optical flow
- Multi-mode flow visualizations:
  * HSV Flow (Hue=Direction, Sat=255, Val=Magnitude or Supersaturated)
  * Speed Heatmap (Turbo colormap calibrated to µm/s)
  * Binary Mask
  * Arrows Only (Vector flow field)
- Velocity magnitude thresholding, morphological erosion, and contour isolation
- Drift correction integration (pixel-level and contour-level vector subtraction & motility %)
- Directional vector collection for polar plots
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .preprocessor import PreProcessor
from .contour_processor import ContourProcessor
from .drift_processor import DriftProcessor


class DenseFlowProcessor:
    """Manages dense optical flow computation, filtering, drift correction, and visualizations."""

    def __init__(self) -> None:
        self.opencl_available = self.check_opencl()
        self.drift_processor = DriftProcessor()
        if not self.opencl_available:
            warnings.warn(
                "OpenCL not available – Dense Flow will run in CPU mode.",
                RuntimeWarning,
            )

    def reset_drift(self) -> None:
        """Resets internal drift processor state."""
        self.drift_processor.reset()

    @staticmethod
    def check_opencl() -> bool:
        """Checks if OpenCL acceleration is functional via cv2.UMat."""
        u = None
        try:
            if not cv2.ocl.haveOpenCL():
                return False
            cv2.ocl.setUseOpenCL(True)
            u = cv2.UMat(np.zeros((4, 4), dtype=np.uint8))
            return bool(cv2.ocl.useOpenCL())
        except Exception:
            return False
        finally:
            del u

    @classmethod
    def compute_flow(cls,
                     prev_gray: np.ndarray,
                     gray: np.ndarray,
                     last_umat_flow: Any,
                     fb_params: dict,
                     use_opencl: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Computes Farneback dense optical flow between two grayscale frames.
        Returns:
            Tuple of (mag, ang, flow_x, flow_y, last_umat_flow)
        """
        if prev_gray.shape != gray.shape:
            prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))

        # Ensure parameters are properly typed and valid
        clean_params = dict(fb_params)
        winsize = max(3, int(clean_params.get('winsize', 15)) | 1)
        clean_params['winsize'] = winsize
        clean_params['levels'] = max(1, int(clean_params.get('levels', 2)))
        clean_params['iterations'] = max(1, int(clean_params.get('iterations', 3)))
        clean_params['poly_n'] = 7 if int(clean_params.get('poly_n', 5)) == 7 else 5
        clean_params['poly_sigma'] = float(clean_params.get('poly_sigma', 1.2))
        clean_params['pyr_scale'] = float(clean_params.get('pyr_scale', 0.5))

        blur_k = 3 if winsize <= 9 else 5
        prev_smooth = cv2.GaussianBlur(prev_gray, (blur_k, blur_k), 0)
        gray_smooth = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        if use_opencl:
            prev_u = None
            cur_u = None
            flow_u = None
            try:
                prev_u = cv2.UMat(prev_smooth)
                cur_u = cv2.UMat(gray_smooth)
                flow_u = cv2.calcOpticalFlowFarneback(prev_u, cur_u, None, flags=0, **clean_params)
                flow = flow_u.get()
            except Exception:
                flow = cv2.calcOpticalFlowFarneback(prev_smooth, gray_smooth, None, flags=0, **clean_params)
            finally:
                del flow_u, prev_u, cur_u
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_smooth, gray_smooth, None, flags=0, **clean_params)

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        mag, ang = cv2.cartToPolar(flow_x, flow_y)

        return mag, ang, flow_x, flow_y, None

    @classmethod
    def visualize_flow(cls,
                       mag: np.ndarray,
                       ang: np.ndarray,
                       flow_x: Optional[np.ndarray],
                       flow_y: Optional[np.ndarray],
                       binary_mask: np.ndarray,
                       original_bgr: Optional[np.ndarray],
                       p: dict,
                       use_opencl: bool,
                       raw_flow_x: Optional[np.ndarray] = None,
                       raw_flow_y: Optional[np.ndarray] = None,
                       drift_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Renders the optical flow field layer according to parameters."""
        visu_layer = p.get('df_visu_mode', p.get('visu_mode', 'HSV Flow'))
        overlay = p.get('df_overlay_enabled', p.get('overlay_enabled', p.get('overlap', False)))
        opacity = float(p.get('df_overlay_alpha', p.get('overlay_alpha', p.get('overlay_opacity', 0.6))))
        supersat = p.get('df_supersat', True)
        show_arrows = p.get('df_show_arrows', True)
        arrow_step = int(p.get('df_arrow_step', 16))
        arrow_scale = float(p.get('df_arrow_scale', 2.0))
        use_contours = p.get('df_use_contours', False)
        enable_drift = bool(p.get('enable_drift_correction', False))
        show_correction = enable_drift and bool(p.get('drift_show_correction', False))
        H, W = mag.shape[:2]

        lower_mag = float(p.get('df_lower_mag_thresh', p.get('lower_mag_thresh', 1.5)))
        upper_mag = float(p.get('df_upper_mag_thresh', p.get('upper_mag_thresh', 0.0)))
        active_mask = (binary_mask > 0) & (mag > 0 if enable_drift else (mag >= lower_mag))
        if not enable_drift and upper_mag > 0.0:
            active_mask &= (mag <= upper_mag)

        if visu_layer == "Binary Mask":
            layer = np.zeros((H, W, 3), dtype=np.uint8)
            layer[binary_mask > 0] = (255, 255, 255)
            active_mask = binary_mask > 0
        elif visu_layer == "Speed Heatmap":
            calib = float(p.get("df_calibration", 1.0))
            fps = float(p.get("df_fps", 30.0))
            scale = calib * fps
            speed_um_s = mag * scale

            heatmap_auto = p.get("df_heatmap_auto_range", p.get("heatmap_auto_range", True))
            if heatmap_auto:
                valid = speed_um_s[active_mask]
                if len(valid) > 0:
                    s_min, s_max = float(valid.min()), float(valid.max())
                else:
                    s_min, s_max = 0.0, 1.0
            else:
                s_min = float(p.get("df_heatmap_vmin", p.get("heatmap_vmin", 0.0)))
                s_max = float(p.get("df_heatmap_vmax", p.get("heatmap_vmax", 100.0)))

            if s_max > s_min:
                m_norm = np.clip((speed_um_s - s_min) / (s_max - s_min) * 255.0, 0, 255).astype(np.uint8)
            else:
                m_norm = np.zeros((H, W), dtype=np.uint8)

            heatmap = cv2.applyColorMap(m_norm, cv2.COLORMAP_TURBO)
            layer = np.zeros((H, W, 3), dtype=np.uint8)
            layer[active_mask] = heatmap[active_mask]
        elif visu_layer == "Arrows Only":
            layer = np.zeros((H, W, 3), dtype=np.uint8)
        else:  # "HSV Flow"
            hue = (ang * (90.0 / np.pi)).astype(np.uint8)
            sat = np.full((H, W), 255, np.uint8)
            if supersat:
                val = np.full((H, W), 255, np.uint8)
            else:
                m_min, m_max = float(mag.min()), float(mag.max())
                val = ((mag - m_min) * (255.0 / (m_max - m_min))).astype(np.uint8) if m_max > m_min else np.zeros((H, W), np.uint8)

            hsv = cv2.merge((hue, sat, val))
            if use_opencl:
                hsv_u = None
                bgr_u = None
                try:
                    hsv_u = cv2.UMat(hsv)
                    bgr_u = cv2.cvtColor(hsv_u, cv2.COLOR_HSV2BGR)
                    bgr = bgr_u.get()
                except Exception:
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                finally:
                    del bgr_u, hsv_u
            else:
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            layer = np.zeros((H, W, 3), dtype=np.uint8)
            layer[active_mask] = bgr[active_mask]

        if overlay and original_bgr is not None:
            if original_bgr.shape[:2] != (H, W):
                original_bgr = cv2.resize(original_bgr, (W, H))
            target_canvas = original_bgr.copy()
            if len(target_canvas.shape) == 2:
                target_canvas = cv2.cvtColor(target_canvas, cv2.COLOR_GRAY2BGR)
            if visu_layer != "Arrows Only" and np.any(active_mask):
                blended = cv2.addWeighted(layer, opacity, target_canvas, 1.0 - opacity, 0.0)
                target_canvas[active_mask] = blended[active_mask]
        else:
            target_canvas = layer

        # Draw grid arrows when not in contour-tracking mode
        if (show_arrows or show_correction) and flow_x is not None and flow_y is not None and not use_contours:
            step = max(8, arrow_step)
            drawn_arrows = 0
            max_arrows = 4000  # Cap drawing to maintain high framerate
            for y in range(step // 2, H, step):
                for x in range(step // 2, W, step):
                    if binary_mask[y, x] == 0:
                        continue
                    m = mag[y, x]
                    if m <= 0.0 or (not enable_drift and (m < lower_mag or (upper_mag > 0.0 and m > upper_mag))):
                        continue
                    pt1 = (x, y)
                    is_drift_px = drift_mask is not None and drift_mask[y, x] > 0
                    if is_drift_px:
                        fx, fy = flow_x[y, x], flow_y[y, x]
                        pt2 = (int(round(x + fx * arrow_scale)), int(round(y + fy * arrow_scale)))
                        cv2.arrowedLine(target_canvas, pt1, pt2, (0, 140, 255), 1, tipLength=0.3)
                    else:
                        if show_correction and raw_flow_x is not None and raw_flow_y is not None:
                            rfx, rfy = raw_flow_x[y, x], raw_flow_y[y, x]
                            rpt2 = (int(round(x + rfx * arrow_scale)), int(round(y + rfy * arrow_scale)))
                            cv2.arrowedLine(target_canvas, pt1, rpt2, (0, 200, 255), 1, tipLength=0.3)
                        fx, fy = flow_x[y, x], flow_y[y, x]
                        pt2 = (int(round(x + fx * arrow_scale)), int(round(y + fy * arrow_scale)))
                        cv2.arrowedLine(target_canvas, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
                    drawn_arrows += 1
                    if drawn_arrows >= max_arrows:
                        break
                if drawn_arrows >= max_arrows:
                    break

        return target_canvas

    @classmethod
    def compute_metrics_and_vis(cls,
                                mag: np.ndarray,
                                ang: np.ndarray,
                                flow_x: Optional[np.ndarray],
                                flow_y: Optional[np.ndarray],
                                original_bgr: Optional[np.ndarray],
                                p: dict,
                                use_opencl: bool,
                                focus_map: Optional[np.ndarray] = None,
                                drift_processor: Optional[DriftProcessor] = None,
                                frame_idx: Optional[int] = None,
                                render_vis: bool = True) -> Tuple[Optional[np.ndarray], float, List[float], list, list, np.ndarray, dict]:
        """Filters flow, applies optional drift correction, computes metrics, and produces annotated visualization."""
        calib = float(p.get('df_calibration', 1.0))
        fps = float(p.get('df_fps', 30.0))
        scale = calib * fps

        lower_thresh = float(p.get('df_lower_mag_thresh', p.get('lower_mag_thresh', 1.5)))
        upper_thresh = float(p.get('df_upper_mag_thresh', p.get('upper_mag_thresh', 0.0)))
        erosion_k = int(p.get('df_erosion', p.get('erosion', 3)))
        use_contours = p.get('df_use_contours', False) or p.get('enable_tracking', False)
        enable_tracking = bool(p.get('enable_tracking', False))
        isolate = p.get('df_isolate_selection', p.get('isolate_selection', False))
        lo_thresh = float(p.get('df_lower_surface_thresh', p.get('lower_surface_thresh', 25)))
        hi_thresh = float(p.get('df_upper_surface_thresh', p.get('upper_surface_thresh', 100)))
        show_boxes = p.get('df_show_boxes', p.get('show_boxes', False))
        show_centroids = p.get('df_show_centroids', p.get('show_centroids', False))
        show_arrows = p.get('df_show_arrows', p.get('show_arrows', True))

        enable_drift = bool(p.get('enable_drift_correction', False))
        drift_show_filtered = bool(p.get('drift_show_filtered', False))
        drift_show_correction = bool(p.get('drift_show_correction', False))
        dp = drift_processor if drift_processor is not None else DriftProcessor()

        H, W = mag.shape[:2]
        if upper_thresh > 0.0:
            binary = ((mag >= lower_thresh) & (mag <= upper_thresh)).astype(np.uint8) * 255
        else:
            binary = (mag >= lower_thresh).astype(np.uint8) * 255

        # True morphological erosion to visibly shrink blobs and strip boundary noise
        if erosion_k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.erode(binary, kernel, iterations=erosion_k)

        mag_filt = mag.copy()
        mag_filt[binary == 0] = 0.0

        filter_stage = p.get("focus_filter_stage", "Pre-processing")
        is_post_focus = (
            p.get("enable_preprocessing", True)
            and p.get("enable_focus_picking", False)
            and p.get("focus_use_for_processing", False)
            and filter_stage == "Post-processing"
            and focus_map is not None
        )

        if not use_contours:
            if is_post_focus:
                thresh_pct = float(p.get("focus_threshold", 40))
                thresh = thresh_pct * 255.0 / 100.0
                mag_filt[focus_map < thresh] = 0.0
                binary[focus_map < thresh] = 0

            ys, xs = np.nonzero(mag_filt > 0)
            total_moving_px = int(len(xs))
            drift_info = None
            motility_pct = None
            mobile_px_count = None
            total_px_count = None

            if render_vis:
                mag_vis = mag_filt.copy()
                ang_vis = ang.copy()
                fx_vis = flow_x.copy() if flow_x is not None else None
                fy_vis = flow_y.copy() if flow_y is not None else None
                binary_vis = binary.copy()
            else:
                mag_vis = ang_vis = fx_vis = fy_vis = None
                binary_vis = binary
            drift_pixel_mask = None

            if enable_drift and flow_x is not None and flow_y is not None:
                drift_res = dp.process_vectors(
                    xs, ys, flow_x[ys, xs], flow_y[ys, xs], (H, W), p, frame_idx=frame_idx
                )
                drift_info = drift_res["drift_info"]
                is_drift = drift_res["is_drift"]
                motile_mask = ~is_drift

                valid_mags = drift_res["corr_mag"][motile_mask]
                valid_angs = drift_res["corr_ang"][motile_mask]

                mobile_px_count = int(np.count_nonzero(motile_mask))
                total_px_count = total_moving_px
                motility_pct = round(100.0 * mobile_px_count / total_px_count, 2) if total_px_count > 0 else 0.0

                if render_vis:
                    # Update visual flow matrices with corrected vectors
                    m_ys, m_xs = ys[motile_mask], xs[motile_mask]
                    mag_vis[m_ys, m_xs] = drift_res["corr_mag"][motile_mask]
                    ang_vis[m_ys, m_xs] = drift_res["corr_ang"][motile_mask]
                    fx_vis[m_ys, m_xs] = drift_res["corr_vx"][motile_mask]
                    fy_vis[m_ys, m_xs] = drift_res["corr_vy"][motile_mask]

                    d_ys, d_xs = ys[is_drift], xs[is_drift]
                    if not drift_show_filtered:
                        binary_vis[d_ys, d_xs] = 0
                        mag_vis[d_ys, d_xs] = 0.0
                    else:
                        drift_pixel_mask = np.zeros((H, W), dtype=np.uint8)
                        drift_pixel_mask[d_ys, d_xs] = 255
            else:
                valid_mags = mag_filt[ys, xs]
                valid_angs = ang[ys, xs]

            mean_speed_px = float(np.mean(valid_mags)) if len(valid_mags) > 0 else 0.0
            metric_val = round(mean_speed_px * scale, 3)

            # Cap vector serialization list conversions to prevent massive framerate drops
            total_count = int(len(valid_angs))
            max_plot_vectors = 5000
            if total_count > max_plot_vectors:
                stride = total_count // max_plot_vectors
                sub_angs = valid_angs[::stride]
                sub_mags = valid_mags[::stride]
            else:
                sub_angs = valid_angs
                sub_mags = valid_mags

            if p.get('collect_raw_speeds', False):
                if total_count > 20000:
                    raw_stride = total_count // 20000
                    raw_speeds_um_s = (valid_mags[::raw_stride] * scale).tolist()
                else:
                    raw_speeds_um_s = (valid_mags * scale).tolist()
            else:
                raw_speeds_um_s = []

            if render_vis:
                ang_out = sub_angs.tolist() if isinstance(sub_angs, np.ndarray) else list(sub_angs)
                mag_out = (sub_mags * scale).tolist() if isinstance(sub_mags, np.ndarray) else list(sub_mags * scale)
            else:
                ang_out = np.asarray(sub_angs, dtype=np.float32)
                mag_out = np.asarray(sub_mags * scale, dtype=np.float32)

            vectors_data = {
                "angles": ang_out,
                "magnitudes": mag_out,
                "mode": "denseflow",
                "count": total_count,
                "motility_pct": motility_pct,
                "mobile_count": mobile_px_count,
                "total_count": total_px_count,
                "drift_info": drift_info,
            }
            if render_vis:
                flow_vis = cls.visualize_flow(
                    mag_vis, ang_vis, fx_vis, fy_vis, binary_vis, original_bgr, p, use_opencl,
                    raw_flow_x=flow_x, raw_flow_y=flow_y, drift_mask=drift_pixel_mask
                )
                if isolate:
                    flow_vis[binary_vis == 0] = 0
                if enable_drift:
                    DriftProcessor.draw_drift_overlay(
                        flow_vis, drift_info, arrow_scale=float(p.get('df_arrow_scale', 2.0)), params=p
                    )
            else:
                flow_vis = None
            return flow_vis, metric_val, raw_speeds_um_s, [], [], binary_vis, vectors_data

        cut_mask = None
        if p.get("enable_watershed", False):
            binary, cut_mask = ContourProcessor.apply_watershed(binary, p)
            ContourProcessor._last_cut_mask = cut_mask

        cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        filtered_cnts = []
        filtered_bboxes = []
        detections_list = []

        if cnts:
            # Fast pre-filtering to drop 1-2 pixel noise specks before expensive minAreaRect
            valid_cnts = [c for c in cnts if c.shape[0] >= 3 and cv2.contourArea(c) >= 3]
            if len(valid_cnts) > 1500:
                valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:1500]

            bboxes = [cv2.minAreaRect(c) for c in valid_cnts]
            areas = [(b[1][0] * b[1][1]) for b in bboxes]
            if areas:
                lo = np.percentile(areas, lo_thresh)
                hi = np.percentile(areas, hi_thresh)
                for c, b, a in zip(valid_cnts, bboxes, areas):
                    if lo <= a <= hi:
                        filtered_cnts.append(c)
                        filtered_bboxes.append(b)
                        center = (int(b[0][0]), int(b[0][1]))
                        detections_list.append(center)

        if is_post_focus and filtered_cnts:
            filtered_bboxes, filtered_cnts, detections_list = PreProcessor.filter_contours_by_focus(
                focus_map, filtered_bboxes, filtered_cnts, detections_list, p
            )

        # Compute raw per-contour velocity vectors
        cnt_records: List[Dict[str, Any]] = []
        c_xs: List[float] = []
        c_ys: List[float] = []
        c_fxs: List[float] = []
        c_fys: List[float] = []
        valid_c_indices: List[int] = []

        for idx, (c, bb, center) in enumerate(zip(filtered_cnts, filtered_bboxes, detections_list)):
            c_mask = np.zeros_like(binary)
            cv2.drawContours(c_mask, [c], -1, 255, thickness=-1)
            active_c = (c_mask > 0) & (mag_filt > 0)
            if np.any(active_c):
                if flow_x is not None and flow_y is not None:
                    mean_fx = float(np.mean(flow_x[active_c]))
                    mean_fy = float(np.mean(flow_y[active_c]))
                    mean_mag = float(np.mean(mag_filt[active_c]))
                    mean_ang = float(np.arctan2(mean_fy, mean_fx) % (2.0 * np.pi))
                    # Keep vector components consistent with mean_mag & mean_ang
                    mean_fx = float(mean_mag * np.cos(mean_ang))
                    mean_fy = float(mean_mag * np.sin(mean_ang))
                else:
                    mean_mag = float(np.mean(mag_filt[active_c]))
                    pts = c[:, 0, :]
                    ax = np.clip(pts[:, 0], 0, W - 1)
                    ay = np.clip(pts[:, 1], 0, H - 1)
                    angs_c = ang[ay, ax]
                    mean_ang = float(np.arctan2(np.mean(np.sin(angs_c)), np.mean(np.cos(angs_c))) % (2.0 * np.pi))
                    mean_fx = float(mean_mag * np.cos(mean_ang))
                    mean_fy = float(mean_mag * np.sin(mean_ang))
            else:
                mean_fx, mean_fy, mean_mag, mean_ang = 0.0, 0.0, 0.0, 0.0

            circle = cv2.minEnclosingCircle(c)
            cx, cy = int(circle[0][0]), int(circle[0][1])
            if mean_mag > 0:
                valid_c_indices.append(idx)
                c_xs.append(float(cx))
                c_ys.append(float(cy))
                c_fxs.append(mean_fx)
                c_fys.append(mean_fy)

            cnt_records.append({
                "c": c,
                "bb": bb,
                "center": center,
                "cx": cx,
                "cy": cy,
                "c_mask": c_mask,
                "raw_fx": mean_fx,
                "raw_fy": mean_fy,
                "raw_mag": mean_mag,
                "raw_ang": mean_ang,
                "corr_fx": mean_fx,
                "corr_fy": mean_fy,
                "corr_mag": mean_mag,
                "corr_ang": mean_ang,
                "is_drift": False,
            })

        drift_info = None
        motility_pct = None
        mobile_cnt = None
        total_cnt = None

        if enable_drift:
            drift_res = dp.process_vectors(c_xs, c_ys, c_fxs, c_fys, (H, W), p, frame_idx=frame_idx)
            drift_info = drift_res["drift_info"]
            for k, rec_idx in enumerate(valid_c_indices):
                rec = cnt_records[rec_idx]
                rec["is_drift"] = bool(drift_res["is_drift"][k])
                rec["corr_fx"] = float(drift_res["corr_vx"][k])
                rec["corr_fy"] = float(drift_res["corr_vy"][k])
                rec["corr_mag"] = float(drift_res["corr_mag"][k])
                rec["corr_ang"] = float(drift_res["corr_ang"][k])

        if render_vis:
            selection_mask = np.zeros_like(binary)
            for rec in cnt_records:
                if enable_drift and rec["is_drift"] and not drift_show_filtered:
                    continue
                cv2.drawContours(selection_mask, [rec["c"]], -1, 255, thickness=-1)

            if isolate:
                effective_mask = selection_mask.copy()
                mag_filt[effective_mask == 0] = 0.0
            else:
                effective_mask = selection_mask.copy() if (enable_drift and not drift_show_filtered) else binary.copy()

            flow_vis = cls.visualize_flow(mag_filt, ang, flow_x, flow_y, effective_mask, original_bgr, p, use_opencl)

            if isolate:
                flow_vis[effective_mask == 0] = 0

            if cut_mask is not None and p.get("enable_watershed", False) and p.get("ws_show_lines", True):
                flow_vis[cut_mask > 0] = (0, 255, 255)
        else:
            effective_mask = binary
            flow_vis = None

        speeds: List[float] = []
        contour_angles: List[float] = []
        contour_mags: List[float] = []
        scale_arrow = float(p.get('df_arrow_scale', 5.0))

        for rec in cnt_records:
            is_d = enable_drift and rec["is_drift"]
            bb = rec["bb"]
            center = rec["center"]
            cx, cy = rec["cx"], rec["cy"]

            if is_d:
                if render_vis and drift_show_filtered and not enable_tracking:
                    box_points = np.array(cv2.boxPoints(bb), dtype="int")
                    if show_boxes:
                        cv2.drawContours(flow_vis, [box_points], -1, (0, 140, 255), 1, cv2.LINE_AA)
                    if show_centroids:
                        cv2.circle(flow_vis, center, 2, (0, 140, 255), -1, cv2.LINE_AA)
                    if show_arrows and rec["raw_mag"] > 0:
                        end_x = int(cx + rec["raw_mag"] * scale_arrow * np.cos(rec["raw_ang"]))
                        end_y = int(cy + rec["raw_mag"] * scale_arrow * np.sin(rec["raw_ang"]))
                        cv2.arrowedLine(flow_vis, (cx, cy), (end_x, end_y), (0, 140, 255), 1, cv2.LINE_AA)
                continue

            # Motile contour
            c_mag = rec["corr_mag"]
            c_ang = rec["corr_ang"]
            if c_mag > 0:
                speeds.append(c_mag)
                contour_angles.append(c_ang)
                contour_mags.append(float(c_mag * scale))

            if render_vis:
                box_points = np.array(cv2.boxPoints(bb), dtype="int")
                if show_boxes:
                    cv2.drawContours(flow_vis, [box_points], -1, (0, 255, 0), 1, cv2.LINE_AA)
                if show_centroids:
                    cv2.circle(flow_vis, center, 2, (0, 0, 255), -1, cv2.LINE_AA)

                if not enable_tracking:
                    if enable_drift and drift_show_correction and rec["raw_mag"] > 0:
                        r_end_x = int(cx + rec["raw_mag"] * scale_arrow * np.cos(rec["raw_ang"]))
                        r_end_y = int(cy + rec["raw_mag"] * scale_arrow * np.sin(rec["raw_ang"]))
                        cv2.arrowedLine(flow_vis, (cx, cy), (r_end_x, r_end_y), (0, 200, 255), 1, cv2.LINE_AA)
                    if (show_arrows or (enable_drift and drift_show_correction)) and c_mag > 0:
                        end_x = int(cx + c_mag * scale_arrow * np.cos(c_ang))
                        end_y = int(cy + c_mag * scale_arrow * np.sin(c_ang))
                        cv2.arrowedLine(flow_vis, (cx, cy), (end_x, end_y), (0, 255, 0), 2, cv2.LINE_AA)

        if enable_drift and not enable_tracking:
            total_cnt = int(len(filtered_cnts))
            mobile_cnt = int(len(speeds))
            motility_pct = round(100.0 * mobile_cnt / total_cnt, 2) if total_cnt > 0 else 0.0
            if render_vis:
                DriftProcessor.draw_drift_overlay(
                    flow_vis, drift_info, arrow_scale=scale_arrow, params=p
                )

        mean_speed_px = float(np.mean(speeds)) if speeds else 0.0
        metric_val = round(mean_speed_px * scale, 3)
        if p.get('collect_raw_speeds', False):
            raw_speeds_um_s = [round(float(s * scale), 3) for s in speeds]
        else:
            raw_speeds_um_s = []

        vectors_data = {
            "angles": contour_angles if render_vis else np.asarray(contour_angles, dtype=np.float32),
            "magnitudes": contour_mags if render_vis else np.asarray(contour_mags, dtype=np.float32),
            "mode": "denseflow_contour",
            "count": int(len(filtered_bboxes)),
            "motility_pct": motility_pct,
            "mobile_count": mobile_cnt,
            "total_count": total_cnt,
            "drift_info": drift_info,
        }
        return flow_vis, metric_val, raw_speeds_um_s, detections_list, filtered_bboxes, effective_mask, vectors_data

    @classmethod
    def process_stateless(cls,
                          prev_gray: np.ndarray,
                          gray: np.ndarray,
                          original_bgr: Optional[np.ndarray],
                          params: dict,
                          use_opencl: bool = False,
                          focus_map: Optional[np.ndarray] = None,
                          drift_processor: Optional[DriftProcessor] = None,
                          frame_idx: Optional[int] = None,
                          render_vis: bool = False):
        """Stateless dense flow computation for batch chunk workers."""
        fb_params = {
            'pyr_scale': params.get('df_pyr_scale', 0.5),
            'levels': params.get('df_levels', 2),
            'winsize': params.get('df_winsize', 15),
            'iterations': params.get('df_iterations', 3),
            'poly_n': params.get('df_poly_n', 5),
            'poly_sigma': params.get('df_poly_sigma', 1.2),
        }
        mag, ang, flow_x, flow_y, _ = cls.compute_flow(prev_gray, gray, None, fb_params, use_opencl)
        flow_vis, metric_val, mags_list, detections_list, filtered_bboxes, _, vectors_data = cls.compute_metrics_and_vis(
            mag, ang, flow_x, flow_y, original_bgr, params, use_opencl,
            focus_map=focus_map, drift_processor=drift_processor, frame_idx=frame_idx, render_vis=render_vis
        )
        return flow_vis, metric_val, mags_list, detections_list, vectors_data

    @classmethod
    def process_both_stateless(cls,
                               prev_bgr: np.ndarray,
                               curr_bgr: np.ndarray,
                               params: dict,
                               use_opencl: bool = False,
                               focus_map: Optional[np.ndarray] = None,
                               drift_processor: Optional[DriftProcessor] = None,
                               frame_idx: Optional[int] = None,
                               render_vis: bool = False):
        """Mode 'both': dense optical flow speed masked by contour binarization."""
        fb_params = {
            'pyr_scale': params.get('df_pyr_scale', 0.5),
            'levels': params.get('df_levels', 2),
            'winsize': params.get('df_winsize', 15),
            'iterations': params.get('df_iterations', 3),
            'poly_n': params.get('df_poly_n', 5),
            'poly_sigma': params.get('df_poly_sigma', 1.2),
        }
        calib = float(params.get('df_calibration', 1.0))
        fps = float(params.get('df_fps', 30.0))

        prev_gray, _ = PreProcessor.apply(prev_bgr, params)
        curr_gray, curr_canvas = PreProcessor.apply(curr_bgr, params)

        enable_focus = params.get("enable_preprocessing", True) and params.get("enable_focus_picking", False)
        in_focus_mask = None
        filter_stage = params.get("focus_filter_stage", "Pre-processing")
        use_for_proc = params.get("focus_use_for_processing", False)
        is_post_focus = enable_focus and use_for_proc and filter_stage == "Post-processing"

        if enable_focus and focus_map is None:
            focus_map, _ = PreProcessor.compute_focus(curr_gray, params)
        if enable_focus and focus_map is not None:
            curr_gray, curr_canvas, in_focus_mask = PreProcessor.apply_focus_mask(curr_gray, curr_canvas, focus_map, params)

        mag, ang, flow_x, flow_y, _ = cls.compute_flow(prev_gray, curr_gray, None, fb_params, use_opencl)

        if in_focus_mask is not None and use_for_proc and not is_post_focus:
            mag[in_focus_mask == 0] = 0.0

        mask = ContourProcessor.binarize(curr_gray, params)
        if in_focus_mask is not None and use_for_proc and not is_post_focus:
            mask = cv2.bitwise_and(mask, in_focus_mask)

        enable_contour = params.get("enable_contour_detection", True)
        filtered_boxes, filtered_cnts, detections_list, coverage_pct = ContourProcessor.detect_and_filter(mask, params)
        if is_post_focus and focus_map is not None and filtered_cnts:
            filtered_boxes, filtered_cnts, detections_list = PreProcessor.filter_contours_by_focus(
                focus_map, filtered_boxes, filtered_cnts, detections_list, params
            )
            mask = np.zeros_like(mask)
            if filtered_cnts:
                cv2.drawContours(mask, filtered_cnts, -1, 255, thickness=-1)

        total_px = mask.size
        covered = int(np.count_nonzero(mask))
        coverage_pct = round(100.0 * covered / total_px, 2) if total_px > 0 else 0.0

        scale = calib * fps
        lower_mag = float(params.get('df_lower_mag_thresh', 1.5))
        upper_mag = float(params.get('df_upper_mag_thresh', 0.0))
        masked_mag = mag.copy()
        masked_mag[mask == 0] = 0.0
        masked_mag[masked_mag < lower_mag] = 0.0
        if upper_mag > 0.0:
            masked_mag[masked_mag > upper_mag] = 0.0

        enable_drift = bool(params.get("enable_drift_correction", False))
        drift_show_filtered = bool(params.get("drift_show_filtered", False))
        dp = drift_processor if drift_processor is not None else DriftProcessor()
        H, W = mag.shape[:2]

        ys, xs = np.nonzero(masked_mag > 0)
        drift_info = None
        if render_vis:
            mag_vis = mag.copy()
            if upper_mag > 0.0:
                mag_vis[mag_vis > upper_mag] = 0.0
            ang_vis = ang.copy()
            fx_vis = flow_x.copy()
            fy_vis = flow_y.copy()
        else:
            mag_vis = ang_vis = fx_vis = fy_vis = None
        drift_pixel_mask = None

        if enable_drift:
            drift_res = dp.process_vectors(
                xs, ys, flow_x[ys, xs], flow_y[ys, xs], (H, W), params, frame_idx=frame_idx
            )
            drift_info = drift_res["drift_info"]
            is_drift = drift_res["is_drift"]
            motile_mask = ~is_drift

            valid = drift_res["corr_mag"][motile_mask]
            valid_angs = drift_res["corr_ang"][motile_mask]

            # Update masked_mag so drift pixels are 0 (non-motile) and motile pixels have corrected speed
            masked_mag[ys[is_drift], xs[is_drift]] = 0.0
            masked_mag[ys[motile_mask], xs[motile_mask]] = valid

            if render_vis:
                mag_vis[ys[motile_mask], xs[motile_mask]] = valid
                ang_vis[ys[motile_mask], xs[motile_mask]] = valid_angs
                fx_vis[ys[motile_mask], xs[motile_mask]] = drift_res["corr_vx"][motile_mask]
                fy_vis[ys[motile_mask], xs[motile_mask]] = drift_res["corr_vy"][motile_mask]
                if not drift_show_filtered:
                    mag_vis[ys[is_drift], xs[is_drift]] = 0.0
                else:
                    drift_pixel_mask = np.zeros((H, W), dtype=np.uint8)
                    drift_pixel_mask[ys[is_drift], xs[is_drift]] = 255
        else:
            valid = masked_mag[ys, xs]
            valid_angs = ang[ys, xs]

        mean_speed_px = float(np.mean(valid)) if len(valid) > 0 else 0.0
        mean_speed = round(mean_speed_px * scale, 3)

        total_valid = int(len(valid))
        if params.get('collect_raw_speeds', False):
            if total_valid > 20000:
                raw_stride = total_valid // 20000
                raw_speeds_um_s = (valid[::raw_stride] * scale).tolist()
            else:
                raw_speeds_um_s = (valid * scale).tolist()
        else:
            raw_speeds_um_s = []

        if render_vis:
            vis_mode = params.get('both_vis_mode', 'Surface')
            out_vis = curr_canvas.copy()

            if vis_mode == 'Surface':
                blob_alpha = float(params.get("blob_alpha", 0.5))
                alpha = max(0.0, min(1.0, blob_alpha))
                red_layer = np.zeros_like(out_vis)
                red_layer[:] = (0, 0, 255)
                blended = cv2.addWeighted(out_vis, 1.0 - alpha, red_layer, alpha, 0)
                out_vis = np.where(mask[:, :, None] > 0, blended, out_vis)
            elif vis_mode == 'Speed':
                full_binary = np.full_like(mask, 255)
                if enable_drift and not drift_show_filtered:
                    full_binary[ys[is_drift], xs[is_drift]] = 0
                out_vis = cls.visualize_flow(
                    mag_vis, ang_vis, fx_vis, fy_vis, full_binary, curr_canvas, params, use_opencl,
                    raw_flow_x=flow_x, raw_flow_y=flow_y, drift_mask=drift_pixel_mask
                )
            elif vis_mode == 'Selection':
                sel_mask = mask.copy()
                if enable_drift and not drift_show_filtered:
                    sel_mask[ys[is_drift], xs[is_drift]] = 0
                out_vis = cls.visualize_flow(
                    mag_vis, ang_vis, fx_vis, fy_vis, sel_mask, curr_canvas, params, use_opencl,
                    raw_flow_x=flow_x, raw_flow_y=flow_y, drift_mask=drift_pixel_mask
                )

            if enable_drift and not params.get("enable_tracking", False):
                DriftProcessor.draw_drift_overlay(
                    out_vis, drift_info, arrow_scale=float(params.get('df_arrow_scale', 2.0)), params=params
                )
        else:
            out_vis = None

        max_plot_vectors = 5000
        if total_valid > max_plot_vectors:
            stride = total_valid // max_plot_vectors
            sub_angs = valid_angs[::stride]
            sub_mags = valid[::stride]
        else:
            sub_angs = valid_angs
            sub_mags = valid

        if enable_contour:
            count = int(len(filtered_boxes))
            moving_cnts = 0
            if total_valid > 0 and filtered_cnts:
                for c in filtered_cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    if np.any(masked_mag[y:y+h, x:x+w] > 0):
                        moving_cnts += 1
            motility_pct = round(moving_cnts / count * 100.0, 2) if count > 0 else 0.0
            mobile_count = moving_cnts
            total_count = count
        else:
            count = covered
            mobile_count = total_valid
            total_count = covered
            motility_pct = round(total_valid / covered * 100.0, 2) if covered > 0 else 0.0

        if render_vis:
            ang_out = sub_angs.tolist() if isinstance(sub_angs, np.ndarray) else list(sub_angs)
            mag_out = (sub_mags * scale).tolist() if isinstance(sub_mags, np.ndarray) else list(sub_mags * scale)
        else:
            ang_out = np.asarray(sub_angs, dtype=np.float32)
            mag_out = np.asarray(sub_mags * scale, dtype=np.float32)

        vectors_data = {
            "angles": ang_out,
            "magnitudes": mag_out,
            "mode": "both",
            "count": count,
            "motility_pct": motility_pct,
            "mobile_count": mobile_count,
            "total_count": total_count,
            "drift_info": drift_info,
        }

        return out_vis, coverage_pct, mean_speed, raw_speeds_um_s, vectors_data, detections_list
