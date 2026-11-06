"""
Pre-processing and Focus Picking Engine for Motilitometer Counter Tracker.

Provides:
- Channel isolation (Grayscale, RGB Color, Red Channel, Green Channel, Blue Channel)
- Image inversion (Negative)
- Pre-processing Gaussian blur
- Focus Picking (Modified Laplacian SML, Tenengrad Sobel, Laplacian Variance)
- Focus visualization rendering (Turbo Heatmap, Grayscale Map, Peaking Overlay)
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np


class PreProcessor:
    """Pre-processing pipeline for frames prior to motility and flow analysis."""

    @staticmethod
    def apply_monochrome(frame: np.ndarray, mode: str = "Grayscale") -> np.ndarray:
        """
        Isolates Red, Green, or Blue channel, or converts to Grayscale.
        """
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return frame.copy()

        if mode == "Red Channel":
            return frame[:, :, 2].copy()
        elif mode == "Green Channel":
            return frame[:, :, 1].copy()
        elif mode == "Blue Channel":
            return frame[:, :, 0].copy()
        else:  # "Grayscale" or "RGB (Color)" fallback for gray ops
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        """Applies gamma correction using lookup tables (LUT)."""
        if abs(gamma - 1.0) < 1e-3 or gamma <= 0:
            return image
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def get_crop_center(W: int, H: int, params: dict) -> Tuple[int, int]:
        """Calculates crop center coordinates (cx, cy) from absolute or relative offset."""
        if params.get('crop_use_abs_center', params.get('use_absolute_center', False)):
            cx = int(params.get('crop_center_x', params.get('center_x', W // 2)))
            cy = int(params.get('crop_center_y', params.get('center_y', H // 2)))
        else:
            cx = W // 2 + int(params.get('crop_offset_x', params.get('offset_x', 0)))
            cy = H // 2 + int(params.get('crop_offset_y', params.get('offset_y', 0)))
        return cx, cy

    @classmethod
    def get_crop_mask(cls, H: int, W: int, params: dict) -> np.ndarray:
        """Generates binary mask (255 inside crop, 0 outside)."""
        crop_type = params.get('crop_type', 'Circular' if params.get('circular_crop', False) else 'None')
        if crop_type == 'None' or not params.get('enable_preprocessing', True):
            return np.full((H, W), 255, dtype=np.uint8)

        cx, cy = cls.get_crop_center(W, H, params)
        mask = np.zeros((H, W), dtype=np.uint8)

        if crop_type == 'Circular':
            r = int(params.get('crop_radius', 150))
            cv2.circle(mask, (cx, cy), max(1, r), 255, -1)
        elif crop_type == 'Square':
            size = int(params.get('crop_radius', 150)) * 2 if 'crop_radius' in params and 'crop_width' not in params else int(params.get('crop_width', 300))
            half = max(1, size // 2)
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(W, cx + half)
            y2 = min(H, cy + half)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        elif crop_type == 'Rectangle':
            w = int(params.get('crop_width', 300))
            h = int(params.get('crop_height', 300))
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(W, cx + w // 2)
            y2 = min(H, cy + h // 2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        else:
            mask.fill(255)
        return mask

    @classmethod
    def apply_crop(cls, gray: np.ndarray, canvas: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies configured crop (Circular, Square, Rectangle) with custom center/offsets.
        Returns:
            Tuple of (cropped_gray, cropped_canvas, crop_mask)
        """
        H, W = gray.shape[:2]
        crop_mask = cls.get_crop_mask(H, W, params)
        if np.all(crop_mask == 255):
            return gray, canvas, crop_mask

        gray = cv2.bitwise_and(gray, crop_mask)
        if len(canvas.shape) == 3 and canvas.shape[2] == 3:
            canvas = cv2.bitwise_and(canvas, canvas, mask=crop_mask)
        else:
            canvas = cv2.bitwise_and(canvas, crop_mask)

        return gray, canvas, crop_mask

    @classmethod
    def draw_crop_outline(cls, canvas: np.ndarray, params: dict, cx: Optional[int] = None, cy: Optional[int] = None) -> None:
        """Draws crop contour outline onto canvas in-place for visual feedback."""
        if not params.get('enable_preprocessing', True) or not params.get('show_crop_outline', True):
            return
        crop_type = params.get('crop_type', 'Circular' if params.get('circular_crop', False) else 'None')
        if crop_type == 'None':
            return
        H, W = canvas.shape[:2]
        if cx is None or cy is None:
            cx, cy = cls.get_crop_center(W, H, params)

        outline_color = (255, 255, 0)  # Cyan in BGR
        thickness = 2

        if crop_type == 'Circular':
            r = int(params.get('crop_radius', 150))
            cv2.circle(canvas, (cx, cy), max(1, r), outline_color, thickness, cv2.LINE_AA)
        elif crop_type == 'Square':
            size = int(params.get('crop_radius', 150)) * 2 if 'crop_radius' in params and 'crop_width' not in params else int(params.get('crop_width', 300))
            half = max(1, size // 2)
            cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), outline_color, thickness, cv2.LINE_AA)
        elif crop_type == 'Rectangle':
            w = int(params.get('crop_width', 300))
            h = int(params.get('crop_height', 300))
            cv2.rectangle(canvas, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), outline_color, thickness, cv2.LINE_AA)

    @classmethod
    def apply(cls, frame: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies configured pre-processing:
        1. Channel isolation
        2. Inversion (if enabled)
        3. Gamma adjustment (if gamma != 1.0)
        4. Pre-processing Gaussian blur (if blur > 0)
        5. Crop (Circular / Square / Rectangle with offset or absolute center)

        Returns:
            Tuple of (gray_frame, canvas_bgr_frame)
        """
        if not params.get("enable_preprocessing", True):
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                canvas = frame.copy()
            else:
                gray = frame.copy()
                canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return gray, canvas

        mono_mode = params.get("monochrome_mode", "Grayscale")
        gray = cls.apply_monochrome(frame, mono_mode)

        if params.get("invert_image", False):
            gray = cv2.bitwise_not(gray)

        gamma = float(params.get("gamma", 1.0))
        if abs(gamma - 1.0) > 1e-3:
            gray = cls.adjust_gamma(gray, gamma)

        prep_blur = int(params.get("prep_blur", 0))
        if prep_blur > 0:
            k = prep_blur | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        # Prepare canvas for visualization / downstream drawing
        if mono_mode == "RGB (Color)" and len(frame.shape) == 3 and frame.shape[2] == 3:
            canvas = frame.copy()
            if params.get("invert_image", False):
                canvas = cv2.bitwise_not(canvas)
            if abs(gamma - 1.0) > 1e-3:
                canvas = cls.adjust_gamma(canvas, gamma)
            if prep_blur > 0:
                k = prep_blur | 1
                canvas = cv2.GaussianBlur(canvas, (k, k), 0)
        else:
            canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        crop_type = params.get("crop_type", "None")
        if crop_type != "None":
            gray, canvas, _ = cls.apply_crop(gray, canvas, params)

        if params.get("remove_static", False):
            r1 = params.get("_ref_frame_1", None)
            r2 = params.get("_ref_frame_2", None)
            if r1 is not None and r2 is not None:
                gray, canvas = cls.apply_static_noise_removal(gray, canvas, params, r1, r2)

        return gray, canvas

    @staticmethod
    def apply_static_noise_removal(gray: np.ndarray, canvas: np.ndarray, params: Dict[str, Any],
                                   r1: Optional[np.ndarray], r2: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Removes static background artifacts / parasites using two reference frames and similarity threshold."""
        if not params.get("remove_static", False):
            return gray, canvas
        if r1 is None or r2 is None:
            return gray, canvas
        if r1.shape != gray.shape or r2.shape != gray.shape:
            return gray, canvas

        similarity_threshold = int(params.get("similarity_threshold", 15))
        diff_ref = cv2.absdiff(r1, r2)
        diff_cur = cv2.absdiff(gray, r1)
        _, mask_ref = cv2.threshold(diff_ref, similarity_threshold, 255, cv2.THRESH_BINARY_INV)
        _, mask_cur = cv2.threshold(diff_cur, similarity_threshold, 255, cv2.THRESH_BINARY_INV)
        static_mask = cv2.bitwise_and(mask_ref, mask_cur)
        gray[static_mask > 0] = 0
        if canvas is not None and len(canvas.shape) == 3:
            canvas[static_mask > 0] = 0
        return gray, canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Focus Picking Engine
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_focus_measure(gray_img: np.ndarray, method: str = "Tenengrad (Sobel)") -> np.ndarray:
        """
        Computes high-frequency focus measure map for a single grayscale image.
        Supported methods:
        - "Modified Laplacian (SML)": Sum of Modified Laplacian via 1D convolutions
        - "Tenengrad (Sobel)": Squared gradient magnitude
        - "Laplacian Variance": Absolute Laplacian response
        """
        img_f = gray_img.astype(np.float32)

        if method == "Modified Laplacian (SML)":
            kernel_x = np.array([[-1, 2, -1]], dtype=np.float32)
            kernel_y = np.array([[-1], [2], [-1]], dtype=np.float32)
            lx = cv2.filter2D(img_f, cv2.CV_32F, kernel_x)
            ly = cv2.filter2D(img_f, cv2.CV_32F, kernel_y)
            fm = np.abs(lx) + np.abs(ly)

        elif method == "Tenengrad (Sobel)":
            gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
            fm = gx * gx + gy * gy

        elif method == "Laplacian Variance":
            lap = cv2.Laplacian(img_f, cv2.CV_32F, ksize=3)
            fm = np.abs(lap)

        else:
            kernel_x = np.array([[-1, 2, -1]], dtype=np.float32)
            kernel_y = np.array([[-1], [2], [-1]], dtype=np.float32)
            fm = np.abs(cv2.filter2D(img_f, cv2.CV_32F, kernel_x)) + np.abs(cv2.filter2D(img_f, cv2.CV_32F, kernel_y))

        return fm

    @classmethod
    def compute_focus(cls, gray_frame: np.ndarray, params: dict) -> Tuple[np.ndarray, float]:
        """
        Computes the normalized focus map (0-255 uint8) and mean focus score for a frame.
        """
        if gray_frame is None or gray_frame.size == 0:
            return np.zeros((10, 10), dtype=np.uint8), 0.0

        if len(gray_frame.shape) == 3:
            gray = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_frame

        method = params.get("focus_metric", "Tenengrad (Sobel)")
        window_size = int(params.get("focus_window_size", 15))

        fm = cls.compute_focus_measure(gray, method=method)

        w = int(window_size)
        if w % 2 == 0:
            w += 1
        w = max(3, w)

        fm_smoothed = cv2.GaussianBlur(fm, (w, w), 0)
        mean_score = float(np.mean(fm_smoothed))

        min_val, max_val = float(np.min(fm_smoothed)), float(np.max(fm_smoothed))
        if max_val - min_val > 1e-6:
            norm = ((fm_smoothed - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            norm = np.zeros_like(gray, dtype=np.uint8)

        return norm, mean_score

    @classmethod
    def apply_focus_mask(cls,
                         gray: np.ndarray,
                         canvas: np.ndarray,
                         focus_map: np.ndarray,
                         params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a binary in-focus mask based on focus_threshold.
        Optionally isolates selection in canvas and/or applies mask to gray for downstream processing.

        Returns:
            Tuple of (gray, canvas, in_focus_mask)
        """
        thresh_pct = float(params.get("focus_threshold", 40))
        thresh_pct = max(0.0, min(100.0, thresh_pct))
        thresh = int(thresh_pct * 255.0 / 100.0)
        _, in_focus_mask = cv2.threshold(focus_map, thresh, 255, cv2.THRESH_BINARY)

        isolate = bool(params.get("focus_isolate_selection", False))
        use_for_proc = bool(params.get("focus_use_for_processing", False))
        filter_stage = params.get("focus_filter_stage", "Pre-processing")

        if isolate:
            if len(canvas.shape) == 3 and canvas.shape[2] == 3:
                canvas = cv2.bitwise_and(canvas, canvas, mask=in_focus_mask)
            else:
                canvas = cv2.bitwise_and(canvas, in_focus_mask)

        if use_for_proc and filter_stage != "Post-processing":
            gray = cv2.bitwise_and(gray, in_focus_mask)

        return gray, canvas, in_focus_mask

    @staticmethod
    def filter_contours_by_focus(focus_map: np.ndarray,
                                 filtered_boxes: list,
                                 filtered_cnts: list,
                                 detections_list: list,
                                 params: dict) -> Tuple[list, list, list]:
        """
        Post-processing focus filter: evaluates focus on focus_map for each detected contour.
        Removes any contour whose average focus measure is below the focus_threshold.

        Returns:
            Tuple of (kept_boxes, kept_cnts, kept_detections)
        """
        if focus_map is None or focus_map.size == 0:
            return filtered_boxes, filtered_cnts, detections_list

        thresh_pct = float(params.get("focus_threshold", 40))
        thresh_pct = max(0.0, min(100.0, thresh_pct))
        thresh = thresh_pct * 255.0 / 100.0

        h_img, w_img = focus_map.shape[:2]
        kept_boxes = []
        kept_cnts = []
        kept_detections = []

        n = min(len(filtered_boxes), len(filtered_cnts), len(detections_list))
        for i in range(n):
            box = filtered_boxes[i]
            c = filtered_cnts[i]
            det = detections_list[i]

            if c is not None and len(c) >= 3:
                x, y, w, h = cv2.boundingRect(c)
                x0 = max(0, min(x, w_img - 1))
                y0 = max(0, min(y, h_img - 1))
                x1 = max(x0 + 1, min(x + w, w_img))
                y1 = max(y0 + 1, min(y + h, h_img))

                roi_focus = focus_map[y0:y1, x0:x1]
                c_local = c.copy()
                c_local[:, :, 0] -= x0
                c_local[:, :, 1] -= y0

                roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                cv2.drawContours(roi_mask, [c_local], -1, 255, thickness=-1)
                pts = roi_focus[roi_mask > 0]
                if len(pts) > 0:
                    c_focus = float(np.mean(pts))
                else:
                    c_focus = float(np.mean(roi_focus)) if roi_focus.size > 0 else 0.0
            elif box is not None:
                box_points = np.array(cv2.boxPoints(box), dtype=np.int32)
                x, y, w, h = cv2.boundingRect(box_points)
                x0 = max(0, min(x, w_img - 1))
                y0 = max(0, min(y, h_img - 1))
                x1 = max(x0 + 1, min(x + w, w_img))
                y1 = max(y0 + 1, min(y + h, h_img))

                roi_focus = focus_map[y0:y1, x0:x1]
                b_local = box_points - np.array([x0, y0], dtype=np.int32)
                roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                cv2.fillPoly(roi_mask, [b_local], 255)
                pts = roi_focus[roi_mask > 0]
                c_focus = float(np.mean(pts)) if len(pts) > 0 else 0.0
            else:
                c_focus = 0.0

            if c_focus >= thresh:
                kept_boxes.append(box)
                kept_cnts.append(c)
                kept_detections.append(det)

        return kept_boxes, kept_cnts, kept_detections

    @staticmethod
    def render_focus_visualization(canvas: np.ndarray,
                                   focus_map: np.ndarray,
                                   focus_score: float,
                                   params: dict,
                                   in_focus_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Renders the focus map visualization layer with sensitivity threshold,
        opacity alpha blending, and HUD text overlay.
        Modes:
        - "Heatmap (Turbo)": false-color intensity map blended over canvas with alpha
        - "Peaking (Green Overlay)": in-focus pixels highlighted in bright green with alpha
        - "Grayscale Map": normalized focus map blended with canvas
        """
        vis_mode = params.get("focus_vis_mode", "Heatmap (Turbo)")
        alpha = float(params.get("focus_alpha", 0.6))
        alpha = max(0.0, min(1.0, alpha))
        thresh_pct = float(params.get("focus_threshold", 40))
        thresh_pct = max(0.0, min(100.0, thresh_pct))
        thresh = int(thresh_pct * 255.0 / 100.0)

        if in_focus_mask is None:
            _, in_focus_mask = cv2.threshold(focus_map, thresh, 255, cv2.THRESH_BINARY)

        out = canvas.copy()

        if vis_mode == "Heatmap (Turbo)":
            heatmap = cv2.applyColorMap(focus_map, cv2.COLORMAP_TURBO)
            blended = cv2.addWeighted(canvas, 1.0 - alpha, heatmap, alpha, 0)
            out = np.where(in_focus_mask[:, :, None] > 0, blended, canvas)

        elif vis_mode == "Peaking (Green Overlay)":
            green_layer = np.zeros_like(canvas)
            green_layer[:] = (0, 255, 0)
            blended = cv2.addWeighted(canvas, 1.0 - alpha, green_layer, alpha, 0)
            out = np.where(in_focus_mask[:, :, None] > 0, blended, canvas)

        else:  # "Grayscale Map"
            gray_vis = cv2.cvtColor(focus_map, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(canvas, 1.0 - alpha, gray_vis, alpha, 0)
            out = np.where(in_focus_mask[:, :, None] > 0, blended, canvas)

        cv2.putText(out, f"Focus: {focus_score:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return out
