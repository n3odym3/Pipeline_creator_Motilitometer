"""
Contour and Surface Processing Engine for Motilitometer Counter Tracker.

Encapsulates:
- Adaptive thresholding & morphological closing
- Contour detection & area percentile filtering
- Surface coverage calculation
- Visual rendering (blobs, bounding boxes, centroids, isolation mask)
- Stateless frame processing for multi-threaded batch operations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .preprocessor import PreProcessor


class ContourProcessor:
    """Handles binarization, contour detection, area filtering, and contour rendering."""

    @staticmethod
    def binarize(gray: np.ndarray, params: dict) -> np.ndarray:
        """Applies adaptive Gaussian thresholding and optional morphological closing."""
        block_size = int(params.get("block_size", 25))
        if block_size < 3:
            block_size = 3
        block_size = block_size | 1  # Must be odd

        bin_thresh = int(params.get("bin_thresh", 15))

        blur_k = int(params.get("blur", 3))
        if blur_k > 0:
            blur_k = blur_k | 1
            gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        else:
            gray_blurred = gray

        mask = cv2.adaptiveThreshold(
            gray_blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, bin_thresh
        )

        morph_close = int(params.get("morph_close", 0))
        if morph_close > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close)

        if params.get("fill_contours", False):
            cnts_fill = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cv2.drawContours(mask, cnts_fill, -1, 255, -1)

        return mask

    _last_cut_mask: Optional[np.ndarray] = None

    @classmethod
    def apply_watershed(cls, mask: np.ndarray, params: dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applies Watershed segmentation to split touching objects in a binary mask.

        Args:
            mask: Binary uint8 image (0 = background, 255 = foreground).
            params: Dictionary containing:
                - ws_peak_thresh: Peak distance threshold percentage (0-100 or 0.0-1.0, default 40)
                - ws_show_lines: Whether to generate cut lines for visualization (default True)

        Returns:
            Tuple of (separated_mask, cut_lines_mask)
        """
        if mask is None or not np.any(mask > 0):
            return mask, None

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist <= 0:
            return mask, None

        peak_pct = float(params.get("ws_peak_thresh", 40))
        if peak_pct > 1.0:
            peak_ratio = peak_pct / 100.0
        else:
            peak_ratio = peak_pct
        peak_ratio = np.clip(peak_ratio, 0.05, 0.95)

        thresh_val = float(peak_ratio * max_dist)
        _, sure_fg = cv2.threshold(dist_transform, thresh_val, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_labels, markers = cv2.connectedComponents(sure_fg)
        if num_labels <= 1:
            return mask, None

        markers = markers + 1
        markers[unknown == 255] = 0

        img_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img_3ch, markers)

        cut_lines = (markers == -1)

        separated_mask = mask.copy()
        separated_mask[cut_lines] = 0
        separated_mask[markers <= 1] = 0

        cut_mask = np.uint8(cut_lines * 255) if params.get("ws_show_lines", True) else None
        return separated_mask, cut_mask

    @classmethod
    def detect_and_filter(cls, mask: np.ndarray, params: dict) -> Tuple[List[Any], List[Any], List[Tuple[int, int]], float]:
        """
        Detects contours in mask and filters them by area percentiles.
        If Watershed is enabled, splits touching objects before detecting contours.

        Returns:
            Tuple of (filtered_bounding_boxes, filtered_contours, detections_list, coverage_pct)
        """
        if params.get("enable_watershed", False):
            mask, cut_mask = cls.apply_watershed(mask, params)
            cls._last_cut_mask = cut_mask
        else:
            cls._last_cut_mask = None

        total_pixels = mask.size
        covered_pixels = int(np.count_nonzero(mask))
        coverage_pct = round(100.0 * covered_pixels / total_pixels, 2) if total_pixels > 0 else 0.0

        if not params.get("enable_contour_detection", True):
            return [], [], [], coverage_pct

        lower_surface_thresh = float(params.get("lower_surface_thresh", 25))
        upper_surface_thresh = float(params.get("upper_surface_thresh", 100))

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not cnts:
            return [], [], [], coverage_pct

        calib_bounding_boxes = [cv2.minAreaRect(c) for c in cnts]
        calib_areas = [(box[1][0]) * (box[1][1]) for box in calib_bounding_boxes]

        filtered_bounding_boxes = []
        filtered_cnts = []
        detections_list = []

        if calib_areas:
            left_thresh = np.percentile(calib_areas, lower_surface_thresh)
            right_thresh = np.percentile(calib_areas, upper_surface_thresh)

            for box, c, area in zip(calib_bounding_boxes, cnts, calib_areas):
                if left_thresh <= area <= right_thresh:
                    filtered_bounding_boxes.append(box)
                    filtered_cnts.append(c)
                    center = (int(box[0][0]), int(box[0][1]))
                    detections_list.append(center)

        return filtered_bounding_boxes, filtered_cnts, detections_list, coverage_pct

    @classmethod
    def render_contours(cls,
                        canvas: np.ndarray,
                        mask: np.ndarray,
                        filtered_bounding_boxes: List[Any],
                        detections_list: List[Tuple[int, int]],
                        orig_frame: np.ndarray,
                        params: dict,
                        cut_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders contour annotations (boxes, blobs, centroids, watershed cuts) on the canvas.
        Returns:
            Tuple of (rendered_frame, selection_mask)
        """
        visu_format = params.get("visu_format", "Frame")
        show_blobs = params.get("show_blobs", True)
        blob_alpha = float(params.get("blob_alpha", 0.5))
        isolate_selection = params.get("isolate_selection", False)
        show_boxes = params.get("show_boxes", True)
        show_centroids = params.get("show_centroids", True)

        if visu_format == "Mask":
            frame_contour = np.zeros_like(canvas, dtype=np.uint8)
        else:
            frame_contour = canvas.copy()

        if len(frame_contour.shape) == 2:
            frame_contour = cv2.cvtColor(frame_contour, cv2.COLOR_GRAY2BGR)

        if show_blobs and mask is not None and np.any(mask > 0):
            alpha = max(0.0, min(1.0, blob_alpha))
            blob_overlay = frame_contour.copy()
            blob_overlay[mask > 0] = (0, 0, 255)
            blended = cv2.addWeighted(frame_contour, 1.0 - alpha, blob_overlay, alpha, 0)
            frame_contour = np.where(mask[:, :, None] > 0, blended, frame_contour)

        selection_mask = np.zeros_like(mask, dtype=np.uint8)

        for bounding_box, center in zip(filtered_bounding_boxes, detections_list):
            box_points = np.array(cv2.boxPoints(bounding_box), dtype="int")
            if isolate_selection:
                cv2.fillPoly(selection_mask, [box_points], 255)
            if show_boxes:
                cv2.drawContours(frame_contour, [box_points], -1, (0, 255, 0), 1, cv2.LINE_AA)
            if show_centroids:
                cv2.circle(frame_contour, center, 2, (0, 0, 255), -1, cv2.LINE_AA)

        # Watershed cut lines visualization
        if cut_mask is None:
            cut_mask = getattr(cls, "_last_cut_mask", None)
        if params.get("enable_watershed", False) and params.get("ws_show_lines", True) and cut_mask is not None and np.any(cut_mask > 0):
            frame_contour[cut_mask > 0] = (0, 255, 255)  # Bright yellow/cyan cut line

        if isolate_selection and mask is not None:
            if visu_format == "Mask":
                frame_contour[selection_mask == 255] = orig_frame[selection_mask == 255]
            else:
                frame_contour[selection_mask != 255] = 0

        return frame_contour, selection_mask

    @classmethod
    def process_stateless(cls, frame: np.ndarray, params: dict) -> Tuple[np.ndarray, float, list]:
        """
        Stateless contour processing for batch video processing chunks.
        Applies pre-processing, binarization, contour detection, and rendering.
        """
        if not isinstance(frame, np.ndarray):
            return frame, 0.0, []

        gray, canvas = PreProcessor.apply(frame, params)
        mask = cls.binarize(gray, params)
        filtered_boxes, _, detections_list, coverage_pct = cls.detect_and_filter(mask, params)

        frame_out, _ = cls.render_contours(canvas, mask, filtered_boxes, detections_list, canvas, params)
        return frame_out, coverage_pct, filtered_boxes
