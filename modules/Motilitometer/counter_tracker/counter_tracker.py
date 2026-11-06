"""
Motilitometer Counter Tracker Main Processing Module.

Orchestrates modular pipelines:
- PreProcessor: Channel isolation, inversion, blur, and focus picking engine.
- ContourProcessor: Binarization, contour detection, area filtering, and surface coverage.
- DenseFlowProcessor: GPU/CPU Farneback optical flow computation, velocity fields, and visual layers.
- ObjectTracker: Norfair object tracking, trajectory management, speed, and vector export.
- DriftProcessor: Sedimentation / drift estimation, locking, vector subtraction, and visualization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.processing_base import ProcessingBase
from .preprocessor import PreProcessor
from .object_tracker import ObjectTracker
from .contour_processor import ContourProcessor
from .denseflow_processor import DenseFlowProcessor
from .drift_processor import DriftProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Top-level Batch Helpers & Workers (Picklable for ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_OPENCL_AVAILABLE: Optional[bool] = None
_WORKER_THREADS_CONFIGURED: bool = False


def _open_batch_video_capture(video_path: str) -> cv2.VideoCapture:
    """Opens a VideoCapture preferring software FFmpeg to avoid concurrent MSMF/D3D11 driver locks across workers."""
    try:
        if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_NONE"):
            cap = cv2.VideoCapture(
                video_path,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE],
            )
        else:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()
    except Exception:
        pass
    return cv2.VideoCapture(video_path)


def batch_worker_init(n_workers: int = 8) -> None:
    """
    Initializes each ProcessPoolExecutor worker process once:
    - Balances internal OpenCV threads (cpu_count // n_workers) so all CPU cores are utilized
      at full speed without triggering an (n_workers * cpu_count) thread storm in the OS scheduler.
    - Probes and caches OpenCL availability once per worker process instead of on every chunk.
    """
    global _WORKER_OPENCL_AVAILABLE, _WORKER_THREADS_CONFIGURED
    import os as _os
    import cv2 as _cv2

    cpu_cores = _os.cpu_count() or 8
    threads_per_worker = max(1, cpu_cores // max(1, int(n_workers)))
    try:
        _cv2.setNumThreads(threads_per_worker)
    except Exception:
        pass
    _WORKER_THREADS_CONFIGURED = True
    _WORKER_OPENCL_AVAILABLE = DenseFlowProcessor.check_opencl()


def _get_worker_opencl(params: dict) -> bool:
    global _WORKER_OPENCL_AVAILABLE, _WORKER_THREADS_CONFIGURED
    if not _WORKER_THREADS_CONFIGURED:
        batch_worker_init(int(params.get("_n_workers", 8)))
    if _WORKER_OPENCL_AVAILABLE is None:
        _WORKER_OPENCL_AVAILABLE = DenseFlowProcessor.check_opencl()
    return bool(_WORKER_OPENCL_AVAILABLE)


def batch_prepare_video_params(video_path: str, total_frames: int, params: dict) -> dict:
    """
    Prepares batch parameters in the worker thread before launching ProcessPoolExecutor:
    1. Determines preprocessed frame shape `_frame_shape` for headless grid subdivision.
    2. Pre-computes locked drift cell states at active range boundaries (e.g. [0-500],[1000-1500])
       so parallel chunk workers in denseflow/both modes have the exact locked drift vectors.
    """
    import cv2 as _cv2
    prepared = dict(params)
    prepared["collect_raw_speeds"] = False
    cap = _open_batch_video_capture(video_path)
    if not cap.isOpened():
        return prepared

    try:
        ret, frame0 = cap.read()
        if not ret or frame0 is None:
            return prepared
        gray0, _ = PreProcessor.apply(frame0, prepared)
        prepared["_frame_shape"] = (int(gray0.shape[0]), int(gray0.shape[1]))

        if bool(prepared.get("drift_only_export", False)) and not bool(prepared.get("drift_reuse_preprocessed", False)):
            prepared["enable_drift_correction"] = True

        enable_drift = bool(prepared.get("enable_drift_correction", False))
        if enable_drift and bool(prepared.get("drift_reuse_preprocessed", False)):
            drift_path, preproc_map = DriftProcessor.find_and_load_drift_for_video(video_path)
            if preproc_map is not None:
                prepared["_preprocessed_drift_map"] = preproc_map
                prepared["_preprocessed_drift_path"] = drift_path
                prepared["_video_path"] = video_path
                prepared["drift_only_export"] = False
                return prepared

        mode = prepared.get("processing_mode", "contour")
        if not enable_drift or mode not in ("denseflow", "both"):
            return prepared

        use_ranges = bool(prepared.get("drift_use_ranges", False))
        lock_live = bool(prepared.get("drift_lock_live", False))
        if not use_ranges and not lock_live:
            return prepared

        # Identify lock boundary frames where drift decision freezes
        lock_frames: List[int] = []
        if lock_live:
            lock_frames.append(1)
        if use_ranges:
            intervals = DriftProcessor.parse_frame_ranges(prepared.get("drift_active_ranges", ""))
            for _, end_f in intervals:
                if 1 <= end_f < total_frames:
                    lock_frames.append(int(end_f))

        lock_frames = sorted(set(lock_frames))
        if not lock_frames:
            return prepared

        use_opencl = DenseFlowProcessor.check_opencl()
        enable_focus = prepared.get("enable_preprocessing", True) and prepared.get("enable_focus_picking", False)
        locked_states: Dict[int, Dict[Tuple[int, int], Dict[str, Any]]] = {}
        dp = DriftProcessor()

        # Evaluate optical flow at each lock boundary frame (and up to 2 preceding frames for cell fallback)
        p_eval = {**prepared, "drift_lock_live": False, "drift_use_ranges": False}
        for lf in lock_frames:
            start_eval = max(1, lf - 2)
            cap.set(_cv2.CAP_PROP_POS_FRAMES, start_eval - 1)
            ret_prev, prev_f = cap.read()
            if not ret_prev or prev_f is None:
                continue

            if mode == "denseflow":
                prev_gray, _ = PreProcessor.apply(prev_f, p_eval)
                if enable_focus:
                    fmap, _ = PreProcessor.compute_focus(prev_gray, p_eval)
                    prev_gray, _, _ = PreProcessor.apply_focus_mask(prev_gray, prev_gray, fmap, p_eval)

                for cur_idx in range(start_eval, lf + 1):
                    ret_c, cur_f = cap.read()
                    if not ret_c or cur_f is None:
                        break
                    gray, orig = PreProcessor.apply(cur_f, p_eval)
                    fmap = None
                    if enable_focus:
                        fmap, _ = PreProcessor.compute_focus(gray, p_eval)
                        gray, orig, _ = PreProcessor.apply_focus_mask(gray, orig, fmap, p_eval)
                    DenseFlowProcessor.process_stateless(
                        prev_gray, gray, orig, p_eval, use_opencl,
                        focus_map=fmap, drift_processor=dp, frame_idx=cur_idx
                    )
                    prev_gray = gray
            else:  # mode == "both"
                prev_bgr = prev_f
                for cur_idx in range(start_eval, lf + 1):
                    ret_c, cur_f = cap.read()
                    if not ret_c or cur_f is None:
                        break
                    DenseFlowProcessor.process_both_stateless(
                        prev_bgr, cur_f, p_eval, use_opencl,
                        drift_processor=dp, frame_idx=cur_idx
                    )
                    prev_bgr = cur_f

            locked_states[lf] = {k: dict(v) for k, v in dp._cell_states.items()}

        prepared["_batch_locked_drift_states"] = locked_states
    finally:
        cap.release()

    return prepared


def batch_worker_process_chunk(video_path: str, start: int, end: int, params: dict):
    """Contour worker for ProcessPoolExecutor."""
    import cv2 as _cv2
    if not _WORKER_THREADS_CONFIGURED:
        batch_worker_init(int(params.get("_n_workers", 8)))
    cap = _open_batch_video_capture(video_path)
    cap.set(_cv2.CAP_PROP_POS_FRAMES, start)
    results = []
    enable_contour = params.get("enable_contour_detection", True)
    enable_focus = params.get("enable_preprocessing", True) and params.get("enable_focus_picking", False)
    use_focus_proc = params.get("focus_use_for_processing", False)
    filter_stage = params.get("focus_filter_stage", "Pre-processing")
    is_post_focus = enable_focus and use_focus_proc and filter_stage == "Post-processing"

    try:
        for fi in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            gray, _ = PreProcessor.apply(frame, params)
            focus_map = None
            in_focus_mask = None
            if enable_focus:
                focus_map, _ = PreProcessor.compute_focus(gray, params)
                gray, _, in_focus_mask = PreProcessor.apply_focus_mask(gray, gray, focus_map, params)

            mask = ContourProcessor.binarize(gray, params)
            if in_focus_mask is not None and use_focus_proc and not is_post_focus:
                mask = _cv2.bitwise_and(mask, in_focus_mask)

            filtered_boxes, filtered_cnts, detections_list, coverage_pct = ContourProcessor.detect_and_filter(mask, params)
            if is_post_focus and focus_map is not None and filtered_cnts:
                filtered_boxes, filtered_cnts, detections_list = PreProcessor.filter_contours_by_focus(
                    focus_map, filtered_boxes, filtered_cnts, detections_list, params
                )
                mask = np.zeros_like(mask)
                if filtered_cnts:
                    _cv2.drawContours(mask, filtered_cnts, -1, 255, thickness=-1)
                total_pixels = mask.size
                covered_pixels = int(np.count_nonzero(mask))
                coverage_pct = round(100.0 * covered_pixels / total_pixels, 2) if total_pixels > 0 else 0.0

            cnt = int(len(filtered_boxes)) if enable_contour else int(np.count_nonzero(mask))
            results.append((start + fi, coverage_pct, detections_list, cnt, None))
    finally:
        cap.release()
    return results


def batch_worker_denseflow_chunk(video_path: str, start: int, end: int, params: dict):
    """Dense Flow worker for ProcessPoolExecutor."""
    import cv2 as _cv2
    use_opencl = _get_worker_opencl(params)
    cap = _open_batch_video_capture(video_path)
    results = []
    prev_gray = None
    enable_focus = params.get("enable_preprocessing", True) and params.get("enable_focus_picking", False)
    dp = DriftProcessor(
        precomputed_lock_states=params.get("_batch_locked_drift_states"),
        preprocessed_drift_map=params.get("_preprocessed_drift_map"),
    )

    try:
        # Seek to start - 1 when start > 0 so chunk boundary frame `start` is not skipped
        if start > 0:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, start - 1)
            ret_prev, prev_frame = cap.read()
            if ret_prev and prev_frame is not None:
                prev_gray, _ = PreProcessor.apply(prev_frame, params)
                if enable_focus:
                    fmap_prev, _ = PreProcessor.compute_focus(prev_gray, params)
                    prev_gray, _, _ = PreProcessor.apply_focus_mask(prev_gray, prev_gray, fmap_prev, params)
        else:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)

        for fi in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = start + fi
            gray, original = PreProcessor.apply(frame, params)
            focus_map = None
            if enable_focus:
                focus_map, _ = PreProcessor.compute_focus(gray, params)
                gray, original, _ = PreProcessor.apply_focus_mask(gray, original, focus_map, params)

            if prev_gray is None or prev_gray.shape != gray.shape:
                prev_gray = gray
                continue
            _, metric_val, mags_list, detections_list, vectors_data = DenseFlowProcessor.process_stateless(
                prev_gray, gray, original, params, use_opencl,
                focus_map=focus_map, drift_processor=dp, frame_idx=frame_idx
            )
            cnt = vectors_data.get("count", len(mags_list))
            results.append((frame_idx, metric_val, mags_list, detections_list, vectors_data))
            prev_gray = gray
    finally:
        if use_opencl:
            try:
                _cv2.ocl.finish()
            except Exception:
                pass
        cap.release()
    return results


def batch_worker_both_chunk(video_path: str, start: int, end: int, params: dict):
    """'Both' mode worker for ProcessPoolExecutor."""
    import cv2 as _cv2
    use_opencl = _get_worker_opencl(params)
    cap = _open_batch_video_capture(video_path)
    results = []
    prev_bgr = None
    dp = DriftProcessor(
        precomputed_lock_states=params.get("_batch_locked_drift_states"),
        preprocessed_drift_map=params.get("_preprocessed_drift_map"),
    )

    try:
        if start > 0:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, start - 1)
            ret_prev, prev_frame = cap.read()
            if ret_prev and prev_frame is not None:
                prev_bgr = prev_frame
        else:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)

        for fi in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = start + fi
            if prev_bgr is None or prev_bgr.shape != frame.shape:
                prev_bgr = frame
                continue
            out_res = DenseFlowProcessor.process_both_stateless(
                prev_bgr, frame, params, use_opencl,
                drift_processor=dp, frame_idx=frame_idx
            )
            coverage_pct, mean_speed, raw_speeds, vectors_data, detections_list = (
                out_res[1], out_res[2], out_res[3], out_res[4], out_res[5]
            )
            cnt = vectors_data.get("count", 0)
            motility = vectors_data.get("motility_pct", 0.0)
            results.append((frame_idx, coverage_pct, mean_speed, raw_speeds, cnt, motility, detections_list,
                            (vectors_data.get("angles"), vectors_data.get("magnitudes")),
                            vectors_data.get("drift_info")))
            prev_bgr = frame
    finally:
        if use_opencl:
            try:
                _cv2.ocl.finish()
            except Exception:
                pass
        cap.release()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main Processor Class
# ─────────────────────────────────────────────────────────────────────────────

class MotilitometerProcessor(ProcessingBase):
    """Combined pipeline for motility, tracking, optical flow, and drift correction analysis.

    Supports processing modes:
    - ``mode='contour'``   : pre-processing -> binarization -> contour detection -> tracking (+ drift correction).
    - ``mode='denseflow'`` : pre-processing -> Farneback dense optical flow (+ drift correction).
    - ``mode='both'``      : pre-processing -> binarization mask + dense optical flow speed (+ drift correction).
    """

    def __init__(self,
                 params: Optional[Dict[str, Any]] = None,
                 *,
                 buffer_size: int = 10,
                 drop_policy: str = 'drop_new',
                 daemon: bool = True) -> None:
        super().__init__(params=params, buffer_size=buffer_size,
                         drop_policy=drop_policy, daemon=daemon)

        # Engines are initialized lazily / inside the worker process
        # to ensure safe pickling across multiprocessing boundaries (spawn method).
        self.tracker: Optional[ObjectTracker] = None
        self.denseflow_processor: Optional[DenseFlowProcessor] = None

        self._df_prev_gray: Optional[np.ndarray] = None
        self._df_last_umat_flow = None
        self._df_prev_bgr: Optional[np.ndarray] = None
        self._live_frame_idx: int = 0
        self._last_reset_trig: int = 0

        self._ref_frame_1: Optional[np.ndarray] = None
        self._ref_frame_2: Optional[np.ndarray] = None
        self._last_r1_trigger: int = 0
        self._last_r2_trigger: int = 0
        self._last_clear_ref_trigger: int = 0

    def __getstate__(self) -> Dict[str, Any]:
        """Excludes unpicklable / worker-specific instances from serialization."""
        state = self.__dict__.copy()
        state['tracker'] = None
        state['denseflow_processor'] = None
        state['_df_prev_gray'] = None
        state['_df_last_umat_flow'] = None
        state['_df_prev_bgr'] = None
        state['_ref_frame_1'] = None
        state['_ref_frame_2'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores state in the worker process and initializes worker engines."""
        self.__dict__.update(state)
        self.tracker = None
        self.denseflow_processor = None
        self._df_prev_gray = None
        self._df_last_umat_flow = None
        self._df_prev_bgr = None
        self._live_frame_idx = 0
        self._last_reset_trig = 0
        self._ref_frame_1 = None
        self._ref_frame_2 = None
        self._last_r1_trigger = 0
        self._last_r2_trigger = 0
        self._last_clear_ref_trigger = 0
        self._ensure_worker_initialized()

    def _ensure_worker_initialized(self) -> None:
        """Initializes engines inside the worker process."""
        if self.tracker is None:
            dist_func = self.params.get('distance_function', 'euclidean') if self.params else 'euclidean'
            dist_thresh = self.params.get('distance_threshold', 30) if self.params else 30
            self.tracker = ObjectTracker(
                distance_function=dist_func,
                distance_threshold=dist_thresh,
            )
        if self.denseflow_processor is None:
            self.denseflow_processor = DenseFlowProcessor()

    @property
    def tracking(self) -> Dict[int, Any]:
        """Provides backwards compatibility for accessing tracked objects dict."""
        return self.tracker.tracking if self.tracker is not None else {}

    def _reset_denseflow_state(self) -> None:
        """Resets internal dense flow and drift state (source change, video restart, etc.)."""
        self._df_prev_gray = None
        self._df_last_umat_flow = None
        self._df_prev_bgr = None
        self._live_frame_idx = 0
        if self.denseflow_processor is not None:
            self.denseflow_processor.reset_drift()
        if self.tracker is not None:
            self.tracker.drift_processor.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # Live Processing Dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _process_data(self, data: Any, p: Dict[str, Any]) -> Any:
        self._ensure_worker_initialized()
        frame_idx = None
        if isinstance(data, tuple) and len(data) == 2:
            frame, frame_idx = data
        else:
            frame = data

        reset_trig = int(p.get("reset_trigger", 0))
        if reset_trig > self._last_reset_trig:
            self._reset_denseflow_state()
            self._last_reset_trig = reset_trig

        if frame_idx is not None:
            eff_frame_idx = int(frame_idx)
            self._live_frame_idx = eff_frame_idx + 1
        else:
            eff_frame_idx = self._live_frame_idx
            self._live_frame_idx += 1

        # Apply base preprocessing without static noise removal first
        p_no_static = {**p, "remove_static": False}
        gray, canvas = PreProcessor.apply(frame, p_no_static)

        # Reference frames capture triggers & static noise removal
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

        if self._ref_frame_1 is not None and self._ref_frame_1.shape != gray.shape:
            self._ref_frame_1 = None
        if self._ref_frame_2 is not None and self._ref_frame_2.shape != gray.shape:
            self._ref_frame_2 = None

        if p.get("remove_static", False) and self._ref_frame_1 is not None and self._ref_frame_2 is not None:
            gray, canvas = PreProcessor.apply_static_noise_removal(
                gray, canvas, p, self._ref_frame_1, self._ref_frame_2
            )

        focus_score = None
        focus_vis = None
        in_focus_mask = None
        focus_map = None
        if p.get("enable_preprocessing", True) and p.get("enable_focus_picking", False):
            focus_map, focus_score = PreProcessor.compute_focus(gray, p)
            gray, canvas, in_focus_mask = PreProcessor.apply_focus_mask(gray, canvas, focus_map, p)
            if p.get("focus_output_map", True):
                focus_vis = PreProcessor.render_focus_visualization(canvas, focus_map, focus_score, p, in_focus_mask=in_focus_mask)

        # 2. Downstream Processing (Active / Processing checkbox)
        is_processing_active = p.get("active", True)
        if not is_processing_active:
            out_frame = (focus_vis if focus_vis is not None else canvas).copy()
            PreProcessor.draw_crop_outline(out_frame, p)
            vectors_data = {"angles": [], "magnitudes": [], "mode": "bypass", "count": 0, "focus_score": focus_score}
            return (out_frame, {}, [], None, 0.0, vectors_data), frame_idx

        mode = p.get('processing_mode', 'contour')
        if mode == 'denseflow':
            res = self._process_denseflow_preprocessed(
                frame, gray, canvas, focus_score, focus_vis, in_focus_mask, p,
                focus_map=focus_map, frame_idx=eff_frame_idx
            )
        elif mode == 'both':
            res = self._process_both_preprocessed(
                frame, gray, canvas, focus_score, focus_vis, in_focus_mask, p,
                focus_map=focus_map, frame_idx=eff_frame_idx
            )
        else:
            res = self._process_contour_preprocessed(
                frame, gray, canvas, focus_score, focus_vis, in_focus_mask, p,
                focus_map=focus_map, frame_idx=eff_frame_idx
            )

        if res and isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], np.ndarray):
            PreProcessor.draw_crop_outline(res[0], p)

        return res, frame_idx

    # ── Mode Contour ─────────────────────────────────────────────────────────

    def _process_contour_preprocessed(self,
                                      frame: np.ndarray,
                                      gray: np.ndarray,
                                      canvas: np.ndarray,
                                      focus_score: Optional[float],
                                      focus_vis: Optional[np.ndarray],
                                      in_focus_mask: Optional[np.ndarray],
                                      p: Dict[str, Any],
                                      focus_map: Optional[np.ndarray] = None,
                                      frame_idx: Optional[int] = None) -> Any:
        # Check manual tracker reset
        self.tracker.check_reset_trigger(
            p.get('reset_trigger', 0),
            p.get('distance_function', 'euclidean'),
            p.get('distance_threshold', 30),
            hit_counter_max=int(p.get('track_max_lost', 15))
        )

        filter_stage = p.get("focus_filter_stage", "Pre-processing")
        is_post_focus = (
            p.get("enable_preprocessing", True)
            and p.get("enable_focus_picking", False)
            and p.get("focus_use_for_processing", False)
            and filter_stage == "Post-processing"
            and focus_map is not None
        )

        # 2. Binarization & Coverage
        mask = ContourProcessor.binarize(gray, p)
        if in_focus_mask is not None and p.get("focus_use_for_processing", False) and not is_post_focus:
            mask = cv2.bitwise_and(mask, in_focus_mask)

        filtered_boxes, filtered_cnts, detections_list, coverage_pct = ContourProcessor.detect_and_filter(mask, p)

        if is_post_focus and filtered_cnts:
            filtered_boxes, filtered_cnts, detections_list = PreProcessor.filter_contours_by_focus(
                focus_map, filtered_boxes, filtered_cnts, detections_list, p
            )
            mask = np.zeros_like(mask)
            if filtered_cnts:
                cv2.drawContours(mask, filtered_cnts, -1, 255, thickness=-1)
            total_pixels = mask.size
            covered_pixels = int(np.count_nonzero(mask))
            coverage_pct = round(100.0 * covered_pixels / total_pixels, 2) if total_pixels > 0 else 0.0

        enable_contour = bool(p.get("enable_contour_detection", True))
        enable_tracking = bool(p.get("enable_tracking", False))
        enable_drift = bool(p.get("enable_drift_correction", False))
        drift_show_filtered = bool(p.get("drift_show_filtered", False))

        if enable_contour:
            count = int(len(filtered_boxes))
        else:
            count = int(np.count_nonzero(mask))

        # 3. Contour Rendering
        base_canvas = focus_vis if focus_vis is not None else canvas
        if enable_tracking and enable_drift:
            # Defer box/centroid drawing until after tracker classifies drift vs motile tracks
            p_render = {**p, "show_boxes": False, "show_centroids": False}
            frame_contour, _ = ContourProcessor.render_contours(
                base_canvas, mask, filtered_boxes, detections_list, base_canvas, p_render
            )
        else:
            frame_contour, _ = ContourProcessor.render_contours(
                base_canvas, mask, filtered_boxes, detections_list, base_canvas, p
            )

        # 4. Tracking (+ Drift Correction)
        frame_contour, tracking_dict, vectors_data, mean_trk_speed, raw_trk_speeds = self.tracker.update(
            detections_list, frame_contour, p, mode_label="contour", frame_idx=frame_idx
        )

        # Render boxes/centroids respecting drift filtering if tracking + drift correction is active
        if enable_tracking and enable_drift and (p.get("show_boxes", False) or p.get("show_centroids", False)):
            show_boxes = bool(p.get("show_boxes", False))
            show_centroids = bool(p.get("show_centroids", False))
            drift_pts = [
                tdata["points"][-1]
                for tdata in self.tracker.tracking.values()
                if tdata.get("is_drift", False) and tdata.get("points")
            ]
            for bb, center in zip(filtered_boxes, detections_list):
                is_drift_det = any(abs(center[0] - dp[0]) <= 3 and abs(center[1] - dp[1]) <= 3 for dp in drift_pts)
                if is_drift_det:
                    if drift_show_filtered:
                        box_points = np.array(cv2.boxPoints(bb), dtype="int")
                        if show_boxes:
                            cv2.drawContours(frame_contour, [box_points], -1, (0, 140, 255), 1, cv2.LINE_AA)
                        if show_centroids:
                            cv2.circle(frame_contour, center, 2, (0, 140, 255), -1, cv2.LINE_AA)
                else:
                    box_points = np.array(cv2.boxPoints(bb), dtype="int")
                    if show_boxes:
                        cv2.drawContours(frame_contour, [box_points], -1, (0, 255, 0), 1, cv2.LINE_AA)
                    if show_centroids:
                        cv2.circle(frame_contour, center, 2, (0, 0, 255), -1, cv2.LINE_AA)

        vectors_data["focus_score"] = focus_score
        vectors_data["count"] = count
        vectors_data["mean_speed"] = round(mean_trk_speed, 3)
        vectors_data["raw_speeds"] = raw_trk_speeds
        if enable_tracking:
            vectors_data["mobile_count"] = vectors_data.get("mobile_tracks")
            vectors_data["total_count"] = vectors_data.get("total_tracks")
        else:
            vectors_data["motility_pct"] = None
            vectors_data["mobile_count"] = None
            vectors_data["total_count"] = None

        use_tracking_speed = bool(p.get("use_tracking_speed", False)) and enable_tracking
        if use_tracking_speed:
            metric_val = (round(mean_trk_speed, 3), raw_trk_speeds)
        else:
            metric_val = coverage_pct

        return frame_contour, tracking_dict, filtered_boxes, mask, metric_val, vectors_data

    # ── Mode Dense Flow ──────────────────────────────────────────────────────

    def _process_denseflow_preprocessed(self,
                                        frame: np.ndarray,
                                        gray: np.ndarray,
                                        original: np.ndarray,
                                        focus_score: Optional[float],
                                        focus_vis: Optional[np.ndarray],
                                        in_focus_mask: Optional[np.ndarray],
                                        p: Dict[str, Any],
                                        focus_map: Optional[np.ndarray] = None,
                                        frame_idx: Optional[int] = None) -> Any:
        # Check manual tracker reset
        self.tracker.check_reset_trigger(
            p.get('reset_trigger', 0),
            p.get('distance_function', 'euclidean'),
            p.get('distance_threshold', 30),
            hit_counter_max=int(p.get('track_max_lost', 15))
        )

        base_canvas = focus_vis if focus_vis is not None else original

        if self._df_prev_gray is None or self._df_prev_gray.shape != gray.shape:
            self._df_prev_gray = gray
            self._df_last_umat_flow = None
            out_canvas = base_canvas.copy()
            drift_info = None
            if p.get("enable_drift_correction", False):
                d_res = self.denseflow_processor.drift_processor.process_vectors(
                    [], [], [], [], gray.shape[:2], p, frame_idx=frame_idx
                )
                drift_info = d_res.get("drift_info")
                if drift_info is not None:
                    DriftProcessor.draw_drift_overlay(
                        out_canvas, drift_info, arrow_scale=float(p.get("df_arrow_scale", 3.0)), params=p
                    )
            return out_canvas, {}, [], None, (0.0, []), {
                "angles": [], "magnitudes": [], "mode": "denseflow", "count": 0,
                "focus_score": focus_score, "drift_info": drift_info
            }

        fb_params = {
            'pyr_scale': p.get('df_pyr_scale', 0.5),
            'levels': p.get('df_levels', 2),
            'winsize': p.get('df_winsize', 15),
            'iterations': p.get('df_iterations', 3),
            'poly_n': p.get('df_poly_n', 5),
            'poly_sigma': p.get('df_poly_sigma', 1.2),
        }
        mag, ang, flow_x, flow_y, self._df_last_umat_flow = DenseFlowProcessor.compute_flow(
            self._df_prev_gray, gray, self._df_last_umat_flow, fb_params, self.denseflow_processor.opencl_available
        )
        self._df_prev_gray = gray

        filter_stage = p.get("focus_filter_stage", "Pre-processing")
        is_post_focus = (
            p.get("enable_preprocessing", True)
            and p.get("enable_focus_picking", False)
            and p.get("focus_use_for_processing", False)
            and filter_stage == "Post-processing"
            and focus_map is not None
        )

        if in_focus_mask is not None and p.get("focus_use_for_processing", False) and not is_post_focus:
            mag[in_focus_mask == 0] = 0.0

        flow_vis, metric_val, raw_speeds, detections_list, filtered_bboxes, mask, vectors_data = (
            DenseFlowProcessor.compute_metrics_and_vis(
                mag, ang, flow_x, flow_y, base_canvas, p,
                self.denseflow_processor.opencl_available,
                focus_map=focus_map,
                drift_processor=self.denseflow_processor.drift_processor,
                frame_idx=frame_idx,
            )
        )

        use_contours = p.get('df_use_contours', False) or p.get('enable_tracking', False)
        enable_tracking = bool(p.get('enable_tracking', False))
        if use_contours and enable_tracking:
            flow_vis, tracking_dict, trk_vectors, mean_trk_speed, raw_trk_speeds = self.tracker.update(
                detections_list, flow_vis, p, mode_label="denseflow_contour", frame_idx=frame_idx
            )
            if trk_vectors.get("angles"):
                vectors_data["angles"].extend(trk_vectors["angles"])
                vectors_data["magnitudes"].extend(trk_vectors["magnitudes"])
            vectors_data["motility_pct"] = trk_vectors.get("motility_pct")
            vectors_data["mobile_count"] = trk_vectors.get("mobile_tracks")
            vectors_data["total_count"] = trk_vectors.get("total_tracks")
            if trk_vectors.get("drift_info") is not None:
                vectors_data["drift_info"] = trk_vectors["drift_info"]
        else:
            tracking_dict = {}
            mean_trk_speed, raw_trk_speeds = 0.0, []
            if not p.get("enable_drift_correction", False):
                vectors_data["motility_pct"] = None
                vectors_data["mobile_count"] = None
                vectors_data["total_count"] = None

        lower_mag = float(p.get('df_lower_mag_thresh', 1.5))
        upper_mag = float(p.get('df_upper_mag_thresh', 0.0))
        if use_contours:
            count = int(len(filtered_bboxes))
        else:
            if upper_mag > 0.0:
                fallback_cnt = int(np.count_nonzero((mag >= lower_mag) & (mag <= upper_mag)))
            else:
                fallback_cnt = int(np.count_nonzero(mag >= lower_mag))
            count = vectors_data.get("count", fallback_cnt)
        vectors_data["count"] = count

        use_tracking_speed = bool(p.get("use_tracking_speed", False)) and bool(enable_tracking)
        if use_tracking_speed:
            final_metric = (round(mean_trk_speed, 3), raw_trk_speeds)
        else:
            final_metric = (metric_val, raw_speeds)

        vectors_data["mean_speed"] = final_metric[0]
        vectors_data["raw_speeds"] = final_metric[1]
        vectors_data["focus_score"] = focus_score
        return flow_vis, tracking_dict, filtered_bboxes, mask, final_metric, vectors_data

    # ── Mode Both ────────────────────────────────────────────────────────────

    def _process_both_preprocessed(self,
                                   frame: np.ndarray,
                                   gray: np.ndarray,
                                   curr_bgr: np.ndarray,
                                   focus_score: Optional[float],
                                   focus_vis: Optional[np.ndarray],
                                   in_focus_mask: Optional[np.ndarray],
                                   p: Dict[str, Any],
                                   focus_map: Optional[np.ndarray] = None,
                                   frame_idx: Optional[int] = None) -> Any:
        base_canvas = focus_vis if focus_vis is not None else curr_bgr
        if (
            self._df_prev_bgr is None
            or self._df_prev_bgr.shape != curr_bgr.shape
            or self._df_prev_gray is None
            or self._df_prev_gray.shape != gray.shape
        ):
            self._df_prev_bgr = curr_bgr.copy()
            self._df_prev_gray = gray.copy()
            out_canvas = base_canvas.copy()
            drift_info = None
            if p.get("enable_drift_correction", False):
                d_res = self.denseflow_processor.drift_processor.process_vectors(
                    [], [], [], [], gray.shape[:2], p, frame_idx=frame_idx
                )
                drift_info = d_res.get("drift_info")
                if drift_info is not None:
                    DriftProcessor.draw_drift_overlay(
                        out_canvas, drift_info, arrow_scale=float(p.get("df_arrow_scale", 3.0)), params=p
                    )
            return out_canvas, {}, [], None, (0.0, 0.0, []), {
                "angles": [], "magnitudes": [], "mode": "both", "count": 0,
                "focus_score": focus_score, "drift_info": drift_info
            }

        fb_params = {
            'pyr_scale': p.get('df_pyr_scale', 0.5),
            'levels': p.get('df_levels', 2),
            'winsize': p.get('df_winsize', 15),
            'iterations': p.get('df_iterations', 3),
            'poly_n': p.get('df_poly_n', 5),
            'poly_sigma': p.get('df_poly_sigma', 1.2),
        }
        calib = float(p.get('df_calibration', 1.0))
        fps = float(p.get('df_fps', 30.0))

        mag, ang, flow_x, flow_y, _ = DenseFlowProcessor.compute_flow(
            self._df_prev_gray, gray, None, fb_params, self.denseflow_processor.opencl_available
        )
        self._df_prev_gray = gray.copy()
        self._df_prev_bgr = curr_bgr.copy()

        filter_stage = p.get("focus_filter_stage", "Pre-processing")
        is_post_focus = (
            p.get("enable_preprocessing", True)
            and p.get("enable_focus_picking", False)
            and p.get("focus_use_for_processing", False)
            and filter_stage == "Post-processing"
            and focus_map is not None
        )

        if in_focus_mask is not None and p.get("focus_use_for_processing", False) and not is_post_focus:
            mag[in_focus_mask == 0] = 0.0

        mask = ContourProcessor.binarize(gray, p)
        if in_focus_mask is not None and p.get("focus_use_for_processing", False) and not is_post_focus:
            mask = cv2.bitwise_and(mask, in_focus_mask)

        enable_contour = bool(p.get("enable_contour_detection", True))
        enable_tracking = bool(p.get("enable_tracking", False))
        filtered_boxes, filtered_cnts, detections_list, coverage_pct = ContourProcessor.detect_and_filter(mask, p)

        if is_post_focus and filtered_cnts:
            filtered_boxes, filtered_cnts, detections_list = PreProcessor.filter_contours_by_focus(
                focus_map, filtered_boxes, filtered_cnts, detections_list, p
            )
            mask = np.zeros_like(mask)
            if filtered_cnts:
                cv2.drawContours(mask, filtered_cnts, -1, 255, thickness=-1)
            total_px = mask.size
            covered = int(np.count_nonzero(mask))
            coverage_pct = round(100.0 * covered / total_px, 2) if total_px > 0 else 0.0
        else:
            total_px = mask.size
            covered = int(np.count_nonzero(mask))
            coverage_pct = round(100.0 * covered / total_px, 2) if total_px > 0 else 0.0

        if enable_contour:
            count = int(len(filtered_boxes))
        else:
            count = covered

        scale = calib * fps
        lower_mag = float(p.get('df_lower_mag_thresh', 1.5))
        upper_mag = float(p.get('df_upper_mag_thresh', 0.0))
        masked_mag = mag.copy()
        masked_mag[mask == 0] = 0.0
        masked_mag[masked_mag < lower_mag] = 0.0
        if upper_mag > 0.0:
            masked_mag[masked_mag > upper_mag] = 0.0

        enable_drift = bool(p.get("enable_drift_correction", False))
        drift_show_filtered = bool(p.get("drift_show_filtered", False))
        H, W = mag.shape[:2]
        ys, xs = np.nonzero(masked_mag > 0)
        drift_info = None

        mag_vis = mag.copy()
        if upper_mag > 0.0:
            mag_vis[mag_vis > upper_mag] = 0.0
        ang_vis = ang.copy()
        fx_vis = flow_x.copy()
        fy_vis = flow_y.copy()
        drift_pixel_mask = None

        if enable_drift:
            drift_res = self.denseflow_processor.drift_processor.process_vectors(
                xs, ys, flow_x[ys, xs], flow_y[ys, xs], (H, W), p, frame_idx=frame_idx
            )
            drift_info = drift_res["drift_info"]
            is_drift = drift_res["is_drift"]
            motile_mask = ~is_drift

            valid = drift_res["corr_mag"][motile_mask]
            valid_angs = drift_res["corr_ang"][motile_mask]

            masked_mag[ys[is_drift], xs[is_drift]] = 0.0
            masked_mag[ys[motile_mask], xs[motile_mask]] = valid

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
        raw_speeds_um_s = (valid * scale).tolist() if p.get('collect_raw_speeds', False) else []

        vis_mode = p.get('both_vis_mode', 'Surface')
        out_vis = base_canvas.copy()

        if vis_mode == 'Surface':
            blob_alpha = float(p.get("blob_alpha", 0.5))
            alpha = max(0.0, min(1.0, blob_alpha))
            red_layer = np.zeros_like(out_vis)
            red_layer[:] = (0, 0, 255)
            blended = cv2.addWeighted(out_vis, 1.0 - alpha, red_layer, alpha, 0)
            out_vis = np.where(mask[:, :, None] > 0, blended, out_vis)
        elif vis_mode == 'Speed':
            full_binary = np.full_like(mask, 255)
            if enable_drift and not drift_show_filtered:
                full_binary[ys[is_drift], xs[is_drift]] = 0
            out_vis = DenseFlowProcessor.visualize_flow(
                mag_vis, ang_vis, fx_vis, fy_vis, full_binary, base_canvas, p,
                self.denseflow_processor.opencl_available,
                raw_flow_x=flow_x, raw_flow_y=flow_y, drift_mask=drift_pixel_mask
            )
        elif vis_mode == 'Selection':
            sel_mask = mask.copy()
            if enable_drift and not drift_show_filtered:
                sel_mask[ys[is_drift], xs[is_drift]] = 0
            out_vis = DenseFlowProcessor.visualize_flow(
                mag_vis, ang_vis, fx_vis, fy_vis, sel_mask, base_canvas, p,
                self.denseflow_processor.opencl_available,
                raw_flow_x=flow_x, raw_flow_y=flow_y, drift_mask=drift_pixel_mask
            )

        if enable_drift and not enable_tracking:
            DriftProcessor.draw_drift_overlay(
                out_vis, drift_info, arrow_scale=float(p.get('df_arrow_scale', 2.0)), params=p
            )

        if enable_tracking:
            self.tracker.check_reset_trigger(
                p.get('reset_trigger', 0),
                p.get('distance_function', 'euclidean'),
                p.get('distance_threshold', 30),
                hit_counter_max=int(p.get('track_max_lost', 15))
            )
            out_vis, tracking_dict, trk_vectors, mean_trk_speed, raw_trk_speeds = self.tracker.update(
                detections_list, out_vis, p, mode_label="both_contour", frame_idx=frame_idx
            )
            motility_pct = trk_vectors.get("motility_pct", 0.0)
            mobile_cnt = trk_vectors.get("mobile_tracks", 0)
            total_cnt = trk_vectors.get("total_tracks", 0)
            if trk_vectors.get("drift_info") is not None:
                drift_info = trk_vectors["drift_info"]
        else:
            tracking_dict = {}
            mean_trk_speed, raw_trk_speeds = 0.0, []
            if enable_contour and filtered_cnts:
                moving_cnts = 0
                if len(valid) > 0:
                    for c in filtered_cnts:
                        x, y, w, h = cv2.boundingRect(c)
                        if np.any(masked_mag[y:y+h, x:x+w] > 0):
                            moving_cnts += 1
                total_cnt = len(filtered_cnts)
                mobile_cnt = moving_cnts
                motility_pct = round(mobile_cnt / total_cnt * 100.0, 2) if total_cnt > 0 else 0.0
            else:
                total_cnt = covered
                mobile_cnt = int(len(valid))
                motility_pct = round(mobile_cnt / total_cnt * 100.0, 2) if total_cnt > 0 else 0.0

        vectors_data = {
            "angles": valid_angs.tolist() if isinstance(valid_angs, np.ndarray) else list(valid_angs),
            "magnitudes": (valid * scale).tolist() if isinstance(valid, np.ndarray) else list(valid * scale),
            "mode": "both",
            "count": count,
            "motility_pct": motility_pct,
            "mobile_count": mobile_cnt,
            "total_count": total_cnt,
            "focus_score": focus_score,
            "drift_info": drift_info,
        }

        use_tracking_speed = bool(p.get("use_tracking_speed", False)) and bool(enable_tracking)
        if use_tracking_speed:
            final_speed = round(mean_trk_speed, 3)
            final_raw = raw_trk_speeds
        else:
            final_speed = mean_speed
            final_raw = raw_speeds_um_s

        vectors_data["mean_speed"] = final_speed
        vectors_data["raw_speeds"] = final_raw

        return out_vis, tracking_dict, [], None, (coverage_pct, final_speed, final_raw), vectors_data

    # ── Stateless Compatibility ──────────────────────────────────────────────

    @staticmethod
    def process_frame_stateless(frame: np.ndarray, params: dict):
        return ContourProcessor.process_stateless(frame, params)
