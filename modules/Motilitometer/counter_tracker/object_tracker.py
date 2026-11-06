"""
Object Tracker Engine for Motilitometer Counter Tracker.

Encapsulates:
- Norfair tracking engine
- Multi-object trajectory accumulation and lifecycle management
- Metric calculation (distance traveled, instantaneous/mean speed)
- Drift correction integration (identifying non-motile drift tracks & vector subtraction)
- Trail rendering and annotation overlays
- Velocity vector aggregation for polar / vector outputs
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from norfair import Tracker, Detection

from .drift_processor import DriftProcessor


class ObjectTracker:
    """Manages centroid tracking, trajectories, drift correction, and on-frame trail overlays."""

    def __init__(self,
                 distance_function: str = "euclidean",
                 distance_threshold: float = 30.0,
                 hit_counter_max: int = 15) -> None:
        self.distance_function = distance_function
        self.distance_threshold = distance_threshold
        self.hit_counter_max = hit_counter_max
        self.tracking: Dict[int, Any] = {}
        self._last_reset_trigger: int = 0
        self.drift_processor = DriftProcessor()
        self._init_tracker()

    def _init_tracker(self) -> None:
        kwargs = {
            "distance_function": self.distance_function,
            "distance_threshold": self.distance_threshold,
        }
        if self.hit_counter_max is not None and self.hit_counter_max > 0:
            kwargs["hit_counter_max"] = int(self.hit_counter_max)
        try:
            self.tracker = Tracker(**kwargs)
        except TypeError:
            self.tracker = Tracker(
                distance_function=self.distance_function,
                distance_threshold=self.distance_threshold,
            )

    @staticmethod
    def _get_random_color() -> Tuple[int, int, int]:
        return tuple(random.randint(50, 255) for _ in range(3))

    def reset(self) -> None:
        """Resets all tracking states, trajectories, and drift state."""
        self._init_tracker()
        self.tracking.clear()
        self.drift_processor.reset()

    def check_reset_trigger(self,
                            trigger_val: int,
                            dist_func: str,
                            dist_thresh: float,
                            hit_counter_max: int = 15) -> None:
        """Resets tracker if reset trigger was incremented, or parameters changed."""
        needs_reset = (
            trigger_val > self._last_reset_trigger
            or dist_func != self.distance_function
            or dist_thresh != self.distance_threshold
            or (hit_counter_max != self.hit_counter_max and hit_counter_max > 0)
        )
        if needs_reset:
            self.distance_function = dist_func
            self.distance_threshold = dist_thresh
            self.hit_counter_max = hit_counter_max
            self.reset()
            self._last_reset_trigger = trigger_val

    def update(self,
               detections_list: List[Tuple[int, int]],
               canvas: Optional[np.ndarray],
               params: dict,
               mode_label: str = "contour",
               frame_idx: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict[int, Any], dict, float, List[float]]:
        """
        Updates tracking with current detections, applies optional drift correction,
        and renders trails/vectors on canvas.

        Returns:
            Tuple of (annotated_canvas, tracking_dict, vectors_data, mean_tracking_speed, raw_tracking_speeds)
        """
        enable_tracking = params.get("enable_tracking", True)
        if not enable_tracking:
            self.tracking.clear()
            drift_info = None
            if bool(params.get("enable_drift_correction", False)):
                frame_shape = (
                    (canvas.shape[0], canvas.shape[1])
                    if canvas is not None
                    else tuple(params.get("_frame_shape", (1080, 1920)))
                )
                drift_res = self.drift_processor.process_vectors([], [], [], [], frame_shape, params, frame_idx=frame_idx)
                drift_info = drift_res["drift_info"]
                if canvas is not None:
                    DriftProcessor.draw_drift_overlay(
                        canvas, drift_info, arrow_scale=float(params.get("arrow_scale", 3.0)), params=params
                    )
            return canvas, {}, {
                "angles": [],
                "magnitudes": [],
                "mode": mode_label,
                "count": 0,
                "motility_pct": None,
                "mobile_tracks": 0,
                "total_tracks": 0,
                "drift_info": drift_info,
            }, 0.0, []

        # Update hit_counter_max dynamically on tracker if supported
        track_max_lost = int(params.get("track_max_lost", self.hit_counter_max or 15))
        if track_max_lost > 0 and hasattr(self.tracker, "hit_counter_max"):
            self.tracker.hit_counter_max = track_max_lost

        norfair_detections = [Detection(np.array(center)) for center in detections_list]
        tracked_objects = self.tracker.update(norfair_detections)

        # Enforce max track age if configured (> 0)
        track_max_age = int(params.get("track_max_age", 0))
        if track_max_age > 0:
            if hasattr(self.tracker, "tracked_objects"):
                self.tracker.tracked_objects = [
                    o for o in self.tracker.tracked_objects if o.age < track_max_age
                ]
            tracked_objects = [o for o in tracked_objects if o.age < track_max_age]

        # Prune dead tracks from internal history so they never accumulate indefinitely
        active_ids = {obj.id for obj in tracked_objects}
        dead_ids = [tid for tid in list(self.tracking.keys()) if tid not in active_ids]
        for tid in dead_ids:
            del self.tracking[tid]

        trail_length = int(params.get("trail_length", 30))
        show_trail = bool(params.get("show_trail", False))
        show_arrow = bool(params.get("show_arrow", False))
        arrow_scale = float(params.get("arrow_scale", 3.0))
        vector_mode = params.get("vector_mode", "Last 2 Points")
        show_id = bool(params.get("show_id", False))
        show_age = bool(params.get("show_age", False))
        show_distance = bool(params.get("show_distance", False))
        show_speed = bool(params.get("show_speed", False))

        enable_drift = bool(params.get("enable_drift_correction", False))
        drift_show_filtered = bool(params.get("drift_show_filtered", False))
        drift_show_correction = bool(params.get("drift_show_correction", False))
        drift_min_corr_speed = float(params.get("drift_min_corr_speed", 0.0))

        speed_window = max(1, int(params.get("speed_window", 5)))
        calib = float(params.get("calibration", params.get("df_calibration", 1.0)))
        fps = float(params.get("fps", params.get("df_fps", 30.0)))
        scale = calib * fps if calib > 0 and fps > 0 else 1.0
        min_speed_thresh = float(params.get("track_min_speed", params.get("df_track_min_speed", 0.0)))
        max_speed_thresh = float(params.get("track_max_speed", params.get("df_track_max_speed", 0.0)))

        if canvas is not None:
            frame_shape = (canvas.shape[0], canvas.shape[1])
        else:
            frame_shape = tuple(params.get("_frame_shape", (1080, 1920)))

        # First pass: update trail history and compute raw per-frame velocity vectors
        track_records: List[Dict[str, Any]] = []
        valid_vec_indices: List[int] = []
        vec_xs: List[float] = []
        vec_ys: List[float] = []
        vec_vxs: List[float] = []
        vec_vys: List[float] = []

        for idx, obj in enumerate(tracked_objects):
            cx, cy = map(int, obj.estimate[0])

            if obj.id not in self.tracking:
                self.tracking[obj.id] = {
                    "color": self._get_random_color(),
                    "points": [],
                    "distance": 0.0,
                    "age": obj.age,
                    "rolling_speed": 0.0,
                    "is_drift": False,
                }

            self.tracking[obj.id]["points"].append((cx, cy))
            self.tracking[obj.id]["age"] = obj.age

            trail = self.tracking[obj.id]["points"]
            pts_window = trail[-(speed_window + 1):]

            # Raw visual vector (Last 2 Points or Trail Start to End)
            raw_v_dx = 0.0
            raw_v_dy = 0.0
            n_span = 1
            if len(trail) >= 2:
                if vector_mode in ("Trail Start to End", "Start to End of Trail", "Start to End"):
                    if trail_length > 0 and len(trail) > trail_length:
                        p_start = trail[-trail_length]
                        n_span = max(1, trail_length - 1)
                    else:
                        p_start = trail[0]
                        n_span = max(1, len(trail) - 1)
                    raw_v_dx = float(cx - p_start[0])
                    raw_v_dy = float(cy - p_start[1])
                else:
                    raw_v_dx = float(cx - trail[-2][0])
                    raw_v_dy = float(cy - trail[-2][1])
                    n_span = 1

            # Per-frame velocity vector over rolling window for drift clustering
            if len(pts_window) >= 2:
                n_win_steps = len(pts_window) - 1
                mean_vx = float(pts_window[-1][0] - pts_window[0][0]) / n_win_steps
                mean_vy = float(pts_window[-1][1] - pts_window[0][1]) / n_win_steps
                raw_step_dists = [
                    float(np.hypot(pts_window[k][0] - pts_window[k - 1][0],
                                   pts_window[k][1] - pts_window[k - 1][1]))
                    for k in range(1, len(pts_window))
                ]
                raw_win_speed_um_s = (float(np.mean(raw_step_dists)) if raw_step_dists else 0.0) * scale
                reusing_drift = (
                    bool(params.get("drift_reuse_preprocessed", False))
                    and bool(
                        params.get("_preprocessed_drift_map")
                        or params.get("_preprocessed_drift_path")
                        or params.get("_video_path")
                    )
                )
                in_speed_band = True
                if not reusing_drift:
                    if min_speed_thresh > 0.0 and raw_win_speed_um_s < min_speed_thresh:
                        in_speed_band = False
                    if max_speed_thresh > 0.0 and raw_win_speed_um_s > max_speed_thresh:
                        in_speed_band = False
                if np.hypot(mean_vx, mean_vy) > 0.0 and in_speed_band:
                    valid_vec_indices.append(idx)
                    vec_xs.append(float(cx))
                    vec_ys.append(float(cy))
                    vec_vxs.append(mean_vx)
                    vec_vys.append(mean_vy)

            track_records.append({
                "obj": obj,
                "cx": cx,
                "cy": cy,
                "trail": trail,
                "pts_window": pts_window,
                "raw_v_dx": raw_v_dx,
                "raw_v_dy": raw_v_dy,
                "n_span": n_span,
                "is_drift": False,
                "drift_dx": 0.0,
                "drift_dy": 0.0,
            })

        # Second pass: run DriftProcessor on active track velocity vectors
        drift_res = self.drift_processor.process_vectors(
            vec_xs, vec_ys, vec_vxs, vec_vys, frame_shape, params, frame_idx=frame_idx
        )
        drift_info = drift_res["drift_info"]

        if enable_drift and len(valid_vec_indices) > 0:
            is_drift_arr = drift_res["is_drift"]
            drift_vx_arr = drift_res["drift_vx"]
            drift_vy_arr = drift_res["drift_vy"]
            for k, rec_idx in enumerate(valid_vec_indices):
                track_records[rec_idx]["is_drift"] = bool(is_drift_arr[k])
                track_records[rec_idx]["drift_dx"] = float(drift_vx_arr[k])
                track_records[rec_idx]["drift_dy"] = float(drift_vy_arr[k])

        vec_angles: List[float] = []
        vec_mags: List[float] = []
        active_speeds: List[float] = []

        # Third pass: compute corrected speeds, filter drift tracks, and draw overlays
        for rec in track_records:
            obj = rec["obj"]
            cx, cy = rec["cx"], rec["cy"]
            trail = rec["trail"]
            pts_window = rec["pts_window"]
            raw_v_dx, raw_v_dy = rec["raw_v_dx"], rec["raw_v_dy"]
            n_span = rec["n_span"]
            is_drift_trk = rec["is_drift"]
            d_dx, d_dy = rec["drift_dx"], rec["drift_dy"]

            self.tracking[obj.id]["is_drift"] = is_drift_trk

            if len(trail) >= 2:
                p1 = trail[-2]
                p2 = trail[-1]
                step_dx = (p2[0] - p1[0]) - (d_dx if (enable_drift and not is_drift_trk) else 0.0)
                step_dy = (p2[1] - p1[1]) - (d_dy if (enable_drift and not is_drift_trk) else 0.0)
                dist = float(np.hypot(step_dx, step_dy))
                self.tracking[obj.id]["distance"] += dist

            if len(pts_window) >= 2:
                if enable_drift and not is_drift_trk:
                    step_dists = [
                        float(np.hypot((pts_window[k][0] - pts_window[k - 1][0]) - d_dx,
                                       (pts_window[k][1] - pts_window[k - 1][1]) - d_dy))
                        for k in range(1, len(pts_window))
                    ]
                else:
                    step_dists = [
                        float(np.hypot(pts_window[k][0] - pts_window[k - 1][0],
                                       pts_window[k][1] - pts_window[k - 1][1]))
                        for k in range(1, len(pts_window))
                    ]
                mean_disp_px = float(np.mean(step_dists)) if step_dists else 0.0
                inst_speed_um_s = mean_disp_px * scale
                self.tracking[obj.id]["rolling_speed"] = inst_speed_um_s
            else:
                inst_speed_um_s = 0.0
                self.tracking[obj.id]["rolling_speed"] = 0.0

            # Post-correction minimum speed filter: treat residual slow vectors as drift/non-motile
            if enable_drift and not is_drift_trk and drift_min_corr_speed > 0.0 and inst_speed_um_s < drift_min_corr_speed:
                is_drift_trk = True
                self.tracking[obj.id]["is_drift"] = True

            # Handle tracks categorized as drift
            if enable_drift and is_drift_trk:
                if canvas is not None and drift_show_filtered:
                    drift_color = (0, 140, 255)  # Distinct orange in BGR for drift tracks
                    if show_trail:
                        trail_pts = trail[-trail_length:]
                        if len(trail_pts) >= 2:
                            for i in range(1, len(trail_pts)):
                                cv2.line(canvas, trail_pts[i - 1], trail_pts[i], drift_color, 1, cv2.LINE_AA)
                    raw_mag = float(np.hypot(raw_v_dx, raw_v_dy))
                    if (show_arrow or drift_show_filtered) and raw_mag > 0.0:
                        ang_rad = float(np.arctan2(raw_v_dy, raw_v_dx))
                        arrow_len = max(10.0, (raw_mag / n_span) * arrow_scale)
                        end_x = int(round(cx + arrow_len * np.cos(ang_rad)))
                        end_y = int(round(cy + arrow_len * np.sin(ang_rad)))
                        cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), drift_color, 1, cv2.LINE_AA, tipLength=0.3)
                    if show_id or show_speed:
                        cv2.putText(canvas, "DRIFT", (cx + 4, cy + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, drift_color, 1, cv2.LINE_AA)
                # Ignore drift track from mean speed and motility count
                continue

            # If below minimum speed threshold or above maximum speed threshold, hide from canvas and exclude from export
            if min_speed_thresh > 0.0 and inst_speed_um_s < min_speed_thresh:
                continue
            if max_speed_thresh > 0.0 and inst_speed_um_s > max_speed_thresh:
                continue

            active_speeds.append(inst_speed_um_s)

            # Corrected vector calculation
            if enable_drift:
                v_dx = raw_v_dx - d_dx * n_span
                v_dy = raw_v_dy - d_dy * n_span
            else:
                v_dx = raw_v_dx
                v_dy = raw_v_dy
            v_mag = float(np.hypot(v_dx, v_dy)) / n_span if len(trail) >= 2 else 0.0
            raw_v_mag = float(np.hypot(raw_v_dx, raw_v_dy)) / n_span if len(trail) >= 2 else 0.0

            if v_mag > 0.0:
                ang_rad = float(np.arctan2(v_dy, v_dx))
                if ang_rad < 0:
                    ang_rad += 2 * np.pi
                vec_angles.append(ang_rad)
                vec_mags.append(v_mag * scale)

            if canvas is not None and show_trail:
                trail_pts = trail[-trail_length:]
                if len(trail_pts) >= 2:
                    for i in range(1, len(trail_pts)):
                        cv2.line(canvas, trail_pts[i - 1], trail_pts[i],
                                 self.tracking[obj.id]["color"], 2, cv2.LINE_AA)

            if canvas is not None:
                # Draw pre-correction vector if Show Pre/Post Correction is enabled
                if enable_drift and drift_show_correction and raw_v_mag > 0.0:
                    raw_ang = float(np.arctan2(raw_v_dy, raw_v_dx))
                    raw_len = max(10.0, raw_v_mag * arrow_scale)
                    r_end_x = int(round(cx + raw_len * np.cos(raw_ang)))
                    r_end_y = int(round(cy + raw_len * np.sin(raw_ang)))
                    cv2.arrowedLine(canvas, (cx, cy), (r_end_x, r_end_y), (0, 200, 255), 1, cv2.LINE_AA, tipLength=0.3)

                # Draw post-correction (or standard) vector arrow
                if (show_arrow or (enable_drift and drift_show_correction)) and v_mag > 0.0:
                    ang_rad = float(np.arctan2(v_dy, v_dx))
                    arrow_len = max(10.0, v_mag * arrow_scale)
                    end_x = int(round(cx + arrow_len * np.cos(ang_rad)))
                    end_y = int(round(cy + arrow_len * np.sin(ang_rad)))
                    cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

            text_lines = []
            if show_id:
                text_lines.append(f"ID:{obj.id}")
            if show_age:
                text_lines.append(f"Age:{obj.age}")
            if show_distance:
                text_lines.append(f"Dist:{self.tracking[obj.id]['distance']:.1f}")
            if show_speed:
                text_lines.append(f"Spd:{inst_speed_um_s:.1f}")

            if canvas is not None:
                for i, line in enumerate(text_lines):
                    cv2.putText(canvas, line, (cx + 4, cy + 12 + i * 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw global/local drift vector overlay on canvas
        if canvas is not None and enable_drift:
            DriftProcessor.draw_drift_overlay(
                canvas, drift_info, arrow_scale=arrow_scale, params=params
            )

        mean_tracking_speed = float(np.mean(active_speeds)) if active_speeds else 0.0
        raw_tracking_speeds = active_speeds

        total_tracks = len(tracked_objects)
        if min_speed_thresh > 0.0:
            mobile_tracks = len(active_speeds)
        else:
            mobile_tracks = len([s for s in active_speeds if s > 0.0])
        motility_pct = round(mobile_tracks / total_tracks * 100.0, 2) if total_tracks > 0 else 0.0

        vectors_data = {
            "angles": vec_angles,
            "magnitudes": vec_mags,
            "mode": mode_label,
            "count": int(len(vec_angles)),
            "motility_pct": motility_pct,
            "mobile_tracks": mobile_tracks,
            "total_tracks": total_tracks,
            "drift_info": drift_info,
        }

        exported_tracking = {
            tid: tdata for tid, tdata in self.tracking.items()
            if (not (enable_drift and tdata.get("is_drift", False)) or drift_show_filtered)
            and (min_speed_thresh <= 0.0 or tdata.get("rolling_speed", 0.0) >= min_speed_thresh)
            and (max_speed_thresh <= 0.0 or tdata.get("rolling_speed", 0.0) <= max_speed_thresh)
        }

        return canvas, exported_tracking, vectors_data, mean_tracking_speed, raw_tracking_speeds
