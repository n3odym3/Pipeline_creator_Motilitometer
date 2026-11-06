import csv
import json
import os
import threading
import time
import uuid as _uuid
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Dict, Any, List

import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.window_base import WindowBase
from core.input_output_types import IOTypes

from .counter_tracker import (
    MotilitometerProcessor,
    _open_batch_video_capture,
    batch_worker_init,
    batch_prepare_video_params,
    batch_worker_process_chunk,
    batch_worker_denseflow_chunk,
    batch_worker_both_chunk,
)
from .object_tracker import ObjectTracker
from .drift_processor import DriftProcessor
from .angle_export import (
    PI_DEFAULT_LIGHT_POS,
    PI_DEFAULT_PRESET,
    PI_LIGHT_POSITIONS,
    PI_PRESETS,
    build_arrays as build_angle_arrays,
    frame_phototaxis_index,
    hist_phototaxis_series,
    normalize_pi_light_pos,
    normalize_pi_preset,
    pi_light_arrows,
    pi_preset_ranges,
)
from .raw_export import build_raw_arrays, save_raw_npz, raw_path_for

def _fmt_time(s: float) -> str:
    return f"{int(s)}s" if s < 60 else f"{int(s // 60)}m {int(s % 60)}s"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


class counter_tracker_win(WindowBase):
    """UI window for the combined Motilitometer processing module.
    Provides controls for the underlying MotilitometerProcessor.
    """

    # 1. Window Configuration
    label: str = "Motilitometer Tracker"
    win_width: int = display_scaling.scale(600)
    win_height: int = display_scaling.scale(700)
    description: str = (
        "Combined Motilitometer tracking, contour detection, dense flow, and phototaxis processor."
    )

    # 2. Input/Output definitions
    accepted_input_types: list[type] = [IOTypes.FRAME, IOTypes.TRIGGER, IOTypes.FILE_PATH]
    outputs: dict[str, type] = {
        "Frame": IOTypes.FRAME,
        "CMD": IOTypes.CMD_DICT,
        "Tracking": IOTypes.TRACKING,
        "Contours": IOTypes.CONTOURS,
        "Result": IOTypes.SAMPLE,
        "Count": IOTypes.SAMPLE,
        "Motility": IOTypes.SAMPLE,
        "Phototaxis": IOTypes.SAMPLE,
        "Raw Speeds": IOTypes.SAMPLE,
        "Vectors": IOTypes.VECTORS,
        "Trigger": IOTypes.TRIGGER,
    }

    # Persistent field defaults matching settings.json fallbacks
    x_axis_mode: str = "Fixed ΔT"
    delta_t: float = 1.0 / 30.0
    auto_export_csv: bool = True
    auto_export_raw: bool = False
    calibration: float = 0.7575
    fps: float = 30.0

    enable_preprocessing: bool = True
    monochrome_mode: str = "Grayscale"
    invert_image: bool = False
    prep_blur: int = 0
    gamma: float = 1.0
    crop_type: str = "None"
    crop_radius: int = 150
    crop_width: int = 300
    crop_height: int = 300
    show_crop_outline: bool = True
    crop_use_abs_center: bool = False
    crop_center_x: int = 0
    crop_center_y: int = 0
    crop_offset_x: int = 0
    crop_offset_y: int = 0
    remove_static: bool = False
    similarity_threshold: int = 15

    enable_focus_picking: bool = False
    focus_metric: str = "Tenengrad (Sobel)"
    focus_window_size: int = 15
    focus_output_map: bool = True
    focus_vis_mode: str = "Heatmap (Turbo)"
    focus_threshold: int = 40
    focus_alpha: float = 0.6
    focus_isolate_selection: bool = False
    focus_use_for_processing: bool = False
    focus_filter_stage: str = "Pre-processing"

    blob_alpha: float = 0.5
    enable_watershed: bool = False
    ws_peak_thresh: int = 40
    ws_show_lines: bool = True

    enable_drift_correction: bool = False
    drift_angle_tol: float = 15.0
    drift_mag_tol: float = 25.0
    drift_min_corr_speed: float = 0.0
    drift_use_hint: bool = False
    drift_hint_preset: str = "Down (90°)"
    drift_hint_angle: float = 90.0
    drift_hint_cone: float = 45.0
    drift_grid_mode: str = "Full (1x1)"
    drift_lock_live: bool = False
    drift_use_ranges: bool = False
    drift_active_ranges: str = "[0-500],[1000-1500]"
    drift_show_vector: bool = True
    drift_show_filtered: bool = False
    drift_show_correction: bool = False
    drift_show_fake_drifters: bool = False
    drift_reuse_preprocessed: bool = False
    drift_only_export: bool = False

    emit_phototaxis_series: bool = False
    pi_light_pos: str = PI_DEFAULT_LIGHT_POS
    pi_preset: str = PI_DEFAULT_PRESET
    pi_up_min: float = 180.0
    pi_up_max: float = 360.0
    pi_down_min: float = 0.0
    pi_down_max: float = 180.0
    pi_weight_speed: bool = True

    # 3. Persistent fields saved into pipeline JSON
    persistent_fields: list[str] = [
        "label", "x_axis_mode", "delta_t", "auto_export_csv", "auto_export_raw",
        "calibration", "fps",
        "enable_preprocessing", "monochrome_mode", "invert_image", "prep_blur",
        "gamma", "crop_type", "crop_radius", "crop_width", "crop_height", "show_crop_outline",
        "crop_use_abs_center", "crop_center_x", "crop_center_y", "crop_offset_x", "crop_offset_y",
        "remove_static", "similarity_threshold",
        "enable_focus_picking", "focus_metric", "focus_window_size", "focus_output_map", "focus_vis_mode",
        "focus_threshold", "focus_alpha", "focus_isolate_selection", "focus_use_for_processing",
        "focus_filter_stage",
        "blob_alpha",
        "enable_watershed", "ws_peak_thresh", "ws_show_lines",
        "enable_drift_correction", "drift_angle_tol", "drift_mag_tol", "drift_min_corr_speed",
        "drift_use_hint", "drift_hint_preset", "drift_hint_angle", "drift_hint_cone",
        "drift_grid_mode", "drift_lock_live", "drift_use_ranges", "drift_active_ranges",
        "drift_show_vector", "drift_show_filtered", "drift_show_correction", "drift_show_fake_drifters",
        "drift_reuse_preprocessed", "drift_only_export",
        "emit_phototaxis_series", "pi_light_pos", "pi_preset",
        "pi_up_min", "pi_up_max", "pi_down_min", "pi_down_max", "pi_weight_speed",
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_thread_running: bool = False
        self.x_axis_mode = kwargs.get("x_axis_mode", "Fixed ΔT")
        self.delta_t = kwargs.get("delta_t", 1.0 / 30.0)
        self.auto_export_csv = kwargs.get("auto_export_csv", True)
        self.auto_export_raw = kwargs.get("auto_export_raw", False)
        self.samples_buffers = {}
        self._buffers_lock = threading.Lock()
        self._skip_first_live_frame = True
        self._last_input_data: Optional[Any] = None

        # Load settings from JSON
        settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
        try:
            with open(settings_path, "r") as f:
                self.settings = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load settings.json: {e}")
            self.settings = {}

        # Initialise processor with default settings
        s_prep = self.settings.get('preprocessing', {})
        s_crop = self.settings.get('crop', {})
        s_mono = self.settings.get('monochrome', {})
        s_stat = self.settings.get('static_removal', {})
        s_bin  = self.settings.get('binarize', {})
        s_cnt  = self.settings.get('contour', {})
        s_trk  = self.settings.get('tracker', {})
        s_glb  = self.settings.get('global', {})
        s_df   = self.settings.get('denseflow', {})
        s_drf  = self.settings.get('drift_correction', {})
        s_pi   = self.settings.get('phototaxis', {})

        self.enable_preprocessing = bool(kwargs.get("enable_preprocessing", s_prep.get("enable_preprocessing", True)))
        self.monochrome_mode = str(kwargs.get("monochrome_mode", s_prep.get("channel", "Grayscale")))
        self.invert_image = bool(kwargs.get("invert_image", s_prep.get("invert_image", False)))
        self.prep_blur = int(kwargs.get("prep_blur", s_prep.get("prep_blur", 0)))
        self.gamma = float(kwargs.get("gamma", s_prep.get("gamma", s_mono.get("gamma", 1.0))))
        self.crop_type = str(kwargs.get("crop_type", s_prep.get("crop_type", s_crop.get("crop_type", "Circular" if s_crop.get("circular_crop", False) else "None"))))
        self.crop_radius = int(kwargs.get("crop_radius", s_prep.get("crop_radius", s_crop.get("crop_radius", 150))))
        self.crop_width = int(kwargs.get("crop_width", s_prep.get("crop_width", s_crop.get("crop_width", 300))))
        self.crop_height = int(kwargs.get("crop_height", s_prep.get("crop_height", s_crop.get("crop_height", 300))))
        self.show_crop_outline = bool(kwargs.get("show_crop_outline", s_prep.get("show_crop_outline", s_crop.get("show_crop_outline", True))))
        self.crop_use_abs_center = bool(kwargs.get("crop_use_abs_center", kwargs.get("use_absolute_center", s_prep.get("use_absolute_center", s_crop.get("use_absolute_center", False)))))
        self.crop_center_x = int(kwargs.get("crop_center_x", kwargs.get("center_x", s_prep.get("center_x", s_crop.get("center_x", 0)))))
        self.crop_center_y = int(kwargs.get("crop_center_y", kwargs.get("center_y", s_prep.get("center_y", s_crop.get("center_y", 0)))))
        self.crop_offset_x = int(kwargs.get("crop_offset_x", kwargs.get("offset_x", s_prep.get("offset_x", s_crop.get("offset_x", 0)))))
        self.crop_offset_y = int(kwargs.get("crop_offset_y", kwargs.get("offset_y", s_prep.get("offset_y", s_crop.get("offset_y", 0)))))
        self.remove_static = bool(kwargs.get("remove_static", s_prep.get("remove_static", s_stat.get("remove_static", False))))
        self.similarity_threshold = int(kwargs.get("similarity_threshold", s_prep.get("similarity_threshold", s_stat.get("similarity_threshold", 15))))
        self.enable_focus_picking = bool(kwargs.get("enable_focus_picking", s_prep.get("enable_focus_picking", False)))
        self.focus_metric = str(kwargs.get("focus_metric", s_prep.get("focus_metric", "Tenengrad (Sobel)")))
        self.focus_window_size = int(kwargs.get("focus_window_size", s_prep.get("focus_window_size", 15)))
        self.focus_output_map = bool(kwargs.get("focus_output_map", s_prep.get("focus_output_map", True)))
        self.focus_vis_mode = str(kwargs.get("focus_vis_mode", s_prep.get("focus_vis_mode", "Heatmap (Turbo)")))
        self.focus_threshold = int(kwargs.get("focus_threshold", s_prep.get("focus_threshold", 40)))
        self.focus_alpha = float(kwargs.get("focus_alpha", s_prep.get("focus_alpha", 0.6)))
        self.focus_isolate_selection = bool(kwargs.get("focus_isolate_selection", s_prep.get("focus_isolate_selection", False)))
        self.focus_use_for_processing = bool(kwargs.get("focus_use_for_processing", s_prep.get("focus_use_for_processing", False)))
        self.focus_filter_stage = str(kwargs.get("focus_filter_stage", s_prep.get("focus_filter_stage", "Pre-processing")))
        self.blob_alpha = float(kwargs.get("blob_alpha", s_bin.get("blob_alpha", 0.5)))
        self.enable_watershed = bool(kwargs.get("enable_watershed", s_cnt.get("enable_watershed", False)))
        self.ws_peak_thresh = int(kwargs.get("ws_peak_thresh", s_cnt.get("ws_peak_thresh", 40)))
        self.ws_show_lines = bool(kwargs.get("ws_show_lines", s_cnt.get("ws_show_lines", True)))
        self.calibration = float(kwargs.get("calibration", kwargs.get("df_calibration", s_df.get("calibration", 0.7575))))
        self.fps = float(kwargs.get("fps", kwargs.get("df_fps", s_df.get("fps", 30.0))))
        self.enable_drift_correction = bool(kwargs.get("enable_drift_correction", s_drf.get("enable_drift_correction", False)))
        self.drift_angle_tol = float(kwargs.get("drift_angle_tol", s_drf.get("drift_angle_tol", 15.0)))
        self.drift_mag_tol = float(kwargs.get("drift_mag_tol", s_drf.get("drift_mag_tol", 25.0)))
        self.drift_min_corr_speed = float(kwargs.get("drift_min_corr_speed", s_drf.get("drift_min_corr_speed", 0.0)))
        self.drift_use_hint = bool(kwargs.get("drift_use_hint", s_drf.get("drift_use_hint", False)))
        self.drift_hint_preset = str(kwargs.get("drift_hint_preset", s_drf.get("drift_hint_preset", "Down (90°)")))
        self.drift_hint_angle = float(kwargs.get("drift_hint_angle", s_drf.get("drift_hint_angle", 90.0)))
        self.drift_hint_cone = float(kwargs.get("drift_hint_cone", s_drf.get("drift_hint_cone", 45.0)))
        self.drift_grid_mode = str(kwargs.get("drift_grid_mode", s_drf.get("drift_grid_mode", "Full (1x1)")))
        self.drift_lock_live = bool(kwargs.get("drift_lock_live", s_drf.get("drift_lock_live", False)))
        self.drift_use_ranges = bool(kwargs.get("drift_use_ranges", s_drf.get("drift_use_ranges", False)))
        self.drift_active_ranges = str(kwargs.get("drift_active_ranges", s_drf.get("drift_active_ranges", "[0-500],[1000-1500]")))
        self.drift_show_vector = bool(kwargs.get("drift_show_vector", s_drf.get("drift_show_vector", True)))
        self.drift_show_filtered = bool(kwargs.get("drift_show_filtered", s_drf.get("drift_show_filtered", False)))
        self.drift_show_correction = bool(kwargs.get("drift_show_correction", s_drf.get("drift_show_correction", False)))
        self.drift_show_fake_drifters = bool(kwargs.get("drift_show_fake_drifters", s_drf.get("drift_show_fake_drifters", False)))
        self.drift_reuse_preprocessed = bool(kwargs.get("drift_reuse_preprocessed", s_drf.get("drift_reuse_preprocessed", False)))
        self.drift_only_export = bool(kwargs.get("drift_only_export", s_drf.get("drift_only_export", False)))
        self.emit_phototaxis_series = bool(kwargs.get("emit_phototaxis_series", s_pi.get("emit_phototaxis_series", False)))
        self.pi_light_pos = normalize_pi_light_pos(kwargs.get("pi_light_pos", s_pi.get("pi_light_pos", PI_DEFAULT_LIGHT_POS)))
        self.pi_preset = normalize_pi_preset(kwargs.get("pi_preset", s_pi.get("pi_preset", PI_DEFAULT_PRESET)))
        self.pi_up_min = float(kwargs.get("pi_up_min", s_pi.get("pi_up_min", 180.0)))
        self.pi_up_max = float(kwargs.get("pi_up_max", s_pi.get("pi_up_max", 360.0)))
        self.pi_down_min = float(kwargs.get("pi_down_min", s_pi.get("pi_down_min", 0.0)))
        self.pi_down_max = float(kwargs.get("pi_down_max", s_pi.get("pi_down_max", 180.0)))
        self.pi_weight_speed = bool(kwargs.get("pi_weight_speed", s_pi.get("pi_weight_speed", True)))
        preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
        if preset_bounds is not None:
            self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
        self._last_video_path: str = ""
        self._live_preprocessed_drift_map: Optional[Dict[str, Any]] = None
        self._live_preprocessed_drift_path: Optional[str] = None
        self._last_r1_trigger = 0
        self._last_r2_trigger = 0
        self._last_clear_ref_trigger = 0

        self.processor = MotilitometerProcessor(params={
            "processing_mode": s_glb.get("processing_mode", "contour"),

            # Preprocessing & Focus Picking
            "enable_preprocessing": self.enable_preprocessing,
            "monochrome_mode": self.monochrome_mode,
            "invert_image": self.invert_image,
            "prep_blur": self.prep_blur,
            "gamma": self.gamma,
            "crop_type": self.crop_type,
            "crop_radius": self.crop_radius,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "show_crop_outline": self.show_crop_outline,
            "crop_use_abs_center": self.crop_use_abs_center,
            "crop_center_x": self.crop_center_x,
            "crop_center_y": self.crop_center_y,
            "crop_offset_x": self.crop_offset_x,
            "crop_offset_y": self.crop_offset_y,
            "remove_static": self.remove_static,
            "similarity_threshold": self.similarity_threshold,
            "capture_r1_trigger": 0,
            "capture_r2_trigger": 0,
            "clear_ref_trigger": 0,
            "enable_focus_picking": self.enable_focus_picking,
            "focus_metric": self.focus_metric,
            "focus_window_size": self.focus_window_size,
            "focus_output_map": self.focus_output_map,
            "focus_vis_mode": self.focus_vis_mode,
            "focus_threshold": self.focus_threshold,
            "focus_alpha": self.focus_alpha,
            "focus_isolate_selection": self.focus_isolate_selection,
            "focus_use_for_processing": self.focus_use_for_processing,
            "focus_filter_stage": self.focus_filter_stage,
            "blob_alpha": self.blob_alpha,

            # Spatial Calibration & Dense Flow
            "df_calibration": self.calibration,
            "calibration": self.calibration,
            "df_fps": self.fps,
            "fps": self.fps,
            "df_pyr_scale": s_df.get("pyr_scale", 0.5),
            "df_levels": s_df.get("levels", 2),
            "df_winsize": s_df.get("winsize", 15),
            "df_iterations": s_df.get("iterations", 3),
            "df_poly_n": s_df.get("poly_n", 5),
            "df_poly_sigma": s_df.get("poly_sigma", 1.2),
            "df_lower_mag_thresh": s_df.get("lower_mag_thresh", 1.5),
            "df_upper_mag_thresh": s_df.get("upper_mag_thresh", 0.0),
            "df_erosion": s_df.get("erosion", 3),
            "df_supersat": s_df.get("supersat", True),
            "df_visu_mode": s_df.get("visu_mode", "HSV Flow"),
            "df_overlay_enabled": s_df.get("overlay_enabled", True),
            "df_overlay_alpha": s_df.get("overlay_alpha", 0.6),
            "df_heatmap_auto_range": s_df.get("heatmap_auto_range", True),
            "df_heatmap_vmin": s_df.get("heatmap_vmin", 0.0),
            "df_heatmap_vmax": s_df.get("heatmap_vmax", 100.0),
            "df_show_arrows": s_df.get("show_arrows", True),
            "df_arrow_step": s_df.get("arrow_step", 16),
            "df_arrow_scale": s_df.get("arrow_scale", 2.0),
            "df_use_contours": s_df.get("use_contours", False),
            "df_isolate_selection": s_df.get("isolate_selection", False),
            "df_lower_surface_thresh": s_df.get("lower_surface_thresh", 25),
            "df_upper_surface_thresh": s_df.get("upper_surface_thresh", 100),

            # Binarize
            "blur": s_bin.get("blur", 3),
            "block_size": s_bin.get("block_size", 25),
            "bin_thresh": s_bin.get("bin_thresh", 15),
            "erosion": s_bin.get("erosion", 0),

            # Contour Detection
            "enable_contour_detection": s_cnt.get("enable_contour_detection", True),
            "lower_surface_thresh": s_cnt.get("lower_surface_thresh", 25),
            "upper_surface_thresh": s_cnt.get("upper_surface_thresh", 100),
            "show_blobs": s_bin.get("show_blobs", True),
            "isolate_selection": s_cnt.get("isolate_selection", False),
            "show_boxes": s_cnt.get("show_boxes", False),
            "show_centroids": s_cnt.get("show_centroids", False),
            "visu_format": s_cnt.get("visu_format", "Frame"),
            "fill_contours": s_cnt.get("fill_contours", False),
            "enable_watershed": self.enable_watershed,
            "ws_peak_thresh": self.ws_peak_thresh,
            "ws_show_lines": self.ws_show_lines,

            # Tracker
            "enable_tracking": s_trk.get("enable_tracking", False),
            "distance_function": s_trk.get("distance_function", "euclidean"),
            "distance_threshold": s_trk.get("distance_threshold", 30),
            "trail_length": s_trk.get("trail_length", 30),
            "show_trail": s_trk.get("show_trail", False),
            "show_arrow": s_trk.get("show_arrow", False),
            "arrow_scale": s_trk.get("arrow_scale", 3.0),
            "vector_mode": s_trk.get("vector_mode", "Last 2 Points"),
            "show_id": s_trk.get("show_id", False),
            "show_age": s_trk.get("show_age", False),
            "show_distance": s_trk.get("show_distance", False),
            "show_speed": s_trk.get("show_speed", False),
            "track_max_lost": s_trk.get("track_max_lost", 15),
            "track_max_age": s_trk.get("track_max_age", 0),
            "use_tracking_speed": s_trk.get("use_tracking_speed", False),
            "speed_window": s_trk.get("speed_window", 5),
            "track_min_speed": s_trk.get("track_min_speed", 0.0),
            "track_max_speed": s_trk.get("track_max_speed", 0.0),

            # Drift Correction
            "enable_drift_correction": self.enable_drift_correction,
            "drift_angle_tol": self.drift_angle_tol,
            "drift_mag_tol": self.drift_mag_tol,
            "drift_min_corr_speed": self.drift_min_corr_speed,
            "drift_use_hint": self.drift_use_hint,
            "drift_hint_preset": self.drift_hint_preset,
            "drift_hint_angle": self.drift_hint_angle,
            "drift_hint_cone": self.drift_hint_cone,
            "drift_grid_mode": self.drift_grid_mode,
            "drift_lock_live": self.drift_lock_live,
            "drift_use_ranges": self.drift_use_ranges,
            "drift_active_ranges": self.drift_active_ranges,
            "drift_show_vector": self.drift_show_vector,
            "drift_show_filtered": self.drift_show_filtered,
            "drift_show_correction": self.drift_show_correction,
            "drift_show_fake_drifters": self.drift_show_fake_drifters,
            "drift_reuse_preprocessed": self.drift_reuse_preprocessed,
            "drift_only_export": self.drift_only_export,

            # Global
            "active": s_glb.get("active", True),
        },
        buffer_size=s_glb.get("buffer_size", 1),
        drop_policy=s_glb.get("drop_policy", "drop_new"))
        self.processor.start()

        self._init_tags()

        # Presets
        self._presets: Dict[str, str] = self._scan_presets()

        # Monitoring thread to forward results
        self._source_name  = ""    # updated from incoming (frame, name) tuples
        self._batch_running = False  # True while a batch job is in progress
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread_running = True
        self._monitor_thread.start()

        self._build_ui(s_glb, s_df, s_bin, s_cnt, s_trk)
        self.update_permission()

    def _init_tags(self):
        """Declares every widget/group tag (one variable per tag; add new ones here)."""
        U = self.UUID

        # Pre-processing & Focus Picking
        self.enable_preprocessing_tag = f"motilitometer_enable_prep_{U}"
        self.preprocessing_group_tag = f"motilitometer_prep_grp_{U}"
        self.monochrome_tag = f"motilitometer_monochrome_{U}"
        self.invert_image_tag = f"motilitometer_invert_image_{U}"
        self.prep_blur_tag = f"motilitometer_prep_blur_{U}"
        self.gamma_tag = f"motilitometer_gamma_{U}"
        self.crop_type_tag = f"motilitometer_crop_type_{U}"
        self.show_crop_outline_tag = f"motilitometer_show_crop_outline_{U}"
        self.crop_circular_group_tag = f"motilitometer_crop_circ_grp_{U}"
        self.crop_radius_tag = f"motilitometer_crop_radius_{U}"
        self.crop_square_group_tag = f"motilitometer_crop_square_grp_{U}"
        self.crop_size_tag = f"motilitometer_crop_size_{U}"
        self.crop_rect_group_tag = f"motilitometer_crop_rect_grp_{U}"
        self.crop_width_tag = f"motilitometer_crop_width_{U}"
        self.crop_height_tag = f"motilitometer_crop_height_{U}"
        self.crop_use_abs_tag = f"motilitometer_crop_use_abs_{U}"
        self.crop_abs_center_group_tag = f"motilitometer_crop_abs_center_grp_{U}"
        self.crop_center_x_tag = f"motilitometer_crop_center_x_{U}"
        self.crop_center_y_tag = f"motilitometer_crop_center_y_{U}"
        self.crop_rel_center_group_tag = f"motilitometer_crop_rel_center_grp_{U}"
        self.crop_offset_x_tag = f"motilitometer_crop_offset_x_{U}"
        self.crop_offset_y_tag = f"motilitometer_crop_offset_y_{U}"
        self.remove_static_tag = f"motilitometer_remove_static_{U}"
        self.static_removal_group_tag = f"motilitometer_static_grp_{U}"
        self.similarity_threshold_tag = f"motilitometer_sim_thresh_{U}"
        self.r1_status_tag = f"motilitometer_r1_status_{U}"
        self.r2_status_tag = f"motilitometer_r2_status_{U}"
        self.enable_focus_picking_tag = f"motilitometer_focus_pick_{U}"
        self.focus_picking_group_tag = f"motilitometer_focus_grp_{U}"
        self.focus_metric_tag = f"motilitometer_focus_metric_{U}"
        self.focus_window_size_tag = f"motilitometer_focus_win_size_{U}"
        self.focus_output_map_tag = f"motilitometer_focus_out_map_{U}"
        self.focus_vis_mode_tag = f"motilitometer_focus_vis_mode_{U}"
        self.focus_threshold_tag = f"motilitometer_focus_thresh_{U}"
        self.focus_alpha_tag = f"motilitometer_focus_alpha_{U}"
        self.focus_isolate_selection_tag = f"motilitometer_focus_isolate_{U}"
        self.focus_use_for_processing_tag = f"motilitometer_focus_use_proc_{U}"
        self.focus_filter_stage_tag = f"motilitometer_focus_stage_{U}"
        self.focus_score_tag = f"motilitometer_focus_score_{U}"

        # Control
        self.active_tag = f"motilitometer_active_{U}"
        self.mode_tag = f"motilitometer_mode_{U}"
        self.processing_container_tag = f"motilitometer_proc_container_{U}"
        self.mode_contour_group_tag = f"motilitometer_mode_contour_{U}"
        self.contour_group_tag = f"motilitometer_contour_group_{U}"

        # X Axis & Calibration
        self.x_mode_tag = f"motilitometer_x_mode_{U}"
        self.delta_t_tag = f"motilitometer_delta_t_{U}"
        self.delta_t_group_tag = f"motilitometer_delta_t_group_{U}"
        self.calibration_tag = f"motilitometer_calib_{U}"
        self.fps_tag = f"motilitometer_fps_{U}"

        # Dense Flow / Both
        self.df_mag_thresh_tag = f"motilitometer_df_mag_thresh_{U}"
        self.df_upper_mag_thresh_tag = f"motilitometer_df_upper_mag_thresh_{U}"
        self.df_erosion_tag = f"motilitometer_df_erosion_{U}"
        self.df_visu_mode_tag = f"motilitometer_df_visu_mode_{U}"
        self.df_overlay_enabled_tag = f"motilitometer_df_overlay_en_{U}"
        self.df_overlay_alpha_tag = f"motilitometer_df_overlay_alpha_{U}"
        self.df_heatmap_settings_group_tag = f"motilitometer_df_hm_settings_grp_{U}"
        self.df_heatmap_auto_tag = f"motilitometer_df_hm_auto_{U}"
        self.df_heatmap_vmin_tag = f"motilitometer_df_hm_vmin_{U}"
        self.df_heatmap_vmax_tag = f"motilitometer_df_hm_vmax_{U}"
        self.df_heatmap_custom_group_tag = f"motilitometer_df_hm_custom_grp_{U}"
        self.df_supersat_tag = f"motilitometer_df_supersat_{U}"
        self.df_show_arrows_tag = f"motilitometer_df_show_arrows_{U}"
        self.df_arrows_group_tag = f"motilitometer_df_arrows_grp_{U}"
        self.df_arrow_step_tag = f"motilitometer_df_arrow_step_{U}"
        self.df_arrow_scale_tag = f"motilitometer_df_arrow_scale_{U}"
        self.df_winsize_tag = f"motilitometer_df_winsize_{U}"
        self.df_levels_tag = f"motilitometer_df_levels_{U}"
        self.df_pyr_scale_tag = f"motilitometer_df_pyr_scale_{U}"
        self.df_iterations_tag = f"motilitometer_df_iterations_{U}"
        self.df_poly_n_tag = f"motilitometer_df_poly_n_{U}"
        self.df_poly_sigma_tag = f"motilitometer_df_poly_sigma_{U}"
        self.df_use_contours_tag = f"motilitometer_df_use_contours_{U}"
        self.df_isolate_tag = f"motilitometer_df_isolate_{U}"
        self.df_lower_str_tag = f"motilitometer_df_l_str_{U}"
        self.df_upper_str_tag = f"motilitometer_df_u_str_{U}"
        self.df_show_boxes_tag = f"motilitometer_df_show_boxes_{U}"
        self.df_show_centroids_tag = f"motilitometer_df_show_centroids_{U}"
        self.df_group_tag = f"motilitometer_df_group_{U}"
        self.df_contour_group_tag = f"motilitometer_df_contour_group_{U}"
        self.both_vis_tag = f"motilitometer_both_vis_{U}"
        self.both_group_tag = f"motilitometer_both_group_{U}"

        # Binarize
        self.blur_tag = f"motilitometer_blur_{U}"
        self.block_size_tag = f"motilitometer_block_size_{U}"
        self.bin_thresh_tag = f"motilitometer_bin_thresh_{U}"
        self.morph_close_tag = f"motilitometer_morph_close_{U}"

        # Contour Detection
        self.enable_contour_tag = f"motilitometer_enable_contour_{U}"
        self.fill_contours_tag = f"motilitometer_fill_contours_{U}"
        self.lower_str_tag = f"motilitometer_l_str_{U}"
        self.upper_str_tag = f"motilitometer_u_str_{U}"
        self.show_blobs_tag = f"motilitometer_show_blobs_{U}"
        self.blob_alpha_tag = f"motilitometer_blob_alpha_{U}"
        self.isolate_tag = f"motilitometer_isolate_{U}"
        self.show_boxes_tag = f"motilitometer_show_boxes_{U}"
        self.show_centroids_tag = f"motilitometer_show_centroids_{U}"
        self.enable_watershed_tag = f"motilitometer_ws_enable_{U}"
        self.ws_peak_thresh_tag = f"motilitometer_ws_thresh_{U}"
        self.ws_show_lines_tag = f"motilitometer_ws_lines_{U}"
        self.watershed_group_tag = f"motilitometer_ws_group_{U}"

        # Object Tracker (shared across Contour, Dense Flow, and Both modes)
        self.enable_tracking_tag = f"motilitometer_enable_tracking_{U}"
        self.dist_func_tag = f"motilitometer_dist_func_{U}"
        self.dist_thresh_tag = f"motilitometer_dist_thresh_{U}"
        self.trail_length_tag = f"motilitometer_trail_length_{U}"
        self.show_trail_tag = f"motilitometer_show_trail_{U}"
        self.show_arrow_tag = f"motilitometer_show_arrow_{U}"
        self.track_arrow_scale_tag = f"motilitometer_track_arrow_scale_{U}"
        self.track_vector_mode_tag = f"motilitometer_track_vec_mode_{U}"
        self.show_id_tag = f"motilitometer_show_id_{U}"
        self.show_age_tag = f"motilitometer_show_age_{U}"
        self.show_distance_tag = f"motilitometer_show_distance_{U}"
        self.show_speed_tag = f"motilitometer_show_speed_{U}"
        self.track_max_lost_tag = f"motilitometer_trk_max_lost_{U}"
        self.track_max_age_tag = f"motilitometer_trk_max_age_{U}"
        self.use_tracking_speed_tag = f"motilitometer_use_trk_spd_{U}"
        self.speed_window_tag = f"motilitometer_spd_window_{U}"
        self.track_min_speed_tag = f"motilitometer_trk_min_spd_{U}"
        self.track_max_speed_tag = f"motilitometer_trk_max_spd_{U}"
        self.tracker_group_tag = f"motilitometer_tracker_group_{U}"

        # Drift Correction (shared across Tracking & Dense Flow)
        self.enable_drift_tag = f"motilitometer_enable_drift_{U}"
        self.drift_group_tag = f"motilitometer_drift_group_{U}"
        self.drift_calc_params_group_tag = f"motilitometer_drift_calc_params_grp_{U}"
        self.drift_reuse_status_tag = f"motilitometer_drift_reuse_status_{U}"
        self.drift_angle_tol_tag = f"motilitometer_drift_ang_tol_{U}"
        self.drift_mag_tol_tag = f"motilitometer_drift_mag_tol_{U}"
        self.drift_min_corr_speed_tag = f"motilitometer_drift_min_corr_spd_{U}"
        self.drift_grid_mode_tag = f"motilitometer_drift_grid_{U}"
        self.drift_use_hint_tag = f"motilitometer_drift_use_hint_{U}"
        self.drift_hint_group_tag = f"motilitometer_drift_hint_grp_{U}"
        self.drift_hint_preset_tag = f"motilitometer_drift_hint_preset_{U}"
        self.drift_hint_angle_group_tag = f"motilitometer_drift_hint_ang_grp_{U}"
        self.drift_hint_angle_tag = f"motilitometer_drift_hint_ang_{U}"
        self.drift_hint_cone_tag = f"motilitometer_drift_hint_cone_{U}"
        self.drift_lock_live_tag = f"motilitometer_drift_lock_live_{U}"
        self.drift_use_ranges_tag = f"motilitometer_drift_use_ranges_{U}"
        self.drift_ranges_group_tag = f"motilitometer_drift_ranges_grp_{U}"
        self.drift_active_ranges_tag = f"motilitometer_drift_ranges_{U}"
        self.drift_show_vector_tag = f"motilitometer_drift_show_vec_{U}"
        self.drift_show_filtered_tag = f"motilitometer_drift_show_filt_{U}"
        self.drift_show_correction_tag = f"motilitometer_drift_show_corr_{U}"
        self.drift_show_fake_drifters_tag = f"motilitometer_drift_show_fake_{U}"
        self.drift_reuse_preprocessed_tag = f"motilitometer_drift_reuse_preproc_{U}"
        self.drift_only_export_tag = f"motilitometer_drift_only_export_{U}"
        self.drift_status_text_tag = f"motilitometer_drift_status_{U}"

        # Coverage/Speed, Count, Motility, Phototaxis and Reset
        self.metric_label_tag = f"motilitometer_metric_label_{U}"
        self.metric_text_tag = f"motilitometer_metric_text_{U}"
        self.count_label_tag = f"motilitometer_count_label_{U}"
        self.count_text_tag = f"motilitometer_count_text_{U}"
        self.motility_label_tag = f"motilitometer_motility_label_{U}"
        self.motility_text_tag = f"motilitometer_motility_text_{U}"
        self.pi_label_tag = f"motilitometer_pi_label_{U}"
        self.pi_text_tag = f"motilitometer_pi_text_{U}"
        self.emit_pi_tag = f"motilitometer_emit_pi_{U}"
        self.pi_light_pos_tag = f"motilitometer_pi_light_pos_{U}"
        self.pi_preset_tag = f"motilitometer_pi_preset_{U}"
        self.pi_toward_lbl_tag = f"motilitometer_pi_toward_lbl_{U}"
        self.pi_away_lbl_tag = f"motilitometer_pi_away_lbl_{U}"
        self.pi_up_min_tag = f"motilitometer_pi_up_min_{U}"
        self.pi_up_max_tag = f"motilitometer_pi_up_max_{U}"
        self.pi_down_min_tag = f"motilitometer_pi_down_min_{U}"
        self.pi_down_max_tag = f"motilitometer_pi_down_max_{U}"
        self.pi_weight_speed_tag = f"motilitometer_pi_weight_{U}"
        self.reset_btn_tag = f"motilitometer_reset_btn_{U}"

        # Batch processing
        self.n_threads_tag = f"motilitometer_n_threads_{U}"
        self.chunk_size_tag = f"motilitometer_chunk_{U}"
        self.auto_export_csv_tag = f"motilitometer_auto_export_csv_{U}"
        self.auto_export_raw_tag = f"motilitometer_auto_export_raw_{U}"
        self.batch_progress_tag = f"motilitometer_progress_{U}"
        self.batch_status_tag = f"motilitometer_status_{U}"
        self.batch_time_tag = f"motilitometer_batch_time_{U}"
        self.batch_group_tag = f"motilitometer_batch_group_{U}"
        self.batch_inputs_tag = f"batch_inputs_{U}"
        self.batch_progress_group_tag = f"batch_progress_group_{U}"
        self.main_settings_group_tag = f"motilitometer_settings_group_{U}"

        # Presets
        self.preset_menu_tag = f"motilitometer_preset_menu_{U}"
        self.preset_menu_list_tag = f"motilitometer_preset_list_{U}"

    # ── UI construction ────────────────────────────────────────────────
    def _build_ui(self, s_glb, s_df, s_bin, s_cnt, s_trk):
        cb = self._update_param_cb
        scale = display_scaling.scale
        add_check, add_int, add_float, add_combo = (dpg.add_checkbox, dpg.add_drag_int, dpg.add_drag_float,
                                                    dpg.add_combo)
        mode = s_glb.get("processing_mode", "contour")
        use_trk_spd = s_trk.get("use_tracking_speed", False) and s_trk.get("enable_tracking", False)

        def tooltip(tag, text):
            with dpg.tooltip(tag):
                dpg.add_text(text)

        with self.window(menubar=True):

            with dpg.menu_bar():
                with dpg.menu(label="Preset", tag=self.preset_menu_tag):
                    dpg.add_menu_item(label="Save Config...", callback=self._export_settings_cb)
                    dpg.add_menu_item(label="Load Config from File...", callback=self._import_settings_cb)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Scan / Refresh Presets", callback=self._refresh_presets_cb)
                    with dpg.menu(label="Presets", tag=self.preset_menu_list_tag):
                        self._populate_preset_menu()

            with dpg.group(tag=self.main_settings_group_tag):
                # ── Top Reset & Live Metric Display ───────────────────────────
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset", tag=self.reset_btn_tag, callback=self._reset_all_cb,
                                   width=scale(150))
                    tooltip(self.reset_btn_tag,
                            "Reset all tracking traces, downstream plots, samples, and statistics to 0.")
                    dpg.add_text(self._metric_label(mode, use_trk_spd), tag=self.metric_label_tag)
                    dpg.add_text("0.00", tag=self.metric_text_tag)

                with dpg.group(horizontal=True):
                    dpg.add_text("Count:", tag=self.count_label_tag)
                    dpg.add_text("0", tag=self.count_text_tag)
                    dpg.add_text(" | Motility:", tag=self.motility_label_tag)
                    dpg.add_text("--", tag=self.motility_text_tag)
                    dpg.add_text(" | Phototaxis:", tag=self.pi_label_tag)
                    dpg.add_text("--", tag=self.pi_text_tag)

                dpg.add_separator()

                # ── Pre-processing Section ─────────────────────────────
                add_check(label="Pre-processing", default_value=self.enable_preprocessing,
                          callback=self._on_preprocessing_toggle, tag=self.enable_preprocessing_tag)
                with dpg.group(tag=self.preprocessing_group_tag, show=self.enable_preprocessing):
                    with dpg.collapsing_header(label="Pre-processing Settings", default_open=True):
                        add_combo(["Grayscale", "RGB (Color)", "Red Channel", "Green Channel", "Blue Channel"],
                                  default_value=self.monochrome_mode, label="Channel", callback=cb,
                                  tag=self.monochrome_tag)
                        add_check(label="Invert Image (Negative)", default_value=self.invert_image, callback=cb,
                                  tag=self.invert_image_tag)
                        add_int(label="Pre-processing Blur", default_value=self.prep_blur, min_value=0,
                                max_value=51, callback=cb, tag=self.prep_blur_tag)
                        add_float(label="Gamma Adjustment", default_value=self.gamma, min_value=0.1,
                                  max_value=5.0, format="%.2f", speed=0.05, callback=cb, tag=self.gamma_tag)

                        # Crop Settings & Center Coordinates
                        add_combo(["None", "Circular", "Square", "Rectangle"], default_value=self.crop_type,
                                  label="Crop Mode", callback=self._on_crop_type_changed, tag=self.crop_type_tag)
                        add_check(label="Show Crop Outline (Post-processing)",
                                  default_value=self.show_crop_outline, callback=cb,
                                  tag=self.show_crop_outline_tag)
                        with dpg.group(tag=self.crop_circular_group_tag, show=(self.crop_type == "Circular")):
                            add_int(label="Crop Radius (px)", default_value=self.crop_radius, min_value=1,
                                    max_value=100000, callback=cb, tag=self.crop_radius_tag)
                        with dpg.group(tag=self.crop_square_group_tag, show=(self.crop_type == "Square")):
                            add_int(label="Crop Size (px)", default_value=self.crop_width, min_value=1,
                                    max_value=100000, callback=cb, tag=self.crop_size_tag)
                        with dpg.group(tag=self.crop_rect_group_tag, show=(self.crop_type == "Rectangle")):
                            with dpg.group(horizontal=True):
                                add_int(label="Crop W (px)", default_value=self.crop_width, min_value=1,
                                        max_value=100000, width=scale(100), callback=cb, tag=self.crop_width_tag)
                                add_int(label="Crop H (px)", default_value=self.crop_height, min_value=1,
                                        max_value=100000, width=scale(100), callback=cb, tag=self.crop_height_tag)
                        add_check(label="Use Absolute Crop Center (X, Y)", default_value=self.crop_use_abs_center,
                                  callback=self._on_crop_center_mode_changed, tag=self.crop_use_abs_tag)
                        with dpg.group(tag=self.crop_abs_center_group_tag, show=self.crop_use_abs_center):
                            with dpg.group(horizontal=True):
                                for label, attr in (("Crop Center X (px)", "center_x"), ("Crop Center Y (px)", "center_y")):
                                    add_int(label=label, default_value=getattr(self, f"crop_{attr}"),
                                            min_value=-100000, max_value=100000, width=scale(100), callback=cb,
                                            tag=getattr(self, f"crop_{attr}_tag"))
                        with dpg.group(tag=self.crop_rel_center_group_tag, show=not self.crop_use_abs_center):
                            with dpg.group(horizontal=True):
                                for label, attr in (("Crop Offset X", "offset_x"), ("Crop Offset Y", "offset_y")):
                                    add_int(label=label, default_value=getattr(self, f"crop_{attr}"),
                                            min_value=-100000, max_value=100000, width=scale(100), callback=cb,
                                            tag=getattr(self, f"crop_{attr}_tag"))

                        # Static Parasite Removal controls
                        add_check(label="Remove static noise", default_value=self.remove_static,
                                  callback=self._on_remove_static_toggle, tag=self.remove_static_tag)
                        with dpg.group(tag=self.static_removal_group_tag, show=self.remove_static):
                            add_int(label="Similarity Thresh", default_value=self.similarity_threshold,
                                    min_value=0, max_value=255, callback=cb, tag=self.similarity_threshold_tag)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Capture R1", callback=self._capture_r1_cb)
                                dpg.add_button(label="Capture R2", callback=self._capture_r2_cb)
                                dpg.add_button(label="Clear Ref", callback=self._clear_ref_cb)
                            with dpg.group(horizontal=True):
                                dpg.add_text("Frame 1: Empty", tag=self.r1_status_tag, color=(255, 100, 100))
                                dpg.add_text("Frame 2: Empty", tag=self.r2_status_tag, color=(255, 100, 100))

                        dpg.add_separator()
                        add_check(label="Focus Picking", default_value=self.enable_focus_picking,
                                  callback=self._on_focus_picking_toggle, tag=self.enable_focus_picking_tag)
                        with dpg.group(tag=self.focus_picking_group_tag, show=self.enable_focus_picking):
                            add_combo(["Tenengrad (Sobel)", "Modified Laplacian (SML)", "Laplacian Variance"],
                                      default_value=self.focus_metric, label="Focus Metric", callback=cb,
                                      tag=self.focus_metric_tag)
                            add_int(label="Focus Window Size", default_value=self.focus_window_size, min_value=3,
                                    max_value=45, speed=1, callback=cb, tag=self.focus_window_size_tag)
                            add_check(label="Output Focus Map (Viewer)", default_value=self.focus_output_map,
                                      callback=cb, tag=self.focus_output_map_tag)
                            add_combo(["Heatmap (Turbo)", "Grayscale Map", "Peaking (Green Overlay)"],
                                      default_value=self.focus_vis_mode, label="Focus Map Mode", callback=cb,
                                      tag=self.focus_vis_mode_tag)
                            add_int(label="Sensitivity (%)", default_value=self.focus_threshold, min_value=0,
                                    max_value=100, format="%d %%", callback=cb, tag=self.focus_threshold_tag)
                            add_float(label="Overlay Opacity", default_value=self.focus_alpha, min_value=0.0,
                                      max_value=1.0, format="%.2f", speed=0.05, callback=cb,
                                      tag=self.focus_alpha_tag)
                            add_check(label="Isolate In-Focus (Mask Background)",
                                      default_value=self.focus_isolate_selection, callback=cb,
                                      tag=self.focus_isolate_selection_tag)
                            add_check(label="Use for Processing (Filter Blobs)",
                                      default_value=self.focus_use_for_processing, callback=cb,
                                      tag=self.focus_use_for_processing_tag)
                            add_combo(["Pre-processing", "Post-processing"], default_value=self.focus_filter_stage,
                                      label="Filter Stage", callback=cb, tag=self.focus_filter_stage_tag)
                            with dpg.group(horizontal=True):
                                dpg.add_text("Focus Score:", color=(180, 220, 255))
                                dpg.add_text("--", tag=self.focus_score_tag, color=(140, 240, 180))
                dpg.add_separator()

                # ── Processing Section ─────────────────────────────────
                is_proc_active = bool(s_glb.get("active", True))
                with dpg.group(horizontal=True):
                    add_check(label="Processing", default_value=is_proc_active,
                              callback=self._on_processing_toggle, tag=self.active_tag)
                    add_combo(["contour", "denseflow", "both"], default_value=mode, width=scale(120),
                              callback=self._on_mode_change_cb, tag=self.mode_tag)
                dpg.add_separator()

                with dpg.group(tag=self.processing_container_tag, show=is_proc_active):
                    # ── Both Mode UI ────────────────────────────────────
                    with dpg.group(tag=self.both_group_tag, show=(mode == "both")):
                        with dpg.collapsing_header(label="Combined Analysis (Surface + Speed)", default_open=True):
                            dpg.add_text("Visualization:")
                            add_combo(["Surface", "Speed", "Selection"],
                                      default_value=s_df.get("both_vis_mode", "Surface"), callback=cb,
                                      tag=self.both_vis_tag)

                    # ── Dense Flow UI ──────────────────────────────────────
                    with dpg.group(tag=self.df_group_tag, show=(mode == "denseflow")):
                        with dpg.collapsing_header(label="Optical Flow Parameters", default_open=True):
                            add_float(label="Min Velocity Thresh", default_value=s_df.get("lower_mag_thresh", 1.5),
                                      min_value=0.0, max_value=20.0, format="%.2f", speed=0.1, callback=cb,
                                      tag=self.df_mag_thresh_tag)
                            add_float(label="Max Velocity Thresh", default_value=s_df.get("upper_mag_thresh", 0.0),
                                      min_value=0.0, max_value=200.0, format="%.2f", speed=0.1, callback=cb,
                                      tag=self.df_upper_mag_thresh_tag)
                            tooltip(self.df_upper_mag_thresh_tag,
                                    "Ignore optical flow pixels with magnitude above this threshold (0.0 = disabled).")
                            add_int(label="Erosion (noise)", default_value=s_df.get("erosion", 3), min_value=0,
                                    max_value=10, callback=cb, tag=self.df_erosion_tag)
                            add_combo(["HSV Flow", "Speed Heatmap", "Binary Mask", "Arrows Only"],
                                      label="Visu Layer", default_value=s_df.get("visu_mode", "HSV Flow"),
                                      callback=self._on_df_visu_mode_changed, tag=self.df_visu_mode_tag)
                            add_check(label="Overlay on Frame", default_value=s_df.get("overlay_enabled", True), callback=cb,
                                      tag=self.df_overlay_enabled_tag)
                            add_float(label="Opacity", default_value=s_df.get("overlay_alpha", 0.6), min_value=0.0,
                                      max_value=1.0, format="%.2f", speed=0.05, callback=cb,
                                      tag=self.df_overlay_alpha_tag)
                            with dpg.group(tag=self.df_heatmap_settings_group_tag,
                                           show=(s_df.get("visu_mode", "HSV Flow") == "Speed Heatmap")):
                                add_check(label="Heatmap Auto Min/Max", default_value=s_df.get("heatmap_auto_range", True),
                                          callback=self._on_df_hm_auto_toggled, tag=self.df_heatmap_auto_tag)
                                with dpg.group(tag=self.df_heatmap_custom_group_tag,
                                               show=not s_df.get("heatmap_auto_range", True)):
                                    with dpg.group(horizontal=True):
                                        add_float(label="Vmin", default_value=s_df.get("heatmap_vmin", 0.0), min_value=0.0,
                                                  max_value=1000.0, format="%.1f µm/s", speed=1.0,
                                                  width=scale(90), callback=cb,
                                                  tag=self.df_heatmap_vmin_tag)
                                        add_float(label="Vmax", default_value=s_df.get("heatmap_vmax", 100.0), min_value=0.1,
                                                  max_value=2000.0, format="%.1f µm/s", speed=1.0,
                                                  width=scale(90), callback=cb,
                                                  tag=self.df_heatmap_vmax_tag)
                            add_check(label="Supersat Visual", default_value=s_df.get("supersat", True), callback=cb,
                                      tag=self.df_supersat_tag)
                            add_check(label="Show Arrows", default_value=s_df.get("show_arrows", True),
                                      callback=self._on_df_show_arrows_toggled, tag=self.df_show_arrows_tag)
                            with dpg.group(tag=self.df_arrows_group_tag, show=s_df.get("show_arrows", True)):
                                add_int(label="Arrow Step", default_value=s_df.get("arrow_step", 16), min_value=8,
                                        max_value=64, callback=cb, tag=self.df_arrow_step_tag)
                                add_float(label="Arrow Scale", default_value=s_df.get("arrow_scale", 2.0), min_value=0.5,
                                          max_value=10.0, format="%.1f", speed=0.1, callback=cb,
                                          tag=self.df_arrow_scale_tag)

                        with dpg.collapsing_header(label="Algorithm Tuning (Farneback)", default_open=False):
                            add_int(label="Window Size", default_value=s_df.get("winsize", 15), min_value=3, max_value=45,
                                    callback=self._on_df_winsize_cb, tag=self.df_winsize_tag)
                            add_int(label="Pyramid Levels", default_value=s_df.get("levels", 2), min_value=1, max_value=6,
                                    callback=cb, tag=self.df_levels_tag)
                            add_float(label="Pyramid Scale", default_value=s_df.get("pyr_scale", 0.5), min_value=0.1,
                                      max_value=0.9, format="%.2f", speed=0.05, callback=cb,
                                      tag=self.df_pyr_scale_tag)
                            add_int(label="Iterations", default_value=s_df.get("iterations", 3), min_value=1, max_value=10,
                                    callback=cb, tag=self.df_iterations_tag)
                            add_combo([5, 7], label="Poly N", default_value=s_df.get("poly_n", 5), callback=cb,
                                      tag=self.df_poly_n_tag)
                            add_float(label="Poly Sigma", default_value=s_df.get("poly_sigma", 1.2), min_value=0.5,
                                      max_value=2.5, format="%.2f", speed=0.05, callback=cb,
                                      tag=self.df_poly_sigma_tag)

                        with dpg.collapsing_header(label="Contour Mask & Selection", default_open=False):
                            add_check(label="Use Contours / Vectors", default_value=s_df.get("use_contours", False),
                                      callback=self._on_df_contour_toggle_cb, tag=self.df_use_contours_tag)
                            with dpg.group(tag=self.df_contour_group_tag, show=s_df.get("use_contours", False)):
                                add_check(label="Isolate Selection", default_value=s_df.get("isolate_selection", False),
                                          callback=cb, tag=self.df_isolate_tag)
                                add_int(label="Lower thresh %", default_value=s_df.get("lower_surface_thresh", 25),
                                        min_value=0, max_value=100, callback=cb, tag=self.df_lower_str_tag)
                                add_int(label="Upper thresh %", default_value=s_df.get("upper_surface_thresh", 100),
                                        min_value=0, max_value=100, callback=cb, tag=self.df_upper_str_tag)
                                add_check(label="Show boxes", default_value=s_df.get("show_boxes", False),
                                          callback=cb, tag=self.df_show_boxes_tag)
                                add_check(label="Show centroids", default_value=s_df.get("show_centroids", False),
                                          callback=cb, tag=self.df_show_centroids_tag)

                    # ── Contour UI ─────────────────────────────────────────
                    with dpg.group(tag=self.mode_contour_group_tag, show=(mode in ("contour", "both"))):
                        with dpg.collapsing_header(label="Binarize Settings", default_open=True):
                            add_int(label="Blur", default_value=s_bin.get("blur", 3), min_value=1, max_value=25,
                                    callback=cb, tag=self.blur_tag)
                            add_int(label="Block size", default_value=s_bin.get("block_size", 25), min_value=3,
                                    max_value=51, callback=cb, tag=self.block_size_tag)
                            add_int(label="Bin thresh", default_value=s_bin.get("bin_thresh", 15), min_value=0,
                                    max_value=50, callback=cb, tag=self.bin_thresh_tag)
                            add_int(label="Close", default_value=s_bin.get("morph_close", 0), min_value=0,
                                    max_value=5, callback=cb, tag=self.morph_close_tag)
                            add_check(label="Show blobs", default_value=s_bin.get("show_blobs", True), callback=cb,
                                      tag=self.show_blobs_tag)
                            add_float(label="Blob Opacity", default_value=self.blob_alpha, min_value=0.0,
                                      max_value=1.0, format="%.2f", speed=0.05, callback=cb,
                                      tag=self.blob_alpha_tag)

                        with dpg.collapsing_header(label="Contour Detection", default_open=True):
                            cnt_enabled = s_cnt.get("enable_contour_detection", True)
                            add_check(label="Enable Contour Detection", default_value=cnt_enabled,
                                      callback=self._on_contour_toggle_cb, tag=self.enable_contour_tag)
                            with dpg.group(tag=self.contour_group_tag, show=cnt_enabled):
                                add_int(label="Lower thresh", default_value=s_cnt.get("lower_surface_thresh", 25),
                                        min_value=0, max_value=100, callback=cb, tag=self.lower_str_tag)
                                add_int(label="Upper thresh", default_value=s_cnt.get("upper_surface_thresh", 100),
                                        min_value=0, max_value=100, callback=cb, tag=self.upper_str_tag)
                                add_check(label="Show boxes", default_value=s_cnt.get("show_boxes", False),
                                          callback=cb, tag=self.show_boxes_tag)
                                add_check(label="Show centroids", default_value=s_cnt.get("show_centroids", False),
                                          callback=cb, tag=self.show_centroids_tag)
                                add_check(label="Isolate selection", default_value=s_cnt.get("isolate_selection", False),
                                          callback=cb, tag=self.isolate_tag)
                                add_check(label="Fill contours", default_value=s_cnt.get("fill_contours", False),
                                          callback=cb, tag=self.fill_contours_tag)
                                dpg.add_separator()
                                add_check(label="Watershed (Separate Touching)", default_value=self.enable_watershed,
                                          callback=self._on_watershed_toggle_cb, tag=self.enable_watershed_tag)
                                with dpg.group(tag=self.watershed_group_tag, show=self.enable_watershed):
                                    add_int(label="Peak Distance (%)", default_value=self.ws_peak_thresh,
                                            min_value=5, max_value=95, format="%d %%", callback=cb,
                                            tag=self.ws_peak_thresh_tag)
                                    add_check(label="Show Cut Lines", default_value=self.ws_show_lines,
                                              callback=cb, tag=self.ws_show_lines_tag)

                    # ── Unified Object Tracker UI ──────────────────────────
                    with dpg.collapsing_header(label="Object Tracker", default_open=False):
                        add_check(label="Enable Tracking", default_value=s_trk.get("enable_tracking", False),
                                  callback=self._on_tracker_toggle_cb, tag=self.enable_tracking_tag)
                        with dpg.group(tag=self.tracker_group_tag, show=s_trk.get("enable_tracking", False)):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Dist Func:")
                                add_combo(["euclidean", "manhattan"], default_value=s_trk.get("distance_function", "euclidean"),
                                          tag=self.dist_func_tag, callback=cb, width=scale(100))
                            add_int(label="Dist Thresh", default_value=s_trk.get("distance_threshold", 30), min_value=1,
                                    max_value=200, tag=self.dist_thresh_tag, callback=cb)
                            add_check(label="Show ID", default_value=s_trk.get("show_id", False), callback=cb,
                                      tag=self.show_id_tag)
                            add_check(label="Show trail", default_value=s_trk.get("show_trail", False), callback=cb,
                                      tag=self.show_trail_tag)
                            add_int(label="Trail length", default_value=s_trk.get("trail_length", 30), min_value=0,
                                    max_value=100, callback=cb, tag=self.trail_length_tag)
                            add_check(label="Show Arrow", default_value=s_trk.get("show_arrow", False), callback=cb,
                                      tag=self.show_arrow_tag)
                            tooltip(self.show_arrow_tag,
                                    "Draw directional arrows visualizing movement vectors on tracked objects (like in Dense Flow).")
                            add_combo(items=["Last 2 Points", "Trail Start to End"],
                                      default_value=s_trk.get("vector_mode", "Last 2 Points"), label="Vector Mode", callback=cb,
                                      tag=self.track_vector_mode_tag, width=scale(135))
                            tooltip(self.track_vector_mode_tag,
                                    "Vector calculation method:\n- Last 2 Points: Between frame t-1 and frame t.\n- Trail Start to End: Net displacement from start of trail to current position.")
                            add_float(label="Arrow Scale", default_value=s_trk.get("arrow_scale", 3.0), min_value=0.5,
                                      max_value=15.0, format="%.1f", speed=0.1, callback=cb,
                                      tag=self.track_arrow_scale_tag)
                            tooltip(self.track_arrow_scale_tag, "Visual length multiplier for vector arrows.")
                            add_check(label="Show age", default_value=s_trk.get("show_age", False), callback=cb,
                                      tag=self.show_age_tag)
                            add_check(label="Show distance", default_value=s_trk.get("show_distance", False), callback=cb,
                                      tag=self.show_distance_tag)
                            add_check(label="Show speed", default_value=s_trk.get("show_speed", False), callback=cb,
                                      tag=self.show_speed_tag)
                            add_int(label="Max Inactive (frames)", default_value=s_trk.get("track_max_lost", 15),
                                    min_value=1, max_value=100, tag=self.track_max_lost_tag, callback=cb)
                            add_int(label="Max Track Age (frames)", default_value=s_trk.get("track_max_age", 0),
                                    min_value=0, max_value=500, tag=self.track_max_age_tag, callback=cb)
                            add_check(label="Export Tracking Speed",
                                      default_value=s_trk.get("use_tracking_speed", False),
                                      tag=self.use_tracking_speed_tag, callback=self._on_tracking_speed_toggle_cb)
                            add_int(label="Speed Window (frames)", default_value=s_trk.get("speed_window", 5),
                                    min_value=2, max_value=50, tag=self.speed_window_tag, callback=cb)
                            add_float(label="Min Track Speed (µm/s)", default_value=s_trk.get("track_min_speed", 0.0),
                                      min_value=0.0, max_value=2000.0, format="%.1f", speed=0.5,
                                      tag=self.track_min_speed_tag, callback=cb)
                            add_float(label="Max Track Speed (µm/s)", default_value=s_trk.get("track_max_speed", 0.0),
                                      min_value=0.0, max_value=2000.0, format="%.1f", speed=0.5,
                                      tag=self.track_max_speed_tag, callback=cb)
                            tooltip(self.track_max_speed_tag,
                                    "Ignore tracks with speed above this threshold (0.0 = disabled).")

                    # ── Drift Correction UI (Tracking & Dense Flow) ────────
                    with dpg.collapsing_header(label="Drift Correction", default_open=False):
                        add_check(label="Enable Drift Correction", default_value=self.enable_drift_correction,
                                  callback=self._on_drift_toggle_cb, tag=self.enable_drift_tag)
                        tooltip(self.enable_drift_tag,
                                "Detects sedimenting/non-motile algae (shared angle & intensity cluster),\n"
                                "excludes them from mean speed (counting them as non-motile),\n"
                                "and subtracts the mean drift vector from motile algae vectors.")
                        with dpg.group(tag=self.drift_group_tag, show=self.enable_drift_correction):
                            add_check(label="Reuse drift, if present", default_value=self.drift_reuse_preprocessed,
                                      callback=self._on_drift_reuse_toggle_cb, tag=self.drift_reuse_preprocessed_tag)
                            tooltip(self.drift_reuse_preprocessed_tag,
                                    "If <video>_drift.csv exists next to the current video, reuse the preprocessed\n"
                                    "drift vector(s) at the current frame index instead of recalculating drift live.\n"
                                    "When a drift file is found, the live drift estimation menus below are hidden.")
                            dpg.add_text("", tag=self.drift_reuse_status_tag, color=(120, 235, 150), show=False)

                            with dpg.group(tag=self.drift_calc_params_group_tag, show=True):
                                add_check(label="Only export drift", default_value=self.drift_only_export,
                                          callback=cb, tag=self.drift_only_export_tag)
                                tooltip(self.drift_only_export_tag,
                                        "During batch processing (Send path / Process ALL), only export <video>_drift.csv\n"
                                        "and skip generating normal CSV/NPZ/processed exports and downstream plot series.")
                                dpg.add_separator()
                                add_float(label="Angle Tolerance", default_value=self.drift_angle_tol,
                                          min_value=1.0, max_value=90.0, format="±%.1f °", speed=0.5,
                                          callback=cb, tag=self.drift_angle_tol_tag)
                                tooltip(self.drift_angle_tol_tag,
                                        "Angular tolerance (± degrees) around the dominant drift direction.")
                                add_float(label="Intensity Tolerance", default_value=self.drift_mag_tol,
                                          min_value=1.0, max_value=100.0, format="±%.1f %%", speed=0.5,
                                          callback=cb, tag=self.drift_mag_tol_tag)
                                tooltip(self.drift_mag_tol_tag,
                                        "Speed/magnitude tolerance (± %) around the dominant drift speed.")
                                add_combo(
                                    items=[
                                        "Full (1x1)",
                                        "1/2 (1x2)",
                                        "1/2 (2x1)",
                                        "1/4 (2x2)",
                                        "1/8 (2x4)",
                                        "1/8 (4x2)",
                                        "1/16 (4x4)",
                                        "1/32 (4x8)",
                                        "1/64 (8x8)",
                                    ],
                                    default_value=self.drift_grid_mode,
                                    label="Sub-sections (Grid)",
                                    callback=cb,
                                    tag=self.drift_grid_mode_tag,
                                    width=scale(120),
                                )
                                tooltip(self.drift_grid_mode_tag,
                                        "Compute drift globally on the full image (1x1) or locally inside 2D sub-sections (1/2, 1/4, 1/8, 1/16...).")

                                dpg.add_separator()
                                add_check(label="Give Hint (Global Direction)", default_value=self.drift_use_hint,
                                          callback=self._on_drift_hint_toggle_cb, tag=self.drift_use_hint_tag)
                                tooltip(self.drift_use_hint_tag,
                                        "Restrict candidate drift vectors to a known global sedimentation direction to reduce false positives.")
                                with dpg.group(tag=self.drift_hint_group_tag, show=self.drift_use_hint):
                                    add_combo(
                                        items=["Down (90°)", "Up (270°)", "Right (0°)", "Left (180°)", "Custom Angle"],
                                        default_value=self.drift_hint_preset,
                                        label="Hint Direction",
                                        callback=self._on_drift_hint_preset_cb,
                                        tag=self.drift_hint_preset_tag,
                                        width=scale(120),
                                    )
                                    with dpg.group(tag=self.drift_hint_angle_group_tag,
                                                   show=(self.drift_hint_preset == "Custom Angle")):
                                        add_float(label="Hint Angle", default_value=self.drift_hint_angle,
                                                  min_value=0.0, max_value=360.0, format="%.1f °", speed=1.0,
                                                  callback=cb, tag=self.drift_hint_angle_tag)
                                    add_float(label="Hint Cone", default_value=self.drift_hint_cone,
                                              min_value=5.0, max_value=180.0, format="±%.1f °", speed=1.0,
                                              callback=cb, tag=self.drift_hint_cone_tag)
                                    tooltip(self.drift_hint_cone_tag,
                                            "Only vectors within Hint Direction ± Hint Cone are eligible to be classified as drift.")

                                dpg.add_separator()
                                add_check(label="Lock Decision (Live Freeze)", default_value=self.drift_lock_live,
                                          callback=cb, tag=self.drift_lock_live_tag)
                                tooltip(self.drift_lock_live_tag,
                                        "Stop recalculating drift and keep the last computed drift vector(s) (useful during phototaxis stimulation).")
                                add_check(label="Use Active Frame Ranges", default_value=self.drift_use_ranges,
                                          callback=self._on_drift_ranges_toggle_cb, tag=self.drift_use_ranges_tag)
                                tooltip(self.drift_use_ranges_tag,
                                        "Define frame intervals where drift is actively calculated; outside these intervals the last drift vector is locked.")
                                with dpg.group(tag=self.drift_ranges_group_tag, show=self.drift_use_ranges):
                                    dpg.add_input_text(
                                        label="Active Ranges",
                                        default_value=self.drift_active_ranges,
                                        callback=cb,
                                        tag=self.drift_active_ranges_tag,
                                        width=scale(150),
                                    )
                                    tooltip(self.drift_active_ranges_tag,
                                            "Nomenclature: e.g. [0-500],[1000-1500]\n"
                                            "- Frames 0 to 500: active drift calculation\n"
                                            "- Frames 501 to 999: locked (reuses frame 500 drift vector)\n"
                                            "- Frames 1000 to 1500: active drift calculation")

                            dpg.add_separator()
                            add_float(label="Min Corr Speed (µm/s)", default_value=self.drift_min_corr_speed,
                                      min_value=0.0, max_value=2000.0, format="%.1f", speed=0.5,
                                      callback=cb, tag=self.drift_min_corr_speed_tag)
                            tooltip(self.drift_min_corr_speed_tag,
                                    "Post-correction minimum speed threshold (µm/s).\n"
                                    "After subtracting the drift vector, any residual vector/track slower than this\n"
                                    "threshold is ignored (classified as drift/non-motile) so it does not lower the mean speed.")
                            add_check(label="Show Drift Vector", default_value=self.drift_show_vector,
                                      callback=cb, tag=self.drift_show_vector_tag)
                            tooltip(self.drift_show_vector_tag,
                                    "Display arrow(s) on the image showing drift direction and intensity (same scaling as swimming arrows).")
                            add_check(label="Show Drift Vectors / Tracks", default_value=self.drift_show_filtered,
                                      callback=cb, tag=self.drift_show_filtered_tag)
                            tooltip(self.drift_show_filtered_tag,
                                    "Show or hide detections/tracks/flow vectors categorized as drift (drawn in orange when shown).")
                            add_check(label="Show Pre/Post Correction", default_value=self.drift_show_correction,
                                      callback=cb, tag=self.drift_show_correction_tag)
                            tooltip(self.drift_show_correction_tag,
                                    "Display both pre-correction (yellow) and post-correction (green) vectors on motile objects.")
                            add_check(label="Add Fake Drifters", default_value=self.drift_show_fake_drifters,
                                      callback=cb, tag=self.drift_show_fake_drifters_tag)
                            tooltip(self.drift_show_fake_drifters_tag,
                                    "Overlay a uniform grid of synthetic 'fake algae' (cyan) that translate purely\n"
                                    "according to the detected local/global drift, allowing direct visual comparison\n"
                                    "with real drifting algae in the video.")
                            with dpg.group(horizontal=True):
                                dpg.add_text("Drift:", color=(220, 180, 255))
                                dpg.add_text("--", tag=self.drift_status_text_tag, color=(200, 240, 220))

                    # ── Phototaxis Index UI ────────────────────────────────
                    t_arrow, a_arrow = pi_light_arrows(self.pi_light_pos)
                    with dpg.collapsing_header(label="Phototaxis Index", default_open=False):
                        with dpg.group(horizontal=True):
                            add_check(label="Emit Series", default_value=bool(self.emit_phototaxis_series),
                                      callback=self._on_pi_setting_cb, tag=self.emit_pi_tag)
                            tooltip(self.emit_pi_tag,
                                    "Emit Phototaxis Index (Toward - Away)/(Toward + Away) [-1, +1] time series\n"
                                    "on Result and Phototaxis output ports.")
                            add_check(label="Weight Speed", default_value=bool(self.pi_weight_speed),
                                      callback=self._on_pi_setting_cb, tag=self.pi_weight_speed_tag)
                            tooltip(self.pi_weight_speed_tag,
                                    "Weight each angle by vector speed magnitude (µm/s) instead of raw vector counts.")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Light Pos:", color=(150, 150, 150))
                            add_combo(
                                items=list(PI_LIGHT_POSITIONS),
                                default_value=self.pi_light_pos,
                                callback=self._on_pi_light_pos_cb,
                                tag=self.pi_light_pos_tag,
                                width=scale(130),
                            )
                            tooltip(self.pi_light_pos_tag,
                                    "Position of the light source in screen coordinates:\n"
                                    "- Bottom (↓): 270° (default)\n"
                                    "- Top (↑): 90°\n"
                                    "- Right (→): 0°\n"
                                    "- Left (←): 180°\n"
                                    "Positive index (+1) = movement toward the light.")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Preset:   ", color=(150, 150, 150))
                            add_combo(
                                items=list(PI_PRESETS),
                                default_value=self.pi_preset,
                                callback=self._on_pi_preset_cb,
                                tag=self.pi_preset_tag,
                                width=scale(210),
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"Toward ({t_arrow}):", tag=self.pi_toward_lbl_tag, color=(100, 220, 140))
                            add_float(tag=self.pi_up_min_tag, default_value=float(self.pi_up_min),
                                      min_value=0.0, max_value=360.0, speed=1.0, format="%.0f°",
                                      width=scale(75), callback=self._on_pi_custom_cb)
                            dpg.add_text("to", color=(150, 150, 150))
                            add_float(tag=self.pi_up_max_tag, default_value=float(self.pi_up_max),
                                      min_value=0.0, max_value=360.0, speed=1.0, format="%.0f°",
                                      width=scale(75), callback=self._on_pi_custom_cb)
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"Away ({a_arrow}):  ", tag=self.pi_away_lbl_tag, color=(230, 140, 110))
                            add_float(tag=self.pi_down_min_tag, default_value=float(self.pi_down_min),
                                      min_value=0.0, max_value=360.0, speed=1.0, format="%.0f°",
                                      width=scale(75), callback=self._on_pi_custom_cb)
                            dpg.add_text("to", color=(150, 150, 150))
                            add_float(tag=self.pi_down_max_tag, default_value=float(self.pi_down_max),
                                      min_value=0.0, max_value=360.0, speed=1.0, format="%.0f°",
                                      width=scale(75), callback=self._on_pi_custom_cb)

                dpg.add_separator()
                with dpg.collapsing_header(label="Speed & Spatial Calibration", default_open=False):
                    add_float(label="Scale (µm/px)", default_value=self.calibration, min_value=0.0001,
                              max_value=10000.0, format="%.4f", speed=0.005, callback=cb, tag=self.calibration_tag)
                    tooltip(self.calibration_tag,
                            "Spatial calibration factor converting pixels to micrometers (µm/pixel).")
                    add_float(label="Acquisition FPS", default_value=self.fps, min_value=0.1, max_value=1000.0,
                              format="%.1f fps", speed=0.5, callback=cb, tag=self.fps_tag)
                    tooltip(self.fps_tag, "Video acquisition framerate (FPS) used to convert velocities to µm/s.")

                dpg.add_separator()
                with dpg.collapsing_header(label="X Axis Scale Settings", default_open=False):
                    add_combo(items=["Timestamp", "Index", "Fixed ΔT"], default_value=self.x_axis_mode,
                              tag=self.x_mode_tag, width=-1, callback=self._on_x_mode_changed)
                    with dpg.group(tag=self.delta_t_group_tag, show=(self.x_axis_mode == "Fixed ΔT")):
                        dpg.add_input_float(label="ΔT (s/frame)", tag=self.delta_t_tag, default_value=self.delta_t,
                                            min_value=1e-6, format="%.4f", width=-1, step=0,
                                            callback=self._on_delta_t_changed)

            dpg.add_separator()
            # ── Batch Processing UI ───────────────────────────────────
            with dpg.collapsing_header(label="Batch Video Processing", default_open=False):
                with dpg.group(tag=self.batch_group_tag):
                    with dpg.group(tag=self.batch_inputs_tag):
                        dpg.add_input_int(label="Threads", tag=self.n_threads_tag, default_value=8, min_value=1,
                                          max_value=32, width=scale(80), step=0)
                        dpg.add_input_int(label="Chunk (frames)", tag=self.chunk_size_tag, default_value=50,
                                          min_value=1, max_value=5000, width=scale(80), step=0)
                        add_check(label="Auto Export CSV", tag=self.auto_export_csv_tag,
                                  default_value=self.auto_export_csv, callback=self._on_auto_export_csv_toggled)
                        add_check(label="Auto Export Raw Data (npz)", tag=self.auto_export_raw_tag,
                                  default_value=self.auto_export_raw, callback=self._on_auto_export_raw_toggled)
                        tooltip(self.auto_export_raw_tag,
                                "Saves <video>.npz next to the video: per-frame angle & speed distributions, "
                                "mean vector and the Result/Count/Motility series.\n"
                                "Replay it in sync with the video using the Motility Reprocessor module.")

                    with dpg.group(tag=self.batch_progress_group_tag, show=False):
                        dpg.add_progress_bar(tag=self.batch_progress_tag, default_value=0.0, width=-1)
                        dpg.add_text("", tag=self.batch_status_tag)
                        dpg.add_text("", tag=self.batch_time_tag, color=(200, 200, 200))

    # ── Small helpers ──────────────────────────────────────────────────
    @staticmethod
    def _metric_label(mode: str, tracked: bool) -> str:
        if tracked:
            if mode == "both":
                return "Surface% | Track Spd:"
            return "Tracking Speed (µm/s):"
        if mode == "denseflow":
            return "Mean Speed (µm/s):"
        if mode == "both":
            return "Surface% | Speed (µm/s):"
        return "Mask coverage:"

    @staticmethod
    def _show(tag, flag):
        dpg.configure_item(tag, show=bool(flag))

    @staticmethod
    def _val(tag, default=None, cast=None):
        """Widget value if the widget exists, else `default` (cast is applied to the widget value only)."""
        if not dpg.does_item_exist(tag):
            return default
        v = dpg.get_value(tag)
        return cast(v) if cast else v

    @staticmethod
    def _set_text(tag, text):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, text)

    def _suid(self, prefix: str, source: str) -> str:
        """Deterministic sample UUID for a given series prefix and source name."""
        key = f"{prefix}_{self.UUID}_{source}" if prefix else f"{self.UUID}_{source}"
        return str(_uuid.uuid5(_uuid.NAMESPACE_DNS, key))

    def _set_ref_status(self, tag, captured: bool, which: int):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, f"Frame {which}: {'Captured' if captured else 'Empty'}")
            dpg.configure_item(tag, color=(100, 255, 100) if captured else (255, 100, 100))

    # ── Presets Management ─────────────────────────────────────
    def _scan_presets(self) -> Dict[str, str]:
        """Scans module directory and presets/ subfolder for .json configuration files."""
        module_dir = os.path.dirname(os.path.abspath(__file__))
        presets: Dict[str, str] = {}
        for folder, prefix in ((module_dir, ""), (os.path.join(module_dir, "presets"), "presets/")):
            try:
                for entry in os.scandir(folder):
                    if entry.is_file() and entry.name.lower().endswith(".json"):
                        presets[f"{prefix}{entry.name}"] = os.path.abspath(entry.path)
            except Exception:
                pass
        return presets

    def _populate_preset_menu(self, parent=None):
        for name, path in self._presets.items():
            kw = {"parent": parent} if parent else {}
            dpg.add_menu_item(label=name, user_data=path, callback=self._on_preset_menu_item_clicked, **kw)

    def _refresh_presets_cb(self, sender=None, app_data=None, user_data=None, select_path: Optional[str] = None, *args, **kwargs):
        """Refreshes the presets list by rescanning the module folder."""
        self._presets = self._scan_presets()
        if select_path and os.path.exists(select_path):
            norm = lambda p: os.path.normcase(os.path.normpath(p))
            if not any(norm(p) == norm(select_path) for p in self._presets.values()):
                self._presets[os.path.basename(select_path)] = select_path

        if dpg.does_item_exist(self.preset_menu_list_tag):
            dpg.delete_item(self.preset_menu_list_tag, children_only=True)
            self._populate_preset_menu(parent=self.preset_menu_list_tag)

    def _on_preset_menu_item_clicked(self, sender, app_data, user_data, *args, **kwargs):
        """Called when a preset is selected from the Presets menu item list."""
        self._load_config_file(user_data, "Applied preset from")

    def _load_config_file(self, file_path, success_msg):
        """Loads a JSON config file, applies it to the widgets and refreshes the preset menu."""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self._apply_imported_params(json.load(f))
                self._refresh_presets_cb(select_path=file_path)
                logger.info(f"CounterTracker: {success_msg} {file_path}")
            except Exception as e:
                logger.error(f"CounterTracker: Error applying config from {file_path}: {e}")

    def _run_file_dialog(self, dialog: str, on_path, **opts):
        """Runs a tkinter file dialog (`asksaveasfilename`/`askopenfilename`) in a thread, then calls on_path(path)."""
        def run_dialog():
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                path = getattr(filedialog, dialog)(
                    initialdir=os.path.dirname(os.path.abspath(__file__)),
                    filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")], **opts)
                root.destroy()
                if path:
                    on_path(path)
            except Exception as e:
                logger.error(f"CounterTracker: Settings dialog error: {e}")

        threading.Thread(target=run_dialog, daemon=True).start()

    def _export_settings_cb(self, sender=None, app_data=None, user_data=None, *args, **kwargs):
        """Opens a file dialog to export current module settings to a JSON file."""
        def save(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._get_current_params(), f, indent=4)
            logger.info(f"CounterTracker: Settings successfully exported to {file_path}")
            self._refresh_presets_cb(select_path=file_path)

        self._run_file_dialog("asksaveasfilename", save, title="Export Counter Tracker Configuration",
                              initialfile="config_counter_tracker.json", defaultextension=".json")

    def _import_settings_cb(self, sender=None, app_data=None, user_data=None, *args, **kwargs):
        """Opens a file dialog to import module settings from any JSON file."""
        self._run_file_dialog(
            "askopenfilename", lambda p: self._load_config_file(p, "Settings successfully imported from"),
            title="Import Counter Tracker Configuration")

    def _get_current_params(self) -> dict:
        """Collects current configuration from UI widgets formatted like settings.json."""
        g = self._val
        crop_type = g(self.crop_type_tag, self.crop_type)
        crop_w = g(self.crop_width_tag, g(self.crop_size_tag, self.crop_width))
        crop_h = g(self.crop_height_tag, g(self.crop_size_tag, self.crop_height))
        calib, fps = g(self.calibration_tag, self.calibration), g(self.fps_tag, self.fps)
        remove_static = g(self.remove_static_tag, self.remove_static)
        sim_thresh = g(self.similarity_threshold_tag, self.similarity_threshold)
        outline = g(self.show_crop_outline_tag, self.show_crop_outline)
        abs_center = g(self.crop_use_abs_tag, self.crop_use_abs_center)
        cx, cy = g(self.crop_center_x_tag, self.crop_center_x), g(self.crop_center_y_tag, self.crop_center_y)
        ox, oy = g(self.crop_offset_x_tag, self.crop_offset_x), g(self.crop_offset_y_tag, self.crop_offset_y)
        radius = g(self.crop_radius_tag, self.crop_radius)

        return {
            "preprocessing": {
                "enable_preprocessing": g(self.enable_preprocessing_tag, True),
                "channel": g(self.monochrome_tag, "Grayscale"),
                "invert_image": g(self.invert_image_tag, False),
                "prep_blur": g(self.prep_blur_tag, 0),
                "gamma": g(self.gamma_tag, self.gamma),
                "crop_type": crop_type,
                "show_crop_outline": outline,
                "crop_radius": radius,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "crop_use_abs_center": abs_center,
                "crop_center_x": cx,
                "crop_center_y": cy,
                "crop_offset_x": ox,
                "crop_offset_y": oy,
                "remove_static": remove_static,
                "similarity_threshold": sim_thresh,
                "enable_focus_picking": g(self.enable_focus_picking_tag, False),
                "focus_metric": g(self.focus_metric_tag, "Tenengrad (Sobel)"),
                "focus_window_size": g(self.focus_window_size_tag, 15),
                "focus_output_map": g(self.focus_output_map_tag, True),
                "focus_vis_mode": g(self.focus_vis_mode_tag, "Heatmap (Turbo)"),
                "focus_threshold": g(self.focus_threshold_tag, 40),
                "focus_alpha": g(self.focus_alpha_tag, 0.6),
                "focus_isolate_selection": g(self.focus_isolate_selection_tag, False),
                "focus_use_for_processing": g(self.focus_use_for_processing_tag, False),
                "focus_filter_stage": g(self.focus_filter_stage_tag, "Pre-processing")
            },
            "static_removal": {
                "remove_static": remove_static,
                "similarity_threshold": sim_thresh,
            },
            "crop": {
                "crop_type": crop_type,
                "circular_crop": crop_type == "Circular",
                "crop_radius": radius,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "show_crop_outline": outline,
                "use_absolute_center": abs_center,
                "center_x": cx,
                "center_y": cy,
                "offset_x": ox,
                "offset_y": oy,
            },
            "binarize": {
                "blur": g(self.blur_tag, 3),
                "block_size": g(self.block_size_tag, 51),
                "bin_thresh": g(self.bin_thresh_tag, 11),
                "morph_close": g(self.morph_close_tag, 3),
                "show_blobs": g(self.show_blobs_tag, True),
                "blob_alpha": g(self.blob_alpha_tag, 0.5)
            },
            "contour": {
                "enable_contour_detection": g(self.enable_contour_tag, True),
                "lower_surface_thresh": g(self.lower_str_tag, 25),
                "upper_surface_thresh": g(self.upper_str_tag, 100),
                "show_boxes": g(self.show_boxes_tag, False),
                "show_centroids": g(self.show_centroids_tag, False),
                "isolate_selection": g(self.isolate_tag, False),
                "fill_contours": g(self.fill_contours_tag, True),
                "enable_watershed": g(self.enable_watershed_tag, False),
                "ws_peak_thresh": g(self.ws_peak_thresh_tag, 40),
                "ws_show_lines": g(self.ws_show_lines_tag, True),
                "visu_format": "Frame"
            },
            "tracker": {
                "enable_tracking": g(self.enable_tracking_tag, False),
                "distance_function": g(self.dist_func_tag, "euclidean"),
                "distance_threshold": g(self.dist_thresh_tag, 30),
                "trail_length": g(self.trail_length_tag, 30),
                "show_trail": g(self.show_trail_tag, False),
                "show_arrow": g(self.show_arrow_tag, False),
                "arrow_scale": g(self.track_arrow_scale_tag, 3.0),
                "vector_mode": g(self.track_vector_mode_tag, "Last 2 Points"),
                "show_id": g(self.show_id_tag, False),
                "show_age": g(self.show_age_tag, False),
                "show_distance": g(self.show_distance_tag, False),
                "show_speed": g(self.show_speed_tag, False),
                "track_max_lost": g(self.track_max_lost_tag, 15),
                "track_max_age": g(self.track_max_age_tag, 0),
                "use_tracking_speed": g(self.use_tracking_speed_tag, False),
                "speed_window": g(self.speed_window_tag, 5),
                "track_min_speed": g(self.track_min_speed_tag, 0.0),
                "track_max_speed": g(self.track_max_speed_tag, 0.0),
                "calibration": calib,
                "fps": fps
            },
            "denseflow": {
                "pyr_scale": g(self.df_pyr_scale_tag, 0.5),
                "levels": g(self.df_levels_tag, 2),
                "winsize": g(self.df_winsize_tag, 15),
                "iterations": g(self.df_iterations_tag, 3),
                "poly_n": g(self.df_poly_n_tag, 5),
                "poly_sigma": g(self.df_poly_sigma_tag, 1.2),
                "lower_mag_thresh": g(self.df_mag_thresh_tag, 1.5),
                "upper_mag_thresh": g(self.df_upper_mag_thresh_tag, 0.0),
                "erosion": g(self.df_erosion_tag, 6),
                "supersat": g(self.df_supersat_tag, True),
                "calibration": calib,
                "fps": fps,
                "use_contours": g(self.df_use_contours_tag, False),
                "show_arrows": g(self.df_show_arrows_tag, True),
                "arrow_step": g(self.df_arrow_step_tag, 16),
                "arrow_scale": g(self.df_arrow_scale_tag, 2.0),
                "visu_mode": g(self.df_visu_mode_tag, "HSV Flow"),
                "overlay_enabled": g(self.df_overlay_enabled_tag, True),
                "overlay_alpha": g(self.df_overlay_alpha_tag, 0.6),
                "heatmap_auto_range": g(self.df_heatmap_auto_tag, True),
                "heatmap_vmin": g(self.df_heatmap_vmin_tag, 0.0),
                "heatmap_vmax": g(self.df_heatmap_vmax_tag, 100.0),
                "isolate_selection": g(self.df_isolate_tag, False),
                "lower_surface_thresh": g(self.df_lower_str_tag, 25),
                "upper_surface_thresh": g(self.df_upper_str_tag, 100),
                "show_boxes": g(self.df_show_boxes_tag, False),
                "show_centroids": g(self.df_show_centroids_tag, False),
                "both_vis_mode": g(self.both_vis_tag, "Surface")
            },
            "drift_correction": {
                "enable_drift_correction": g(self.enable_drift_tag, False),
                "drift_angle_tol": g(self.drift_angle_tol_tag, 15.0),
                "drift_mag_tol": g(self.drift_mag_tol_tag, 25.0),
                "drift_min_corr_speed": g(self.drift_min_corr_speed_tag, 0.0),
                "drift_use_hint": g(self.drift_use_hint_tag, False),
                "drift_hint_preset": g(self.drift_hint_preset_tag, "Down (90°)"),
                "drift_hint_angle": g(self.drift_hint_angle_tag, 90.0),
                "drift_hint_cone": g(self.drift_hint_cone_tag, 45.0),
                "drift_grid_mode": g(self.drift_grid_mode_tag, "Full (1x1)"),
                "drift_lock_live": g(self.drift_lock_live_tag, False),
                "drift_use_ranges": g(self.drift_use_ranges_tag, False),
                "drift_active_ranges": g(self.drift_active_ranges_tag, "[0-500],[1000-1500]"),
                "drift_show_vector": g(self.drift_show_vector_tag, True),
                "drift_show_filtered": g(self.drift_show_filtered_tag, False),
                "drift_show_correction": g(self.drift_show_correction_tag, False),
                "drift_show_fake_drifters": g(self.drift_show_fake_drifters_tag, False),
                "drift_reuse_preprocessed": g(self.drift_reuse_preprocessed_tag, False),
                "drift_only_export": g(self.drift_only_export_tag, False)
            },
            "phototaxis": {
                "emit_phototaxis_series": g(self.emit_pi_tag, self.emit_phototaxis_series),
                "pi_light_pos": normalize_pi_light_pos(g(self.pi_light_pos_tag, self.pi_light_pos)),
                "pi_preset": normalize_pi_preset(g(self.pi_preset_tag, self.pi_preset)),
                "pi_up_min": g(self.pi_up_min_tag, self.pi_up_min),
                "pi_up_max": g(self.pi_up_max_tag, self.pi_up_max),
                "pi_down_min": g(self.pi_down_min_tag, self.pi_down_min),
                "pi_down_max": g(self.pi_down_max_tag, self.pi_down_max),
                "pi_weight_speed": g(self.pi_weight_speed_tag, self.pi_weight_speed),
            },
            "global": {
                "active": g(self.active_tag, True),
                "processing_mode": g(self.mode_tag, "contour"),
                "buffer_size": getattr(self.processor, "buffer_size", 1),
                "drop_policy": getattr(self.processor, "drop_policy", "drop_new"),
                "x_axis_mode": g(self.x_mode_tag, self.x_axis_mode),
                "delta_t": g(self.delta_t_tag, self.delta_t),
                "auto_export_csv": g(self.auto_export_csv_tag, self.auto_export_csv),
                "auto_export_raw": g(self.auto_export_raw_tag, self.auto_export_raw)
            }
        }

    def _apply_imported_params(self, p: dict):
        """Applies loaded dictionary parameters to UI widgets and updates the processor."""
        if not isinstance(p, dict):
            return

        flat: Dict[str, Any] = {}
        if any(k in p for k in ["preprocessing", "binarize", "contour", "tracker", "denseflow", "drift_correction", "phototaxis", "global"]):
            s_prep = p.get("preprocessing", {})
            s_crop = p.get("crop", {})
            s_mono = p.get("monochrome", {})
            s_bin  = p.get("binarize", {})
            s_cnt  = p.get("contour", {})
            s_trk  = p.get("tracker", {})
            s_df   = p.get("denseflow", {})
            s_drf  = p.get("drift_correction", {})
            s_pi   = p.get("phototaxis", {})
            s_glb  = p.get("global", {})
            s_stat = p.get("static_removal", {})

            # Preprocessing & Crop
            if "enable_preprocessing" in s_prep: flat["enable_preprocessing"] = s_prep.get("enable_preprocessing")
            if "channel" in s_prep or "monochrome_mode" in s_prep:
                flat["monochrome_mode"] = s_prep.get("channel", s_prep.get("monochrome_mode"))
            elif "channel" in s_mono:
                flat["monochrome_mode"] = s_mono.get("channel")
            if "invert_image" in s_prep: flat["invert_image"] = s_prep.get("invert_image")
            elif "invert_image" in s_mono: flat["invert_image"] = s_mono.get("invert_image")
            if "prep_blur" in s_prep: flat["prep_blur"] = s_prep.get("prep_blur")
            if "gamma" in s_prep: flat["gamma"] = s_prep.get("gamma")
            elif "gamma" in s_mono: flat["gamma"] = s_mono.get("gamma")

            crop_mode_val = s_prep.get("crop_type", s_crop.get("crop_type", None))
            if crop_mode_val is None and "circular_crop" in s_crop:
                crop_mode_val = "Circular" if s_crop.get("circular_crop") else "None"
            if crop_mode_val is not None: flat["crop_type"] = crop_mode_val

            for crop_k in ["crop_radius", "crop_width", "crop_height", "show_crop_outline"]:
                if crop_k in s_prep: flat[crop_k] = s_prep.get(crop_k)
                elif crop_k in s_crop: flat[crop_k] = s_crop.get(crop_k)
            if "crop_width" in flat: flat["crop_size"] = flat["crop_width"]

            abs_c = s_prep.get("crop_use_abs_center", s_crop.get("crop_use_abs_center", s_crop.get("use_absolute_center", None)))
            if abs_c is not None: flat["crop_use_abs_center"] = abs_c
            cx = s_prep.get("crop_center_x", s_crop.get("crop_center_x", s_crop.get("center_x", None)))
            if cx is not None: flat["crop_center_x"] = cx
            cy = s_prep.get("crop_center_y", s_crop.get("crop_center_y", s_crop.get("center_y", None)))
            if cy is not None: flat["crop_center_y"] = cy
            ox = s_prep.get("crop_offset_x", s_crop.get("crop_offset_x", s_crop.get("offset_x", None)))
            if ox is not None: flat["crop_offset_x"] = ox
            oy = s_prep.get("crop_offset_y", s_crop.get("crop_offset_y", s_crop.get("offset_y", None)))
            if oy is not None: flat["crop_offset_y"] = oy

            if "remove_static" in s_prep: flat["remove_static"] = s_prep.get("remove_static")
            elif "remove_static" in s_stat: flat["remove_static"] = s_stat.get("remove_static")
            if "similarity_threshold" in s_prep: flat["similarity_threshold"] = s_prep.get("similarity_threshold")
            elif "similarity_threshold" in s_stat: flat["similarity_threshold"] = s_stat.get("similarity_threshold")

            for fk in ["enable_focus_picking", "focus_metric", "focus_window_size", "focus_output_map",
                       "focus_vis_mode", "focus_threshold", "focus_alpha", "focus_isolate_selection",
                       "focus_use_for_processing", "focus_filter_stage"]:
                if fk in s_prep:
                    flat[fk] = s_prep.get(fk)

            # Binarize
            for bk in ["blur", "block_size", "bin_thresh", "morph_close", "show_blobs", "blob_alpha"]:
                if bk in s_bin:
                    flat[bk] = s_bin.get(bk)

            # Contour
            for ck in ["enable_contour_detection", "lower_surface_thresh", "upper_surface_thresh",
                       "show_boxes", "show_centroids", "isolate_selection", "fill_contours",
                       "enable_watershed", "ws_peak_thresh", "ws_show_lines"]:
                if ck in s_cnt:
                    flat[ck] = s_cnt.get(ck)

            # Tracker
            for tk in ["enable_tracking", "distance_function", "distance_threshold", "trail_length",
                       "show_trail", "show_arrow", "arrow_scale", "vector_mode", "show_id", "show_age",
                       "show_distance", "show_speed", "track_max_lost", "track_max_age",
                       "use_tracking_speed", "speed_window", "track_min_speed", "track_max_speed",
                       "calibration", "fps"]:
                if tk in s_trk:
                    flat[tk] = s_trk.get(tk)

            # Denseflow
            if "calibration" in s_df: flat["calibration"] = s_df.get("calibration")
            if "fps" in s_df: flat["fps"] = s_df.get("fps")
            if "df_calibration" in s_df: flat["df_calibration"] = s_df.get("df_calibration")
            if "df_fps" in s_df: flat["df_fps"] = s_df.get("df_fps")
            for df_k in ["pyr_scale", "levels", "winsize", "iterations", "poly_n", "poly_sigma",
                        "lower_mag_thresh", "upper_mag_thresh", "erosion", "supersat", "use_contours", "show_arrows",
                        "arrow_step", "arrow_scale", "visu_mode", "overlay_enabled", "overlay_alpha",
                        "heatmap_auto_range", "heatmap_vmin", "heatmap_vmax", "isolate_selection",
                        "lower_surface_thresh", "upper_surface_thresh", "show_boxes", "show_centroids"]:
                if df_k in s_df:
                    flat[f"df_{df_k}"] = s_df.get(df_k)
            if "both_vis_mode" in s_df:
                flat["both_vis_mode"] = s_df.get("both_vis_mode")

            # Drift Correction
            for drk in [
                "enable_drift_correction", "drift_angle_tol", "drift_mag_tol", "drift_min_corr_speed",
                "drift_use_hint", "drift_hint_preset", "drift_hint_angle", "drift_hint_cone",
                "drift_grid_mode", "drift_lock_live", "drift_use_ranges", "drift_active_ranges",
                "drift_show_vector", "drift_show_filtered", "drift_show_correction", "drift_show_fake_drifters",
                "drift_reuse_preprocessed", "drift_only_export"
            ]:
                if drk in s_drf:
                    flat[drk] = s_drf.get(drk)

            # Phototaxis
            for pik in [
                "emit_phototaxis_series", "pi_light_pos", "pi_preset",
                "pi_up_min", "pi_up_max", "pi_down_min", "pi_down_max", "pi_weight_speed",
            ]:
                if pik in s_pi:
                    flat[pik] = s_pi.get(pik)

            # Global
            for gk in ["active", "processing_mode", "x_axis_mode", "delta_t", "auto_export_csv", "auto_export_raw"]:
                if gk in s_glb:
                    flat[gk] = s_glb.get(gk)

        # Flat overrides from root
        for k, v in p.items():
            if not isinstance(v, dict):
                flat[k] = v

        tag_map = {
            "processing_mode": self.mode_tag,
            "active": self.active_tag,
            "enable_preprocessing": self.enable_preprocessing_tag,
            "monochrome_mode": self.monochrome_tag,
            "invert_image": self.invert_image_tag,
            "prep_blur": self.prep_blur_tag,
            "gamma": self.gamma_tag,
            "crop_type": self.crop_type_tag,
            "show_crop_outline": self.show_crop_outline_tag,
            "crop_radius": self.crop_radius_tag,
            "crop_width": self.crop_width_tag,
            "crop_size": self.crop_size_tag,
            "crop_height": self.crop_height_tag,
            "crop_use_abs_center": self.crop_use_abs_tag,
            "crop_center_x": self.crop_center_x_tag,
            "crop_center_y": self.crop_center_y_tag,
            "crop_offset_x": self.crop_offset_x_tag,
            "crop_offset_y": self.crop_offset_y_tag,
            "remove_static": self.remove_static_tag,
            "similarity_threshold": self.similarity_threshold_tag,
            "enable_focus_picking": self.enable_focus_picking_tag,
            "focus_metric": self.focus_metric_tag,
            "focus_window_size": self.focus_window_size_tag,
            "focus_output_map": self.focus_output_map_tag,
            "focus_vis_mode": self.focus_vis_mode_tag,
            "focus_threshold": self.focus_threshold_tag,
            "focus_alpha": self.focus_alpha_tag,
            "focus_isolate_selection": self.focus_isolate_selection_tag,
            "focus_use_for_processing": self.focus_use_for_processing_tag,
            "focus_filter_stage": self.focus_filter_stage_tag,

            "blur": self.blur_tag,
            "block_size": self.block_size_tag,
            "bin_thresh": self.bin_thresh_tag,
            "morph_close": self.morph_close_tag,
            "show_blobs": self.show_blobs_tag,
            "blob_alpha": self.blob_alpha_tag,

            "enable_contour_detection": self.enable_contour_tag,
            "lower_surface_thresh": self.lower_str_tag,
            "upper_surface_thresh": self.upper_str_tag,
            "show_boxes": self.show_boxes_tag,
            "show_centroids": self.show_centroids_tag,
            "isolate_selection": self.isolate_tag,
            "fill_contours": self.fill_contours_tag,
            "enable_watershed": self.enable_watershed_tag,
            "ws_peak_thresh": self.ws_peak_thresh_tag,
            "ws_show_lines": self.ws_show_lines_tag,

            "calibration": self.calibration_tag,
            "df_calibration": self.calibration_tag,
            "fps": self.fps_tag,
            "df_fps": self.fps_tag,
            "df_pyr_scale": self.df_pyr_scale_tag,
            "pyr_scale": self.df_pyr_scale_tag,
            "df_levels": self.df_levels_tag,
            "levels": self.df_levels_tag,
            "df_winsize": self.df_winsize_tag,
            "winsize": self.df_winsize_tag,
            "df_iterations": self.df_iterations_tag,
            "iterations": self.df_iterations_tag,
            "df_poly_n": self.df_poly_n_tag,
            "poly_n": self.df_poly_n_tag,
            "df_poly_sigma": self.df_poly_sigma_tag,
            "poly_sigma": self.df_poly_sigma_tag,
            "df_lower_mag_thresh": self.df_mag_thresh_tag,
            "lower_mag_thresh": self.df_mag_thresh_tag,
            "df_upper_mag_thresh": self.df_upper_mag_thresh_tag,
            "upper_mag_thresh": self.df_upper_mag_thresh_tag,
            "df_erosion": self.df_erosion_tag,
            "erosion": self.df_erosion_tag,
            "df_supersat": self.df_supersat_tag,
            "supersat": self.df_supersat_tag,
            "df_use_contours": self.df_use_contours_tag,
            "use_contours": self.df_use_contours_tag,
            "df_show_arrows": self.df_show_arrows_tag,
            "show_arrows": self.df_show_arrows_tag,
            "df_arrow_step": self.df_arrow_step_tag,
            "arrow_step": self.df_arrow_step_tag,
            "df_arrow_scale": self.df_arrow_scale_tag,
            "arrow_scale": self.df_arrow_scale_tag,
            "df_visu_mode": self.df_visu_mode_tag,
            "visu_mode": self.df_visu_mode_tag,
            "df_overlay_enabled": self.df_overlay_enabled_tag,
            "overlay_enabled": self.df_overlay_enabled_tag,
            "df_overlay_alpha": self.df_overlay_alpha_tag,
            "overlay_alpha": self.df_overlay_alpha_tag,
            "df_heatmap_auto_range": self.df_heatmap_auto_tag,
            "heatmap_auto_range": self.df_heatmap_auto_tag,
            "df_heatmap_vmin": self.df_heatmap_vmin_tag,
            "heatmap_vmin": self.df_heatmap_vmin_tag,
            "df_heatmap_vmax": self.df_heatmap_vmax_tag,
            "heatmap_vmax": self.df_heatmap_vmax_tag,
            "df_isolate_selection": self.df_isolate_tag,
            "df_lower_surface_thresh": self.df_lower_str_tag,
            "df_upper_surface_thresh": self.df_upper_str_tag,
            "df_show_boxes": self.df_show_boxes_tag,
            "df_show_centroids": self.df_show_centroids_tag,
            "both_vis_mode": self.both_vis_tag,

            "enable_tracking": self.enable_tracking_tag,
            "distance_function": self.dist_func_tag,
            "distance_threshold": self.dist_thresh_tag,
            "trail_length": self.trail_length_tag,
            "show_trail": self.show_trail_tag,
            "show_arrow": self.show_arrow_tag,
            "track_arrow_scale": self.track_arrow_scale_tag,
            "vector_mode": self.track_vector_mode_tag,
            "show_id": self.show_id_tag,
            "show_age": self.show_age_tag,
            "show_distance": self.show_distance_tag,
            "show_speed": self.show_speed_tag,
            "track_max_lost": self.track_max_lost_tag,
            "track_max_age": self.track_max_age_tag,
            "use_tracking_speed": self.use_tracking_speed_tag,
            "speed_window": self.speed_window_tag,
            "track_min_speed": self.track_min_speed_tag,
            "track_max_speed": self.track_max_speed_tag,

            "enable_drift_correction": self.enable_drift_tag,
            "drift_angle_tol": self.drift_angle_tol_tag,
            "drift_mag_tol": self.drift_mag_tol_tag,
            "drift_min_corr_speed": self.drift_min_corr_speed_tag,
            "drift_use_hint": self.drift_use_hint_tag,
            "drift_hint_preset": self.drift_hint_preset_tag,
            "drift_hint_angle": self.drift_hint_angle_tag,
            "drift_hint_cone": self.drift_hint_cone_tag,
            "drift_grid_mode": self.drift_grid_mode_tag,
            "drift_lock_live": self.drift_lock_live_tag,
            "drift_use_ranges": self.drift_use_ranges_tag,
            "drift_active_ranges": self.drift_active_ranges_tag,
            "drift_show_vector": self.drift_show_vector_tag,
            "drift_show_filtered": self.drift_show_filtered_tag,
            "drift_show_correction": self.drift_show_correction_tag,
            "drift_show_fake_drifters": self.drift_show_fake_drifters_tag,
            "drift_reuse_preprocessed": self.drift_reuse_preprocessed_tag,
            "drift_only_export": self.drift_only_export_tag,

            "emit_phototaxis_series": self.emit_pi_tag,
            "pi_light_pos": self.pi_light_pos_tag,
            "pi_preset": self.pi_preset_tag,
            "pi_up_min": self.pi_up_min_tag,
            "pi_up_max": self.pi_up_max_tag,
            "pi_down_min": self.pi_down_min_tag,
            "pi_down_max": self.pi_down_max_tag,
            "pi_weight_speed": self.pi_weight_speed_tag,

            "x_axis_mode": self.x_mode_tag,
            "delta_t": self.delta_t_tag,
            "auto_export_csv": self.auto_export_csv_tag,
            "auto_export_raw": self.auto_export_raw_tag
        }

        if "pi_light_pos" in flat:
            flat["pi_light_pos"] = normalize_pi_light_pos(flat["pi_light_pos"])
        if "pi_preset" in flat:
            flat["pi_preset"] = normalize_pi_preset(flat["pi_preset"])

        for key, tag in tag_map.items():
            if key in flat and dpg.does_item_exist(tag):
                try:
                    dpg.set_value(tag, flat[key])
                except Exception:
                    pass

        # Update visibility and callbacks
        if "enable_preprocessing" in flat:
            self._on_preprocessing_toggle(None, flat["enable_preprocessing"])
        if "crop_type" in flat:
            self._on_crop_type_changed(None, flat["crop_type"])
        if "crop_use_abs_center" in flat:
            self._on_crop_center_mode_changed(None, flat["crop_use_abs_center"])
        if "remove_static" in flat:
            self._on_remove_static_toggle(None, flat["remove_static"])
        if "enable_focus_picking" in flat:
            self._on_focus_picking_toggle(None, flat["enable_focus_picking"])
        if "processing_mode" in flat:
            self._on_mode_change_cb(None, flat["processing_mode"])
        if "visu_mode" in flat or "df_visu_mode" in flat:
            self._on_df_visu_mode_changed(None, flat.get("visu_mode", flat.get("df_visu_mode")))
        if "heatmap_auto_range" in flat or "df_heatmap_auto_range" in flat:
            self._on_df_hm_auto_toggled(None, flat.get("heatmap_auto_range", flat.get("df_heatmap_auto_range")))
        if "show_arrows" in flat or "df_show_arrows" in flat:
            self._on_df_show_arrows_toggled(None, flat.get("show_arrows", flat.get("df_show_arrows")))
        if "use_contours" in flat or "df_use_contours" in flat:
            self._on_df_contour_toggle_cb(None, flat.get("use_contours", flat.get("df_use_contours")))
        if "enable_watershed" in flat:
            self._on_watershed_toggle_cb(None, flat["enable_watershed"])
        if "enable_tracking" in flat:
            self._on_tracker_toggle_cb(None, flat["enable_tracking"])
        if "use_tracking_speed" in flat:
            self._on_tracking_speed_toggle_cb(None, flat["use_tracking_speed"])
        if "enable_drift_correction" in flat:
            self._on_drift_toggle_cb(None, flat["enable_drift_correction"])
        if "drift_reuse_preprocessed" in flat:
            self._on_drift_reuse_toggle_cb(None, flat["drift_reuse_preprocessed"])
        if "drift_use_hint" in flat:
            self._on_drift_hint_toggle_cb(None, flat["drift_use_hint"])
        if "drift_hint_preset" in flat:
            self._on_drift_hint_preset_cb(None, flat["drift_hint_preset"])
        if "drift_use_ranges" in flat:
            self._on_drift_ranges_toggle_cb(None, flat["drift_use_ranges"])
        if any(k in flat for k in ("pi_light_pos", "pi_preset", "pi_up_min", "pi_up_max", "pi_down_min", "pi_down_max", "emit_phototaxis_series", "pi_weight_speed")):
            if "pi_light_pos" in flat:
                self.pi_light_pos = normalize_pi_light_pos(flat["pi_light_pos"])
            if "pi_preset" in flat:
                self.pi_preset = normalize_pi_preset(flat["pi_preset"])
            preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
            if preset_bounds is not None:
                self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
            self._sync_pi_ui_widgets()
        if "x_axis_mode" in flat:
            self._on_x_mode_changed(None, flat["x_axis_mode"])

        self._update_param_cb(None, None)

    def _apply_preset_data(self, data: Dict[str, Any]) -> None:
        super()._apply_preset_data(data)
        self.pi_light_pos = normalize_pi_light_pos(self.pi_light_pos)
        self.pi_preset = normalize_pi_preset(self.pi_preset)
        preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
        if preset_bounds is not None:
            self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
        if dpg.does_item_exist(self.emit_pi_tag):
            dpg.set_value(self.emit_pi_tag, bool(self.emit_phototaxis_series))
        if dpg.does_item_exist(self.pi_weight_speed_tag):
            dpg.set_value(self.pi_weight_speed_tag, bool(self.pi_weight_speed))
        self._sync_pi_ui_widgets()

    # ── Widget callbacks ───────────────────────────────────────────────
    def _on_preprocessing_toggle(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.preprocessing_group_tag, app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_crop_type_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.crop_circular_group_tag, app_data == "Circular")
        self._show(self.crop_square_group_tag, app_data == "Square")
        self._show(self.crop_rect_group_tag, app_data == "Rectangle")
        self._update_param_cb(sender, app_data, user_data)

    def _on_crop_center_mode_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.crop_abs_center_group_tag, app_data)
        self._show(self.crop_rel_center_group_tag, not app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_remove_static_toggle(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.static_removal_group_tag, app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _capture_r1_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._last_r1_trigger += 1
        self._update_param_cb(sender, app_data, user_data)

    def _capture_r2_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._last_r2_trigger += 1
        self._update_param_cb(sender, app_data, user_data)

    def _clear_ref_frames(self):
        """Drops both static-removal reference frames and resets their status labels."""
        self._last_clear_ref_trigger += 1
        for attr in ("_ref_frame_1", "_ref_frame_2"):
            if hasattr(self.processor, attr):
                setattr(self.processor, attr, None)
        self._set_ref_status(self.r1_status_tag, False, 1)
        self._set_ref_status(self.r2_status_tag, False, 2)

    def _clear_ref_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._clear_ref_frames()
        self._update_param_cb(sender, app_data, user_data)

    def _on_processing_toggle(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.processing_container_tag, app_data)
        if not app_data:
            self._skip_first_live_frame = True
            for tag in (self.metric_text_tag, self.count_text_tag, self.motility_text_tag, self.pi_text_tag):
                self._set_text(tag, "--")
        self._update_param_cb(sender, app_data, user_data)

    def _on_focus_picking_toggle(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.focus_picking_group_tag, app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_x_mode_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.x_axis_mode = app_data
        self._show(self.delta_t_group_tag, app_data == "Fixed ΔT")
        self._reemit_all_histories()

    def _on_delta_t_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.delta_t = app_data
        self._reemit_all_histories()

    def _reemit_all_histories(self):
        with self._buffers_lock:
            uuids = list(self.samples_buffers)
        for s_uuid in uuids:
            self._emit_full_history(s_uuid)

    def _on_auto_export_csv_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.auto_export_csv = app_data

    def _on_auto_export_raw_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.auto_export_raw = app_data

    def _build_x_axis(self, raw_x: list[float], has_frame_idx: bool = False) -> list[float]:
        mode = self.x_axis_mode
        n = len(raw_x)
        dt = float(self.delta_t) if self.delta_t else 1.0
        if has_frame_idx:
            try:
                num_x = [float(x) for x in raw_x]
            except (ValueError, TypeError):
                num_x = [float(i) for i in range(n)]
            return num_x if mode == "Index" else [x * dt for x in num_x]
        if mode == "Index":
            return [float(i) for i in range(n)]
        if mode == "Fixed ΔT":
            return [float(i) * dt for i in range(n)]
        try:  # "Timestamp"
            return [float(x) for x in raw_x]
        except (ValueError, TypeError):
            return [float(i) * dt for i in range(n)]

    def _send_sample(self, sample: dict, conn_keys, swallow: bool = True):
        """Sends `sample` once to every module connected to any of `conn_keys`."""
        sent = set()
        for key in conn_keys:
            for module in self.connections.get(key, []):
                if module in sent:
                    continue
                sent.add(module)
                try:
                    module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)
                except Exception as e:
                    if not swallow:
                        logger.error(f"Error emitting sample history to module on {key}: {e}")

    def _clear_downstream_plots(self):
        target_keys = ("Result", "Count", "Motility", "Phototaxis")
        with self._buffers_lock:
            items = [(s_uuid, buff["name"]) for s_uuid, buff in self.samples_buffers.items()]
        for s_uuid, name in items:
            for action in ("select", "clear"):
                self._send_sample({"name": name, "uuid": s_uuid, "x": [], "y": [], "action": action},
                                  target_keys)
        self._send_sample({"name": self.label, "uuid": str(self.UUID), "x": [], "y": [], "action": "clear"},
                          target_keys)

    def _accumulate_point(self, s_uuid: str, name: str, val: float, timestamp: float, frame_idx: int | None = None):
        with self._buffers_lock:
            buff = self.samples_buffers.setdefault(
                s_uuid, {"x": [], "y": [], "name": name, "t0": None, "has_frame_idx": False})
            if buff["t0"] is None:
                buff["t0"] = timestamp

            # Ensure x and y stay strictly synchronized in length
            if len(buff["x"]) != len(buff["y"]):
                min_len = min(len(buff["x"]), len(buff["y"]))
                buff["x"] = list(buff["x"][:min_len])
                buff["y"] = list(buff["y"][:min_len])

            if _is_num(frame_idx):
                buff["has_frame_idx"] = True
                f_val = int(frame_idx)
                # Overwrite if frame_idx is already present, otherwise insert keeping the buffer sorted
                i = bisect_left(buff["x"], f_val)
                if i < len(buff["x"]) and buff["x"][i] == f_val:
                    buff["y"][i] = float(val)
                else:
                    buff["x"].insert(i, f_val)
                    buff["y"].insert(i, float(val))
            else:
                # Store relative time to t0
                buff["x"].append(float(timestamp - buff["t0"]))
                buff["y"].append(float(val))

    def _emit_full_history(self, s_uuid: str, conn_keys: Optional[List[str]] = None):
        with self._buffers_lock:
            buff = self.samples_buffers.get(s_uuid)
            if not buff:
                return
            raw_x_copy = list(buff["x"])
            y_copy = list(buff["y"])
            name = buff["name"]
            has_f_idx = buff.get("has_frame_idx", False)
        x_scaled = self._build_x_axis(raw_x_copy, has_frame_idx=has_f_idx)
        out_sample = {"name": name, "uuid": s_uuid, "x": x_scaled, "y": y_copy, "action": "select"}
        self._send_sample(out_sample, conn_keys or ["Result"], swallow=False)

    def _on_contour_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Show/hide contour controls depending on enable_contour_detection."""
        self._show(self.contour_group_tag, dpg.get_value(self.enable_contour_tag))
        self._update_param_cb(sender, app_data, user_data)

    def _on_watershed_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Show/hide watershed controls depending on enable_watershed."""
        self._show(self.watershed_group_tag, dpg.get_value(self.enable_watershed_tag))
        self._update_param_cb(sender, app_data, user_data)

    def _on_tracker_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Show/hide tracker controls depending on enable_tracking."""
        self._show(self.tracker_group_tag, dpg.get_value(self.enable_tracking_tag))
        self._update_param_cb(sender, app_data, user_data)

    def _on_drift_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Show/hide drift correction controls depending on enable_drift_correction."""
        enabled = bool(dpg.get_value(self.enable_drift_tag))
        self._show(self.drift_group_tag, enabled)
        if not enabled:
            self._set_text(self.drift_status_text_tag, "--")
        else:
            self._resolve_live_drift_for_source(self._source_name)
        self._update_param_cb(sender, app_data, user_data)

    def _on_drift_reuse_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Resolve preprocessed drift file when toggling 'Reuse drift, if present' and update UI visibility."""
        self.drift_reuse_preprocessed = bool(self._val(self.drift_reuse_preprocessed_tag, self.drift_reuse_preprocessed))
        if self.drift_reuse_preprocessed:
            self._resolve_live_drift_for_source(self._source_name)
        else:
            self._update_drift_reuse_ui()
        self._update_param_cb(sender, app_data, user_data)

    def _update_drift_reuse_ui(self):
        """
        When 'Reuse drift, if present' is checked AND a drift file is found for the current video,
        hides all drift calculation/export menus (`drift_calc_params_group_tag`) and displays a
        status confirmation badge.
        """
        reuse_checked = bool(self._val(self.drift_reuse_preprocessed_tag, self.drift_reuse_preprocessed))
        drift_found = reuse_checked and (self._live_preprocessed_drift_map is not None)
        if drift_found:
            # Ensure a hidden 'Only export drift' checkbox from Pass 1 cannot stay active
            self.drift_only_export = False
            if dpg.does_item_exist(self.drift_only_export_tag):
                try:
                    dpg.set_value(self.drift_only_export_tag, False)
                except Exception:
                    pass
        self._show(self.drift_calc_params_group_tag, not drift_found)

        if dpg.does_item_exist(self.drift_reuse_status_tag):
            if not reuse_checked:
                self._show(self.drift_reuse_status_tag, False)
            elif drift_found:
                fname = os.path.basename(self._live_preprocessed_drift_path or "drift")
                gs = self._live_preprocessed_drift_map.get("grid_shape", (1, 1))
                n_fr = len(self._live_preprocessed_drift_map.get("frames", {}))
                try:
                    dpg.configure_item(
                        self.drift_reuse_status_tag,
                        default_value=f"✓ Found: {fname} [{gs[0]}x{gs[1]}, {n_fr}f]",
                        color=(120, 235, 150),
                        show=True,
                    )
                except Exception:
                    pass
            else:
                try:
                    dpg.configure_item(
                        self.drift_reuse_status_tag,
                        default_value="No drift file found (live calc active)",
                        color=(255, 185, 100),
                        show=True,
                    )
                except Exception:
                    pass

    def _on_drift_hint_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Show/hide global direction hint settings."""
        self._show(self.drift_hint_group_tag, dpg.get_value(self.drift_use_hint_tag))
        self._update_param_cb(sender, app_data, user_data)

    def _on_drift_hint_preset_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Show/hide custom hint angle input depending on Hint Direction preset."""
        preset = dpg.get_value(self.drift_hint_preset_tag)
        self._show(self.drift_hint_angle_group_tag, preset == "Custom Angle")
        self._update_param_cb(sender, app_data, user_data)

    def _on_drift_ranges_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Show/hide active frame ranges input."""
        self._show(self.drift_ranges_group_tag, dpg.get_value(self.drift_use_ranges_tag))
        self._update_param_cb(sender, app_data, user_data)

    def _sync_pi_ui_widgets(self) -> None:
        t_arrow, a_arrow = pi_light_arrows(self.pi_light_pos)
        if dpg.does_item_exist(self.pi_light_pos_tag):
            dpg.set_value(self.pi_light_pos_tag, self.pi_light_pos)
        if dpg.does_item_exist(self.pi_preset_tag):
            dpg.set_value(self.pi_preset_tag, self.pi_preset)
        if dpg.does_item_exist(self.pi_toward_lbl_tag):
            dpg.set_value(self.pi_toward_lbl_tag, f"Toward ({t_arrow}):")
        if dpg.does_item_exist(self.pi_away_lbl_tag):
            dpg.set_value(self.pi_away_lbl_tag, f"Away ({a_arrow}):  ")
        for tag, val in (
            (self.pi_up_min_tag, self.pi_up_min),
            (self.pi_up_max_tag, self.pi_up_max),
            (self.pi_down_min_tag, self.pi_down_min),
            (self.pi_down_max_tag, self.pi_down_max),
        ):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, float(val))

    def _on_pi_light_pos_cb(self, sender=None, app_data=None, *args, **kwargs) -> None:
        self.pi_light_pos = normalize_pi_light_pos(
            app_data if app_data is not None else self._val(self.pi_light_pos_tag, self.pi_light_pos)
        )
        preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
        if preset_bounds is not None:
            self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
        self._sync_pi_ui_widgets()
        self._on_pi_setting_cb()

    def _on_pi_preset_cb(self, sender=None, app_data=None, *args, **kwargs) -> None:
        self.pi_preset = normalize_pi_preset(
            app_data if app_data is not None else self._val(self.pi_preset_tag, self.pi_preset)
        )
        self.pi_light_pos = normalize_pi_light_pos(self._val(self.pi_light_pos_tag, self.pi_light_pos))
        preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
        if preset_bounds is not None:
            self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
        self._sync_pi_ui_widgets()
        self._on_pi_setting_cb()

    def _on_pi_custom_cb(self, *args, **kwargs) -> None:
        self.pi_preset = "Custom Range"
        if dpg.does_item_exist(self.pi_preset_tag):
            dpg.set_value(self.pi_preset_tag, "Custom Range")
        self._on_pi_setting_cb()

    def _on_pi_setting_cb(self, *args, **kwargs) -> None:
        self.emit_phototaxis_series = bool(self._val(self.emit_pi_tag, self.emit_phototaxis_series))
        self.pi_light_pos = normalize_pi_light_pos(self._val(self.pi_light_pos_tag, self.pi_light_pos))
        self.pi_preset = normalize_pi_preset(self._val(self.pi_preset_tag, self.pi_preset))
        self.pi_up_min = float(self._val(self.pi_up_min_tag, self.pi_up_min))
        self.pi_up_max = float(self._val(self.pi_up_max_tag, self.pi_up_max))
        self.pi_down_min = float(self._val(self.pi_down_min_tag, self.pi_down_min))
        self.pi_down_max = float(self._val(self.pi_down_max_tag, self.pi_down_max))
        self.pi_weight_speed = bool(self._val(self.pi_weight_speed_tag, self.pi_weight_speed))

    def _format_pi_value(self, pi_val: float, light_pos: Optional[str] = None) -> str:
        if not np.isfinite(pi_val):
            return "--"
        t_arrow, a_arrow = pi_light_arrows(light_pos or self.pi_light_pos)
        if pi_val > 0.05:
            return f"{pi_val:+.2f} (Toward {t_arrow})"
        if pi_val < -0.05:
            return f"{pi_val:+.2f} (Away {a_arrow})"
        return f"{pi_val:+.2f} (Neutral)"

    def _on_df_visu_mode_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.df_heatmap_settings_group_tag, app_data == "Speed Heatmap")
        self._update_param_cb(sender, app_data, user_data)

    def _on_df_contour_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.df_contour_group_tag, dpg.get_value(self.df_use_contours_tag))
        self._update_param_cb(sender, app_data, user_data)

    def _on_df_hm_auto_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.df_heatmap_custom_group_tag, not app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_df_show_arrows_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self._show(self.df_arrows_group_tag, app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_df_winsize_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        val = int(app_data)
        dpg.set_value(self.df_winsize_tag, val + 1 if val % 2 == 0 else val)
        self._update_param_cb(sender, app_data, user_data)

    def _on_tracking_speed_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._update_param_cb(sender, app_data, user_data)  # also refreshes the metric label

    def _on_mode_change_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        mode = dpg.get_value(self.mode_tag)
        self._show(self.mode_contour_group_tag, mode in ("contour", "both"))
        self._show(self.df_group_tag, mode == "denseflow")
        self._show(self.both_group_tag, mode == "both")
        try:
            self.processor._reset_denseflow_state()
        except AttributeError:
            pass
        self._update_param_cb(sender, app_data, user_data)  # also refreshes the metric label

    def _reset_all_cb(self, sender=None, app_data=None, user_data=None, *args, **kwargs):
        """Full reset of tracker, downstream plots, internal buffers, and UI metric displays to 0."""
        self._reset_trigger_count = getattr(self, '_reset_trigger_count', 0) + 1
        self._clear_ref_frames()
        if bool(self._val(self.drift_reuse_preprocessed_tag, self.drift_reuse_preprocessed)):
            self._resolve_live_drift_for_source(self._source_name)

        self.processor.update_params(
            reset_trigger=self._reset_trigger_count,
            clear_ref_trigger=self._last_clear_ref_trigger,
            _preprocessed_drift_map=self._live_preprocessed_drift_map if bool(self._val(self.drift_reuse_preprocessed_tag, False)) else None,
            _preprocessed_drift_path=self._live_preprocessed_drift_path if bool(self._val(self.drift_reuse_preprocessed_tag, False)) else None,
            _video_path=self._last_video_path,
        )
        self._worker_drift_synced_key = (
            self._live_preprocessed_drift_path,
            bool(self._val(self.drift_reuse_preprocessed_tag, False)),
            self._live_preprocessed_drift_map is not None,
        )
        try:
            self.processor._reset_denseflow_state()
        except AttributeError:
            pass
        self._clear_downstream_plots()
        self.samples_buffers.clear()
        self._skip_first_live_frame = True

        # Reset UI display text to 0 / defaults
        self._set_text(self.metric_text_tag, "0.00")
        self._set_text(self.count_text_tag, "0")
        self._set_text(self.motility_text_tag, "--")
        self._set_text(self.pi_text_tag, "--")
        self._set_text(self.focus_score_tag, "--")
        self._set_text(self.drift_status_text_tag, "--")

    def _collect_params(self) -> dict:
        """Snapshot of the current widget values as processor params (main thread only)."""
        g = self._val
        s_df = self.settings.get('denseflow', {})
        mode = dpg.get_value(self.mode_tag)

        calib = g(self.calibration_tag, self.calibration, float)
        fps = g(self.fps_tag, self.fps, float)
        self.calibration, self.fps = calib, fps

        # Crop width/height come from the single "size" widget in Square mode
        if g(self.crop_type_tag) == "Square" and dpg.does_item_exist(self.crop_size_tag):
            crop_w = crop_h = int(dpg.get_value(self.crop_size_tag))
        else:
            crop_w = g(self.crop_width_tag, self.crop_width, int)
            crop_h = g(self.crop_height_tag, self.crop_height, int)

        boxes_tag, centroids_tag = ((self.df_show_boxes_tag, self.df_show_centroids_tag) if mode == "denseflow"
                                    else (self.show_boxes_tag, self.show_centroids_tag))

        return {
            "processing_mode": mode,

            # Preprocessing & Focus Picking
            "enable_preprocessing": g(self.enable_preprocessing_tag, True),
            "monochrome_mode": g(self.monochrome_tag, "Grayscale"),
            "invert_image": g(self.invert_image_tag, False),
            "prep_blur": g(self.prep_blur_tag, 0, int),
            "gamma": g(self.gamma_tag, self.gamma, float),
            "crop_type": g(self.crop_type_tag, self.crop_type),
            "crop_radius": g(self.crop_radius_tag, self.crop_radius, int),
            "crop_width": crop_w,
            "crop_height": crop_h,
            "show_crop_outline": g(self.show_crop_outline_tag, self.show_crop_outline),
            "crop_use_abs_center": g(self.crop_use_abs_tag, self.crop_use_abs_center),
            "crop_center_x": g(self.crop_center_x_tag, self.crop_center_x, int),
            "crop_center_y": g(self.crop_center_y_tag, self.crop_center_y, int),
            "crop_offset_x": g(self.crop_offset_x_tag, self.crop_offset_x, int),
            "crop_offset_y": g(self.crop_offset_y_tag, self.crop_offset_y, int),
            "remove_static": g(self.remove_static_tag, self.remove_static),
            "similarity_threshold": g(self.similarity_threshold_tag, self.similarity_threshold, int),
            "enable_focus_picking": g(self.enable_focus_picking_tag, False),
            "focus_metric": g(self.focus_metric_tag, "Tenengrad (Sobel)"),
            "focus_window_size": g(self.focus_window_size_tag, 15, int),
            "focus_output_map": g(self.focus_output_map_tag, True),
            "focus_vis_mode": g(self.focus_vis_mode_tag, "Heatmap (Turbo)"),
            "focus_threshold": g(self.focus_threshold_tag, 40, int),
            "focus_alpha": g(self.focus_alpha_tag, 0.6, float),
            "focus_isolate_selection": g(self.focus_isolate_selection_tag, False),
            "focus_use_for_processing": g(self.focus_use_for_processing_tag, False),
            "focus_filter_stage": g(self.focus_filter_stage_tag, "Pre-processing"),

            # Calibration & Dense Flow
            "df_calibration": calib,
            "calibration": calib,
            "df_fps": fps,
            "fps": fps,
            "df_pyr_scale": g(self.df_pyr_scale_tag, s_df.get("pyr_scale", 0.5), float),
            "df_levels": g(self.df_levels_tag, s_df.get("levels", 2), int),
            "df_winsize": g(self.df_winsize_tag, s_df.get("winsize", 15), int),
            "df_iterations": g(self.df_iterations_tag, s_df.get("iterations", 3), int),
            "df_poly_n": g(self.df_poly_n_tag, s_df.get("poly_n", 5), int),
            "df_poly_sigma": g(self.df_poly_sigma_tag, s_df.get("poly_sigma", 1.2), float),
            "df_lower_mag_thresh": g(self.df_mag_thresh_tag),
            "df_upper_mag_thresh": g(self.df_upper_mag_thresh_tag, s_df.get("upper_mag_thresh", 0.0), float),
            "df_erosion": g(self.df_erosion_tag),
            "df_visu_mode": g(self.df_visu_mode_tag),
            "df_overlay_enabled": g(self.df_overlay_enabled_tag),
            "df_overlay_alpha": g(self.df_overlay_alpha_tag),
            "df_heatmap_auto_range": g(self.df_heatmap_auto_tag),
            "df_heatmap_vmin": g(self.df_heatmap_vmin_tag),
            "df_heatmap_vmax": g(self.df_heatmap_vmax_tag),
            "df_supersat": g(self.df_supersat_tag),
            "df_show_arrows": g(self.df_show_arrows_tag),
            "df_arrow_step": g(self.df_arrow_step_tag),
            "df_arrow_scale": g(self.df_arrow_scale_tag),
            "df_use_contours": g(self.df_use_contours_tag),
            "df_isolate_selection": g(self.df_isolate_tag),
            "df_lower_surface_thresh": g(self.df_lower_str_tag),
            "df_upper_surface_thresh": g(self.df_upper_str_tag),

            # Binarize
            "blur": g(self.blur_tag),
            "block_size": g(self.block_size_tag),
            "bin_thresh": g(self.bin_thresh_tag),
            "morph_close": g(self.morph_close_tag),

            # Contours
            "enable_contour_detection": g(self.enable_contour_tag),
            "fill_contours": g(self.fill_contours_tag),
            "lower_surface_thresh": g(self.lower_str_tag),
            "upper_surface_thresh": g(self.upper_str_tag),
            "show_blobs": g(self.show_blobs_tag),
            "blob_alpha": g(self.blob_alpha_tag, 0.5, float),
            "isolate_selection": g(self.isolate_tag),
            "show_boxes": g(boxes_tag, False),
            "show_centroids": g(centroids_tag, False),
            "enable_watershed": g(self.enable_watershed_tag, False),
            "ws_peak_thresh": g(self.ws_peak_thresh_tag, 40),
            "ws_show_lines": g(self.ws_show_lines_tag, True),

            # Tracker
            "enable_tracking": g(self.enable_tracking_tag, False),
            "distance_function": g(self.dist_func_tag, "euclidean"),
            "distance_threshold": g(self.dist_thresh_tag, 30),
            "trail_length": g(self.trail_length_tag, 30, int),
            "show_trail": g(self.show_trail_tag, False),
            "show_arrow": g(self.show_arrow_tag, False),
            "arrow_scale": g(self.track_arrow_scale_tag, 3.0, float),
            "vector_mode": g(self.track_vector_mode_tag, "Last 2 Points"),
            "show_id": g(self.show_id_tag, False),
            "show_age": g(self.show_age_tag, False),
            "show_distance": g(self.show_distance_tag, False),
            "show_speed": g(self.show_speed_tag, False),
            "track_max_lost": g(self.track_max_lost_tag, 15, int),
            "track_max_age": g(self.track_max_age_tag, 0, int),
            "use_tracking_speed": g(self.use_tracking_speed_tag, False, bool),
            "speed_window": g(self.speed_window_tag, 5, int),
            "track_min_speed": g(self.track_min_speed_tag, 0.0, float),
            "track_max_speed": g(self.track_max_speed_tag, 0.0, float),

            # Drift Correction
            "enable_drift_correction": g(self.enable_drift_tag, False, bool),
            "drift_angle_tol": g(self.drift_angle_tol_tag, 15.0, float),
            "drift_mag_tol": g(self.drift_mag_tol_tag, 25.0, float),
            "drift_min_corr_speed": g(self.drift_min_corr_speed_tag, 0.0, float),
            "drift_use_hint": g(self.drift_use_hint_tag, False, bool),
            "drift_hint_preset": g(self.drift_hint_preset_tag, "Down (90°)", str),
            "drift_hint_angle": g(self.drift_hint_angle_tag, 90.0, float),
            "drift_hint_cone": g(self.drift_hint_cone_tag, 45.0, float),
            "drift_grid_mode": g(self.drift_grid_mode_tag, "Full (1x1)", str),
            "drift_lock_live": g(self.drift_lock_live_tag, False, bool),
            "drift_use_ranges": g(self.drift_use_ranges_tag, False, bool),
            "drift_active_ranges": g(self.drift_active_ranges_tag, "[0-500],[1000-1500]", str),
            "drift_show_vector": g(self.drift_show_vector_tag, True, bool),
            "drift_show_filtered": g(self.drift_show_filtered_tag, False, bool),
            "drift_show_correction": g(self.drift_show_correction_tag, False, bool),
            "drift_show_fake_drifters": g(self.drift_show_fake_drifters_tag, False, bool),
            "drift_reuse_preprocessed": g(self.drift_reuse_preprocessed_tag, False, bool),
            "drift_only_export": (
                False
                if (g(self.drift_reuse_preprocessed_tag, False, bool) and self._live_preprocessed_drift_map is not None)
                else g(self.drift_only_export_tag, False, bool)
            ),
            "_preprocessed_drift_map": self._live_preprocessed_drift_map if g(self.drift_reuse_preprocessed_tag, False, bool) else None,
            "_preprocessed_drift_path": self._live_preprocessed_drift_path if g(self.drift_reuse_preprocessed_tag, False, bool) else None,
            "_video_path": self._last_video_path,

            # Phototaxis
            "emit_phototaxis_series": g(self.emit_pi_tag, self.emit_phototaxis_series, bool),
            "pi_light_pos": normalize_pi_light_pos(g(self.pi_light_pos_tag, self.pi_light_pos, str)),
            "pi_preset": normalize_pi_preset(g(self.pi_preset_tag, self.pi_preset, str)),
            "pi_up_min": g(self.pi_up_min_tag, self.pi_up_min, float),
            "pi_up_max": g(self.pi_up_max_tag, self.pi_up_max, float),
            "pi_down_min": g(self.pi_down_min_tag, self.pi_down_min, float),
            "pi_down_max": g(self.pi_down_max_tag, self.pi_down_max, float),
            "pi_weight_speed": g(self.pi_weight_speed_tag, self.pi_weight_speed, bool),

            "both_vis_mode": g(self.both_vis_tag),
            "active": g(self.active_tag),
            "collect_raw_speeds": len(self.connections.get("Raw Speeds", [])) > 0 or len(self.connections.get("CMD", [])) > 0,
        }

    def _resolve_live_drift_for_source(self, source_name: str):
        """Resolves the video path for `source_name` (from Video Reader or last batch path)
        and loads `<video>_drift.csv` (or `.npz` fallback) if present so Live mode can reuse preprocessed drift."""
        video_path = ""
        src_base = os.path.basename(source_name) if source_name else ""
        src_stem = os.path.splitext(src_base)[0] if src_base else ""

        def _matches_source(cand_path: str, cand_name: str = "", cand_stem: str = "") -> bool:
            if not source_name:
                return True
            c_base = cand_name or os.path.basename(cand_path)
            c_stem = cand_stem or os.path.splitext(c_base)[0]
            return src_base == c_base or src_stem == c_stem

        if source_name and os.path.isfile(source_name):
            video_path = source_name
        else:
            try:
                from core.module_registry import get_registered_modules
                reg = get_registered_modules()
                modules_iter = reg.values() if isinstance(reg, dict) else reg
                fallback_selected = ""
                for mod in modules_iter:
                    last_sel = getattr(mod, "last_video_selected", None)
                    if isinstance(last_sel, str) and last_sel and os.path.isfile(last_sel):
                        if not fallback_selected:
                            fallback_selected = last_sel
                        if _matches_source(last_sel):
                            video_path = last_sel
                            break
                    for attr in ("video_list", "videos"):
                        vlist = getattr(mod, attr, None)
                        if isinstance(vlist, (list, tuple)):
                            for p in vlist:
                                if isinstance(p, dict):
                                    c_path = p.get("path", "")
                                    if isinstance(c_path, str) and c_path and os.path.isfile(c_path):
                                        if _matches_source(c_path, str(p.get("name", "")), str(p.get("basename", ""))):
                                            video_path = c_path
                                            break
                                elif isinstance(p, str) and p and os.path.isfile(p):
                                    if _matches_source(p):
                                        video_path = p
                                        break
                        if video_path:
                            break
                    if video_path:
                        break
                if not video_path and self._last_video_path and os.path.isfile(self._last_video_path):
                    if _matches_source(self._last_video_path):
                        video_path = self._last_video_path
                if not video_path and fallback_selected:
                    video_path = fallback_selected
            except Exception:
                if self._last_video_path and os.path.isfile(self._last_video_path):
                    video_path = self._last_video_path

        if video_path:
            self._last_video_path = video_path
            d_path, d_map = DriftProcessor.find_and_load_drift_for_video(video_path)
            self._live_preprocessed_drift_path = d_path
            self._live_preprocessed_drift_map = d_map
        else:
            self._live_preprocessed_drift_path = None
            self._live_preprocessed_drift_map = None

        self._update_drift_reuse_ui()

    def _update_param_cb(self, sender, app_data, user_data=None, *args):
        """Callback to push UI changes to the processor."""
        self.x_axis_mode = self._val(self.x_mode_tag, self.x_axis_mode)
        self.delta_t = self._val(self.delta_t_tag, self.delta_t)
        self.drift_reuse_preprocessed = bool(self._val(self.drift_reuse_preprocessed_tag, self.drift_reuse_preprocessed))
        self.drift_only_export = bool(self._val(self.drift_only_export_tag, self.drift_only_export))
        if self.drift_reuse_preprocessed:
            self._resolve_live_drift_for_source(self._source_name)
        else:
            self._update_drift_reuse_ui()

        params = self._collect_params()
        dpg.set_value(self.metric_label_tag, self._metric_label(
            params["processing_mode"], params["use_tracking_speed"] and params["enable_tracking"]))
        params.update(
            capture_r1_trigger=self._last_r1_trigger,
            capture_r2_trigger=self._last_r2_trigger,
            clear_ref_trigger=self._last_clear_ref_trigger,
        )
        self.processor.update_params(**params)
        self._worker_drift_synced_key = (
            self._live_preprocessed_drift_path,
            bool(params.get("drift_reuse_preprocessed", False)),
            self._live_preprocessed_drift_map is not None,
        )
        if self._last_input_data is not None and self._monitor_thread_running and not self._batch_running:
            self.processor.clear()
            self.processor.submit(self._last_input_data)

    def _get_flat_params(self) -> dict:
        """Snapshot of current UI params (called from main thread only)."""
        params = self._collect_params()
        params["_ref_frame_1"] = getattr(self.processor, "_ref_frame_1", None)
        params["_ref_frame_2"] = getattr(self.processor, "_ref_frame_2", None)
        return params

    def update_permission(self) -> None:
        """Update UI permissions based on application mode."""
        from core.app_state import app_state

        is_user = app_state.mode == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def on_close(self) -> None:
        """Called when module window is closed or application shuts down."""
        self._monitor_thread_running = False
        if getattr(self, "_monitor_thread", None) is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        if getattr(self, "processor", None) is not None:
            self.processor.stop()
        super().on_close()

    def close(self) -> None:
        """Close window and unregister."""
        self.on_close()
        super().close()

    def destroy(self) -> None:
        """Destroy window and release resources."""
        self.close()

    def is_ready(self) -> bool:
        if getattr(self, "processor", None) is not None:
            return self.processor.is_ready()
        return False

    @staticmethod
    def _parse_frame_payload(data):
        """Extracts (frame, name, frame_idx) from a raw frame / (frame, name|idx[, name|idx]) payload."""
        frame, name, frame_idx = data, "", None
        if isinstance(data, (tuple, list)):
            frame = data[0] if len(data) else None
            if len(data) == 2:
                second = data[1]
                if isinstance(second, str):
                    name = second
                elif _is_num(second):
                    frame_idx = int(second)
                else:
                    name = str(second)
            elif len(data) >= 3:
                a, b = data[1], data[2]
                if isinstance(a, str):
                    name = a
                    if _is_num(b):
                        frame_idx = int(b)
                elif isinstance(b, str):
                    name = b
                    if _is_num(a):
                        frame_idx = int(a)
                else:
                    if _is_num(a):
                        frame_idx = int(a)
                    name = str(b)
        return frame, name, frame_idx

    def _reset_series(self):
        """Clears downstream plots and local sample buffers (new source / new run)."""
        self._clear_downstream_plots()
        with self._buffers_lock:
            self.samples_buffers.clear()
        self._skip_first_live_frame = True
        try:
            self.processor._reset_denseflow_state()
        except AttributeError:
            pass

    def input_cb(self, *args, **kwargs):
        data      = kwargs.get("data")      if "data"      in kwargs else (args[0] if args else None)
        data_type = kwargs.get("data_type") if "data_type" in kwargs else None

        is_start_trigger = False
        if data_type == IOTypes.TRIGGER:
            is_start_trigger = not (isinstance(data, str) and data.lower() in ("stop", "done", "aborted", "killed"))
        elif isinstance(data, str) and data.upper() == "START":
            is_start_trigger = True

        if is_start_trigger:
            self._reset_series()
            if isinstance(data, str) and data.upper() == "START":
                self._reset_all_cb()
            return True

        # ── FILE_PATH: launch parallel batch processing ────────────────
        if data_type == IOTypes.FILE_PATH:
            path, name = data if (isinstance(data, tuple) and len(data) == 2) else (data, "")
            self._last_input_data = None
            self.processor.clear()
            self._reset_series()
            if path and not self._batch_running:
                # Read params & mode in main thread BEFORE launching background thread
                threading.Thread(target=self._batch_process_video,
                                 args=(path, name, self._get_flat_params()), daemon=True).start()
            return True

        # ── FRAME (live mode) ─────────────────────────────────────────
        if self._batch_running:
            return False  # Locked during batch

        frame, name, frame_idx = self._parse_frame_payload(data)
        reuse_on = bool(self._val(self.drift_reuse_preprocessed_tag, False))
        source_changed = bool(name and self._source_name != name)
        if source_changed:
            self._source_name = name
            self._resolve_live_drift_for_source(name)
            self._reset_series()
        elif reuse_on and (self._live_preprocessed_drift_map is None or frame_idx == 0):
            self._resolve_live_drift_for_source(self._source_name or name)

        expected_sync_key = (
            self._live_preprocessed_drift_path,
            reuse_on,
            self._live_preprocessed_drift_map is not None,
        )
        if getattr(self, "_worker_drift_synced_key", None) != expected_sync_key or source_changed:
            self.processor.update_params(
                drift_reuse_preprocessed=reuse_on,
                drift_only_export=False if (reuse_on and self._live_preprocessed_drift_map is not None) else bool(self._val(self.drift_only_export_tag, False)),
                _preprocessed_drift_map=self._live_preprocessed_drift_map if reuse_on else None,
                _preprocessed_drift_path=self._live_preprocessed_drift_path if reuse_on else None,
                _video_path=self._last_video_path,
            )
            self._worker_drift_synced_key = expected_sync_key

        if not isinstance(frame, np.ndarray):
            return False
        self._last_input_data = (frame, frame_idx) if frame_idx is not None else frame
        return self.processor.submit(self._last_input_data)

    # ── Batch processing ───────────────────────────────────────────────
    def _set_batch_lock(self, locked: bool):
        """Hide main UI and show progress UI during batch."""
        self._batch_running = locked
        try:
            self._show(self.main_settings_group_tag, not locked)
            self._show(self.batch_inputs_tag, not locked)
            self._show(self.batch_progress_group_tag, locked)
        except Exception:
            pass

    def _emit_trigger(self, video_path):
        """Emit completion trigger so downstream queues are never blocked."""
        for module in self.connections.get("Trigger", []):
            try:
                module.input_cb(data=video_path, data_type=IOTypes.TRIGGER)
            except Exception as e:
                logger.error(f"CounterTracker failed to emit Trigger: {e}")

    def _emit_batch_series(self, s_uuid, name, x, y, raw_x, *module_lists):
        """Sends a full series to the given module lists and stores it so x-axis changes can re-emit it."""
        sample = {"uuid": s_uuid, "name": name, "x": list(x), "y": list(y), "action": "select"}
        for modules in module_lists:
            for module in modules:
                module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)
        with self._buffers_lock:
            self.samples_buffers[s_uuid] = {"name": name, "uuid": s_uuid, "x": list(raw_x), "y": list(y), "t0": 0.0,
                                            "has_frame_idx": True}

    def _track_batch(self, flat, mode, params):
        """Object tracking across the full sequence (headless). Returns (speeds, motilities, raw speeds)."""
        tracker = ObjectTracker(distance_function=params.get('distance_function', 'euclidean'),
                                distance_threshold=float(params.get('distance_threshold', 30.0)),
                                hit_counter_max=int(params.get('track_max_lost', 15)))
        p_track = dict(params)
        p_track["collect_raw_speeds"] = False
        p_track.update({k: False for k in ("show_trail", "show_arrow", "show_id", "show_age", "show_distance",
                                           "show_speed")})
        speeds, motilities, raw_speeds = [], [], []
        self._batch_track_vectors = []
        self._batch_track_drift_infos = []
        total = max(1, len(flat))
        dpg.set_value(self.batch_progress_tag, 0.0)
        dpg.set_value(self.batch_status_tag, f"Tracking objects across {total} frames…")
        dpg.set_value(self.batch_time_tag, "Tracking: 0s | ETA: --")
        start = time.time()

        for i, item in enumerate(flat):
            if mode == 'contour':
                dets = item[2]
            elif mode == 'denseflow':
                dets = item[3]
            else:  # 'both'
                dets = item[6] if len(item) > 6 else []

            _, _, trk_vectors, mean_speed, _ = tracker.update(
                dets, canvas=None, params=p_track, mode_label=mode, frame_idx=int(item[0])
            )
            self._batch_track_vectors.append((trk_vectors.get("angles"), trk_vectors.get("magnitudes")))
            self._batch_track_drift_infos.append(trk_vectors.get("drift_info"))
            speeds.append(round(mean_speed, 3))
            motilities.append(trk_vectors.get("motility_pct", 0.0))

            if i % 25 == 0 or i == total - 1:
                progress = (i + 1) / total
                active_tracks = len(tracker.tracking) if hasattr(tracker, 'tracking') else len(dets)
                dpg.set_value(self.batch_progress_tag, progress)
                dpg.set_value(self.batch_status_tag,
                              f"Tracking objects: frame {i + 1}/{total} ({int(progress * 100)}% | {active_tracks} tracks)")
                elapsed = time.time() - start
                dpg.set_value(self.batch_time_tag,
                              f"Tracking: {_fmt_time(elapsed)} | ETA: {_fmt_time(max(0.0, elapsed / progress - elapsed))}")

        dpg.set_value(self.batch_progress_tag, 1.0)
        dpg.set_value(self.batch_status_tag, "Emitting aggregated data…")
        return speeds, motilities, raw_speeds

    def _batch_process_video(self, video_path: str, video_name: str, params: dict):
        """Parallel batch processing of a full video file.

        Strategy
        --------
        - In denseflow mode : calls batch_worker_denseflow_chunk (optical flow pair-by-pair).
        - In contour mode   : calls batch_worker_process_chunk (binarize + contour detection).
        - Chunks are processed in parallel; results are gathered in order.
        - Results are replayed through outputs one by one.
        """
        import cv2

        source = video_name or self.label
        self._source_name = source
        self._last_video_path = video_path
        # params already passed in from main thread
        if not params.get("active", True):
            logger.info("CounterTracker batch: processing is inactive (active=False). Skipping analysis and sample emission.")
            self._emit_trigger(video_path)
            return

        mode = params.get('processing_mode', 'contour')

        # ── Get total frames ──────────────────────────────────────────
        cap = _open_batch_video_capture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if total_frames <= 0:
            logger.error(f"CounterTracker batch: could not read video {video_path}")
            self._emit_trigger(video_path)
            return

        chunk_size = max(1, dpg.get_value(self.chunk_size_tag))
        n_threads  = max(1, dpg.get_value(self.n_threads_tag))
        chunks     = [(s, min(s + chunk_size, total_frames)) for s in range(0, total_frames, chunk_size)]
        n_chunks   = len(chunks)

        self._set_batch_lock(True)
        dpg.set_value(self.batch_progress_tag, 0.0)
        dpg.set_value(self.batch_status_tag, f"Starting… {total_frames} frames")
        dpg.set_value(self.batch_time_tag, "Time: 0s | ETA: --")

        start_time_process = time.time()

        # Prepare preprocessed frame shape and precompute any drift lock boundaries in worker thread
        params = batch_prepare_video_params(video_path, total_frames, params)
        params["_n_workers"] = n_threads

        # ── Run in parallel ───────────────────────────────────────────
        ordered_results: list = [None] * n_chunks
        completed = 0
        worker_fn = (batch_worker_denseflow_chunk if mode == 'denseflow'
                     else batch_worker_both_chunk if mode == 'both'
                     else batch_worker_process_chunk)

        with ProcessPoolExecutor(max_workers=n_threads, initializer=batch_worker_init, initargs=(n_threads,)) as exe:
            futures = {exe.submit(worker_fn, video_path, s, e, params): idx for idx, (s, e) in enumerate(chunks)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as exc:
                    logger.error(f"CounterTracker batch chunk {idx} error: {exc}")
                    ordered_results[idx] = []

                completed += 1
                elapsed = time.time() - start_time_process
                progress = completed / n_chunks
                dpg.set_value(self.batch_time_tag,
                              f"Elapsed: {_fmt_time(elapsed)} | ETA: {_fmt_time(elapsed / progress - elapsed)}")
                dpg.set_value(self.batch_progress_tag, progress)
                dpg.set_value(self.batch_status_tag,
                              f"Processing: {completed}/{n_chunks} chunks ({int(progress * 100)}%)")

        # ── Flatten in order ──────────────────────────────────────────
        flat = [item for chunk in ordered_results if chunk for item in chunk]
        flat.sort(key=lambda x: x[0])

        # ── Emit results at once ──────────────────────────────────────
        dpg.set_value(self.batch_status_tag, "Emitting aggregated data…")

        enable_tracking = bool(params.get("enable_tracking", False))
        use_tracking_speed = bool(params.get("use_tracking_speed", False))

        # 1. Raw timestamps, scaled using UI settings
        raw_frame_indices = [float(item[0]) for item in flat]
        x_values = self._build_x_axis([i / video_fps for i in raw_frame_indices])

        # 2. Object tracking across full sequence (also needed in contour mode when drift or phototaxis is active)
        need_tracking_for_drift = (
            mode == 'contour'
            and (bool(params.get("enable_drift_correction", False)) or bool(params.get("drift_only_export", False)))
        )
        need_tracking_for_pi = (
            mode == 'contour'
            and (bool(params.get("emit_phototaxis_series", False)) or bool(self.connections.get("Phototaxis")))
        )
        run_tracker = enable_tracking or need_tracking_for_drift or need_tracking_for_pi
        tracking_speeds, tracking_motilities, all_raw_tracking_speeds = (
            self._track_batch(flat, mode, params) if run_tracker else ([], [], []))

        # Gather per-frame drift_info (from tracker if tracking is active, else from denseflow/both worker)
        if run_tracker and len(getattr(self, "_batch_track_drift_infos", [])) == len(flat):
            frame_drift_infos = list(self._batch_track_drift_infos)
        elif mode == 'denseflow':
            frame_drift_infos = [
                item[4].get("drift_info") if isinstance(item[4], dict) else None
                for item in flat
            ]
        elif mode == 'both':
            frame_drift_infos = [
                item[8] if len(item) > 8 else None
                for item in flat
            ]
        else:
            frame_drift_infos = [None] * len(flat)

        calib_val = float(params.get("calibration", params.get("df_calibration", 1.0)) or 1.0)
        fps_val = float(params.get("fps", params.get("df_fps", video_fps)) or video_fps)
        scale_um_s = calib_val * fps_val
        default_grid_shape = DriftProcessor.parse_grid_mode(params.get("drift_grid_mode", "Full (1x1)"))

        # ── Mode "Only export drift": only save <video>_drift.csv and skip other exports/plots ──
        if bool(params.get("drift_only_export", False)):
            drift_csv_path = DriftProcessor.drift_path_for(video_path)
            ok = DriftProcessor.save_drift_csv(
                drift_csv_path,
                frame_indices=[int(i) for i in raw_frame_indices],
                drift_infos=frame_drift_infos,
                scale=scale_um_s,
                default_grid_shape=default_grid_shape,
            )
            if ok:
                self._live_preprocessed_drift_path = drift_csv_path
                if self._val(self.drift_reuse_preprocessed_tag, False):
                    self._live_preprocessed_drift_map = DriftProcessor.load_drift_csv(drift_csv_path)
                    self.processor.update_params(
                        _preprocessed_drift_map=self._live_preprocessed_drift_map,
                        _preprocessed_drift_path=self._live_preprocessed_drift_path,
                    )
                else:
                    self._live_preprocessed_drift_map = None
                self._update_drift_reuse_ui()
            dpg.set_value(self.batch_progress_tag, 1.0)
            dpg.set_value(self.batch_status_tag, f"Done – Drift exported ({len(flat)} frames)")
            self._set_batch_lock(False)
            self._emit_trigger(video_path)
            return

        # 3. Y values per mode
        surf_y = []
        if mode == 'both':
            surf_y = [float(item[1]) for item in flat]
            base_y = [float(item[2]) for item in flat]
            cnt_y = [float(item[4]) if len(item) > 4 else 0.0 for item in flat]
            default_motility = [float(item[5]) if len(item) > 5 else 0.0 for item in flat]
        elif mode == 'denseflow':
            base_y = [float(item[1]) for item in flat]
            cnt_y = [float(item[4].get("count", len(item[3])) if isinstance(item[4], dict) else item[3])
                     for item in flat]
            if params.get("enable_drift_correction", False):
                default_motility = [
                    float(item[4].get("motility_pct") or 0.0) if isinstance(item[4], dict) else 0.0
                    for item in flat
                ]
            else:
                default_motility = []
        else:  # 'contour'
            surf_y = base_y = [float(item[1]) for item in flat]
            cnt_y = [float(item[3]) for item in flat]
            default_motility = []

        if enable_tracking and use_tracking_speed:
            y_values, motility_y = tracking_speeds, tracking_motilities
        elif enable_tracking:
            y_values, motility_y = base_y, tracking_motilities
        else:
            y_values, motility_y = base_y, default_motility

        coverage_modules = self.connections.get("Result", [])
        count_modules = self.connections.get("Count", [])
        motility_modules = self.connections.get("Motility", [])
        raw_speeds_modules = self.connections.get("Raw Speeds", [])

        if mode == 'both':
            self._emit_batch_series(self._suid("Coverage", source), f"Coverage_{source}", x_values, surf_y,
                                    raw_frame_indices, coverage_modules)
            self._emit_batch_series(self._suid("Speed", source), f"Speed_{source}", x_values, y_values,
                                    raw_frame_indices, coverage_modules)
        else:
            primary_prefix = "Speed" if (mode == 'denseflow' or (enable_tracking and use_tracking_speed)) else "Coverage"
            self._emit_batch_series(self._suid(primary_prefix, source), f"{primary_prefix}_{source}", x_values, y_values, raw_frame_indices,
                                    coverage_modules)

        if cnt_y:
            self._emit_batch_series(self._suid("Count", source), f"Count_{source}", x_values, cnt_y,
                                    raw_frame_indices, coverage_modules, count_modules)
        if motility_y:
            self._emit_batch_series(self._suid("Motility", source), f"Motility_{source}", x_values, motility_y,
                                    raw_frame_indices, coverage_modules, motility_modules)

        # 4. Set UI metric display to the final frame value
        if flat:
            if mode == 'both':
                dpg.set_value(self.metric_text_tag, f"{surf_y[-1]:.1f}% | {y_values[-1]:.3f}")
            elif mode == 'denseflow' or (use_tracking_speed and enable_tracking):
                dpg.set_value(self.metric_text_tag, f"{y_values[-1]:.3f}")
            else:
                dpg.set_value(self.metric_text_tag, f"{y_values[-1]:.2f} %")

            if cnt_y:
                self._set_text(self.count_text_tag, f"{int(cnt_y[-1]):,}")
            self._set_text(self.motility_text_tag, f"{motility_y[-1]:.1f}%" if motility_y else "--")

        # Per-frame vector distributions computed in a single pass (shared by CSV, NPZ, Phototaxis, RawSpeed & CMD)
        if run_tracker and len(self._batch_track_vectors) == len(flat):
            frame_vectors = list(self._batch_track_vectors)
        elif mode == 'denseflow':
            frame_vectors = [(item[4].get("angles"), item[4].get("magnitudes")) if isinstance(item[4], dict)
                             else (None, None) for item in flat]
        elif mode == 'both':
            frame_vectors = [item[7] if len(item) > 7 else (None, None) for item in flat]
        else:
            frame_vectors = []

        pi_light_pos = normalize_pi_light_pos(params.get("pi_light_pos", self.pi_light_pos))
        pi_preset = normalize_pi_preset(params.get("pi_preset", self.pi_preset))
        pi_up_min = float(params.get("pi_up_min", self.pi_up_min))
        pi_up_max = float(params.get("pi_up_max", self.pi_up_max))
        pi_down_min = float(params.get("pi_down_min", self.pi_down_min))
        pi_down_max = float(params.get("pi_down_max", self.pi_down_max))
        pi_weight_speed = bool(params.get("pi_weight_speed", self.pi_weight_speed))

        raw_arrays = build_raw_arrays(frame_vectors) if frame_vectors else None
        phototaxis_y: List[float] = []
        angle_cols = []
        if raw_arrays is not None:
            vx_m = raw_arrays["vx_mean"].astype(np.float64)
            vy_m = raw_arrays["vy_mean"].astype(np.float64)
            spd_m = raw_arrays["vec_speed_mean"].astype(np.float64)
            valid_v = np.isfinite(spd_m) & (spd_m > 0.0) & np.isfinite(vx_m) & np.isfinite(vy_m)
            with np.errstate(divide="ignore", invalid="ignore"):
                ang_mean_arr = np.where(valid_v, np.degrees(np.arctan2(-vy_m, vx_m)) % 360.0, np.nan)
                ang_r_arr = np.where(valid_v, np.hypot(vx_m, vy_m) / np.where(valid_v, spd_m, 1.0), np.nan)
            ang_mean = [float(v) if np.isfinite(v) else float("nan") for v in ang_mean_arr]
            ang_r = [float(v) if np.isfinite(v) else float("nan") for v in ang_r_arr]

            pi_arr = hist_phototaxis_series(
                raw_arrays["ang_counts"],
                raw_arrays["ang_mag_sums"],
                toward_min=pi_up_min,
                toward_max=pi_up_max,
                away_min=pi_down_min,
                away_max=pi_down_max,
                weight_by_mag=pi_weight_speed,
            )
            phototaxis_y = [round(float(v), 4) if np.isfinite(v) else float("nan") for v in pi_arr]
            angle_cols = [("AngleMean_deg", ang_mean), ("AngleR", ang_r), ("Phototaxis", phototaxis_y)]

        # Emit pooled Raw Speeds from 64-bin histogram (capped at 20,000 pts, matching MotilityReprocessor)
        if raw_speeds_modules and raw_arrays is not None:
            total_counts = np.sum(raw_arrays["speed_counts"], axis=0).astype(np.int64)
            edges = raw_arrays["speed_edges"]
            centers = (edges[:-1] + edges[1:]) / 2.0
            pooled_speeds = np.repeat(centers, total_counts)
            if pooled_speeds.size > 20000:
                pooled_speeds = pooled_speeds[:: pooled_speeds.size // 20000]
            if pooled_speeds.size > 0:
                raw_sample = {
                    "uuid": self._suid("RawSpeed", source),
                    "name": f"RawSpeed_{source}",
                    "x": [],
                    "y": pooled_speeds.tolist(),
                    "action": "select",
                }
                for module in raw_speeds_modules:
                    module.input_cb(sample=raw_sample, data_type=IOTypes.SAMPLE)

        emit_pi_series = bool(params.get("emit_phototaxis_series", self.emit_phototaxis_series))
        phototaxis_modules = self.connections.get("Phototaxis", [])
        if phototaxis_y and (emit_pi_series or phototaxis_modules):
            valid_pi_pts = [
                (xv, yv, rx)
                for xv, yv, rx in zip(x_values, phototaxis_y, raw_frame_indices)
                if np.isfinite(yv)
            ]
            if valid_pi_pts:
                pi_x, pi_y, pi_rx = zip(*valid_pi_pts)
                target_pi_lists = (coverage_modules, phototaxis_modules) if emit_pi_series else (phototaxis_modules,)
                self._emit_batch_series(
                    self._suid("Phototaxis", source),
                    f"Phototaxis_{source}",
                    pi_x,
                    pi_y,
                    pi_rx,
                    *target_pi_lists,
                )

        if phototaxis_y:
            finite_pi = [v for v in phototaxis_y if np.isfinite(v)]
            self._set_text(
                self.pi_text_tag,
                self._format_pi_value(finite_pi[-1], pi_light_pos) if finite_pi else "--",
            )
        else:
            self._set_text(self.pi_text_tag, "--")

        # Per-frame drift series (global + per-subcell when grid > 1x1) when drift correction is enabled
        drift_enabled = bool(params.get("enable_drift_correction", False))
        drift_cols = []
        drift_speed_y, drift_angle_y, drift_dx_y, drift_dy_y = [], [], [], []
        drift_cell_speed_arr = None
        drift_cell_angle_arr = None
        drift_cell_dx_arr = None
        drift_cell_dy_arr = None

        if drift_enabled and flat:
            g_rows, g_cols = default_grid_shape
            for d_info in frame_drift_infos:
                if isinstance(d_info, dict) and d_info.get("grid_shape"):
                    g_rows, g_cols = int(d_info["grid_shape"][0]), int(d_info["grid_shape"][1])
                    break
            has_subcells = (g_rows > 1 or g_cols > 1)
            n_frames = len(flat)
            if has_subcells:
                drift_cell_speed_arr = np.zeros((n_frames, g_rows, g_cols), dtype=np.float32)
                drift_cell_angle_arr = np.full((n_frames, g_rows, g_cols), np.nan, dtype=np.float32)
                drift_cell_dx_arr = np.zeros((n_frames, g_rows, g_cols), dtype=np.float32)
                drift_cell_dy_arr = np.zeros((n_frames, g_rows, g_cols), dtype=np.float32)

            for i_f, d_info in enumerate(frame_drift_infos):
                if isinstance(d_info, dict) and d_info.get("enabled", False):
                    spd = float(d_info.get("mean_mag_um_s", 0.0))
                    ang = float(d_info.get("mean_ang_deg", 0.0))
                    dx = float(d_info.get("mean_dx", d_info.get("dx", 0.0)))
                    dy = float(d_info.get("mean_dy", d_info.get("dy", 0.0)))
                else:
                    spd, ang, dx, dy = 0.0, float("nan"), 0.0, 0.0
                drift_speed_y.append(round(spd, 4))
                drift_angle_y.append(round(ang, 3) if np.isfinite(ang) else float("nan"))
                drift_dx_y.append(round(dx, 5))
                drift_dy_y.append(round(dy, 5))

                if has_subcells and isinstance(d_info, dict):
                    cells_raw = d_info.get("cells") or {}
                    cells_map = {}
                    if isinstance(cells_raw, dict):
                        for ck, cv in cells_raw.items():
                            if isinstance(ck, tuple) and len(ck) == 2 and isinstance(cv, dict):
                                cells_map[(int(ck[0]), int(ck[1]))] = cv
                    elif isinstance(cells_raw, list):
                        for cell in cells_raw:
                            if isinstance(cell, dict):
                                cells_map[(int(cell.get("r", 0)), int(cell.get("c", 0)))] = cell
                    for r in range(g_rows):
                        for c in range(g_cols):
                            cell = cells_map.get((r, c))
                            if cell is not None:
                                c_dx = float(cell.get("dx", 0.0))
                                c_dy = float(cell.get("dy", 0.0))
                                c_spd = float(cell.get("speed_um_s", cell.get("mag_um_s", np.hypot(c_dx, c_dy) * scale_um_s)))
                                c_ang = float(
                                    cell.get(
                                        "ang_deg",
                                        np.degrees(cell["ang_rad"]) if "ang_rad" in cell else ((np.degrees(np.arctan2(c_dy, c_dx)) + 360.0) % 360.0),
                                    )
                                )
                                drift_cell_speed_arr[i_f, r, c] = c_spd
                                drift_cell_angle_arr[i_f, r, c] = c_ang
                                drift_cell_dx_arr[i_f, r, c] = c_dx
                                drift_cell_dy_arr[i_f, r, c] = c_dy

            drift_cols.append(("DriftSpeed", drift_speed_y))
            drift_cols.append(("DriftAngle_deg", drift_angle_y))
            if has_subcells and drift_cell_speed_arr is not None and drift_cell_angle_arr is not None:
                for r in range(g_rows):
                    for c in range(g_cols):
                        drift_cols.append((f"DriftSpeed_r{r}_c{c}", [round(float(v), 4) for v in drift_cell_speed_arr[:, r, c]]))
                        drift_cols.append((
                            f"DriftAngle_deg_r{r}_c{c}",
                            [round(float(v), 3) if np.isfinite(v) else float("nan") for v in drift_cell_angle_arr[:, r, c]]
                        ))

            # Also write <video>_drift.csv when drift correction was actively computed during export
            if (dpg.get_value(self.auto_export_csv_tag) or dpg.get_value(self.auto_export_raw_tag)) and not bool(params.get("drift_reuse_preprocessed", False)):
                drift_csv_path = DriftProcessor.drift_path_for(video_path)
                ok = DriftProcessor.save_drift_csv(
                    drift_csv_path,
                    frame_indices=[int(i) for i in raw_frame_indices],
                    drift_infos=frame_drift_infos,
                    scale=scale_um_s,
                    default_grid_shape=(g_rows, g_cols),
                )
                if ok:
                    self._live_preprocessed_drift_path = drift_csv_path
                    if self._val(self.drift_reuse_preprocessed_tag, False):
                        self._live_preprocessed_drift_map = DriftProcessor.load_drift_csv(drift_csv_path)
                    else:
                        self._live_preprocessed_drift_map = None
                    self._update_drift_reuse_ui()

        # Auto Export CSV logic
        if dpg.get_value(self.auto_export_csv_tag):
            csv_path = os.path.splitext(video_path)[0] + ".csv"
            try:
                if mode == 'both':
                    columns = [("Coverage", surf_y), ("Speed", y_values), ("Count", cnt_y), ("Motility", motility_y)]
                elif mode == 'denseflow':
                    columns = [("Speed", y_values), ("Count", cnt_y)]
                    if enable_tracking or motility_y:
                        columns.append(("Motility", motility_y))
                elif enable_tracking and use_tracking_speed:
                    columns = [("Speed", y_values), ("Coverage", surf_y), ("Count", cnt_y), ("Motility", motility_y)]
                elif enable_tracking:
                    columns = [("Coverage", surf_y), ("Speed", tracking_speeds), ("Count", cnt_y),
                               ("Motility", motility_y)]
                else:
                    columns = [("Coverage", y_values), ("Count", cnt_y)]
                columns += angle_cols
                columns += drift_cols

                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.x_axis_mode] + [f"{n}_{source}" for n, _ in columns])
                    writer.writerows(zip(x_values, *(c for _, c in columns)))
            except Exception as e:
                logger.error(f"CounterTracker failed to export CSV to {csv_path}: {e}")

        # Auto Export raw data (<video>.npz): per-frame distributions, replayable by Motility Reprocessor
        if dpg.get_value(self.auto_export_raw_tag):
            raw_path = raw_path_for(video_path)
            try:
                if mode == 'both':
                    metric_kind = "both"
                elif mode == 'denseflow' or (enable_tracking and use_tracking_speed):
                    metric_kind = "speed"
                else:
                    metric_kind = "surface"
                save_raw_npz(
                    raw_path, frame_idx=raw_frame_indices, fps=video_fps, mode=mode, source=source,
                    frame_vectors=frame_vectors or [(None, None)] * len(flat),
                    result=y_values, surface=surf_y if mode == 'both' else None,
                    count=cnt_y, motility=motility_y,
                    phototaxis=phototaxis_y if phototaxis_y else None,
                    metric_kind=metric_kind,
                    drift_speed=drift_speed_y if drift_enabled else None,
                    drift_angle_deg=drift_angle_y if drift_enabled else None,
                    drift_dx=drift_dx_y if drift_enabled else None,
                    drift_dy=drift_dy_y if drift_enabled else None,
                    drift_cell_speed=drift_cell_speed_arr if drift_enabled else None,
                    drift_cell_angle_deg=drift_cell_angle_arr if drift_enabled else None,
                    drift_cell_dx=drift_cell_dx_arr if drift_enabled else None,
                    drift_cell_dy=drift_cell_dy_arr if drift_enabled else None,
                    precomputed_arrays=raw_arrays,
                    extra_meta={"enable_tracking": enable_tracking, "use_tracking_speed": use_tracking_speed,
                                "enable_drift_correction": drift_enabled,
                                "drift_grid_mode": params.get("drift_grid_mode", "Full (1x1)"),
                                "emit_phototaxis_series": emit_pi_series,
                                "pi_light_pos": pi_light_pos,
                                "pi_preset": pi_preset,
                                "pi_up_min": pi_up_min,
                                "pi_up_max": pi_up_max,
                                "pi_down_min": pi_down_min,
                                "pi_down_max": pi_down_max,
                                "pi_weight_speed": pi_weight_speed,
                                "x_axis_mode": self.x_axis_mode, "delta_t": float(self.delta_t or 0.0),
                                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                logger.info(f"CounterTracker raw data exported to {raw_path}")
            except Exception as e:
                logger.error(f"CounterTracker failed to export raw data to {raw_path}: {e}")

        # Generate .processed metadata file
        processed_path = os.path.splitext(video_path)[0] + ".processed"
        try:
            clean_params = {k: v for k, v in params.items() if not k.startswith("_") and not isinstance(v, np.ndarray)}
            metadata = {"mode": mode, "parameters": clean_params, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error(f"CounterTracker failed to write metadata to {processed_path}: {e}")

        # Emit summary CMD dict if any module is connected to CMD
        cmd_modules = self.connections.get("CMD", [])
        if cmd_modules and flat:
            mean_b_spd = float(np.mean(y_values)) if y_values else 0.0
            std_b_spd = 0.0
            if raw_arrays is not None and np.any(np.isfinite(raw_arrays["vec_speed_std"])):
                std_b_spd = float(np.nanmean(raw_arrays["vec_speed_std"]))
            mean_b_mot = float(np.mean(motility_y)) if motility_y else 0.0
            non_mot_b = max(0.0, min(100.0, 100.0 - mean_b_mot)) if motility_y else 0.0

            d_on_b = False
            d_spd_b = 0.0
            d_ang_b = 270.0
            if drift_speed_y and any(s > 0 for s in drift_speed_y):
                d_on_b = True
                d_spd_b = float(np.nanmean(drift_speed_y))
                d_ang_screen = float(np.nanmean(drift_angle_y)) if drift_angle_y else 270.0
                d_ang_b = (360.0 - d_ang_screen) % 360.0

            b_cmd_payload = {
                "speed_mean": round(mean_b_spd, 2),
                "speed_std": round(std_b_spd, 2),
                "non_motile_pct": round(non_mot_b, 1),
                "drift_on": d_on_b,
                "drift_speed": round(d_spd_b, 2),
                "drift_angle": round(d_ang_b, 1),
                "calibration": calib_val,
                "fps": fps_val,
                "n_individuals": int(cnt_y[-1]) if cnt_y else 100,
            }
            for module in cmd_modules:
                module.input_cb(data=b_cmd_payload, data_type=IOTypes.CMD_DICT)

        dpg.set_value(self.batch_progress_tag, 1.0)
        dpg.set_value(self.batch_status_tag, f"Done – {len(flat)} frames processed")
        self._set_batch_lock(False)
        self._emit_trigger(video_path)

    # ── Live result forwarding ─────────────────────────────────────────
    def _push_history(self, s_uuid, name, val, timestamp, frame_idx, conn_keys):
        self._accumulate_point(s_uuid, name, val, timestamp, frame_idx)
        self._emit_full_history(s_uuid, conn_keys=conn_keys)

    def _monitor_loop(self):
        while self._monitor_thread_running:
            result = self.processor.get()
            if result is None:
                time.sleep(0.005)
                continue
            # Drain queue to newest result if a backlog exists
            while (newer := self.processor.get()) is not None:
                result = newer

            # result is (res, frame_idx) from _process_data
            if isinstance(result, tuple) and len(result) == 2 and not isinstance(result[0], np.ndarray):
                res, frame_idx = result
            else:
                res, frame_idx = result, None

            frame, tracking, contours, mask, metric_val, *rest = res
            vectors_data = rest[0] if rest else None
            vd = vectors_data if isinstance(vectors_data, dict) else {}

            # Check if downstream processing is active
            is_proc_active = bool(self._val(self.active_tag, True)) and vd.get("mode") != "bypass"

            # Determine actual mode from metric_val to handle race conditions when switching modes
            actual_mode = 'contour'
            raw_speeds = []
            count_val, motility_val, is_motility_mode = 0, None, False
            pi_val = float("nan")

            if not is_proc_active:
                for tag in (self.metric_text_tag, self.count_text_tag, self.motility_text_tag, self.pi_text_tag):
                    self._set_text(tag, "--")
            else:
                if isinstance(metric_val, (tuple, list)):
                    if len(metric_val) == 3:
                        actual_mode = 'both'
                        coverage_pct, mean_speed, raw_speeds = metric_val
                        dpg.set_value(self.metric_text_tag, f"{coverage_pct:.1f}% | {mean_speed:.3f}")
                    elif len(metric_val) == 2:
                        actual_mode = 'denseflow'
                        mean_speed, raw_speeds = metric_val
                        dpg.set_value(self.metric_text_tag, f"{mean_speed:.3f}")
                else:
                    dpg.set_value(self.metric_text_tag, f"{metric_val:.2f} %")

                count_val = vd.get("count", 0)
                motility_val = vd.get("motility_pct")
                mobile_cnt, total_cnt = vd.get("mobile_count"), vd.get("total_count")
                self._set_text(self.count_text_tag, f"{count_val:,}" if isinstance(count_val, int) else f"{count_val}")

                mode = self._val(self.mode_tag, actual_mode)
                is_motility_mode = (
                    self._val(self.enable_tracking_tag, False)
                    or mode == 'both'
                    or (mode == 'denseflow' and self._val(self.enable_drift_tag, False))
                )
                if is_motility_mode and motility_val is not None:
                    suffix = f" ({mobile_cnt}/{total_cnt})" if mobile_cnt is not None and total_cnt is not None else ""
                    self._set_text(self.motility_text_tag, f"{motility_val:.1f}%{suffix}")
                else:
                    self._set_text(self.motility_text_tag, "--")

                pi_light_pos = normalize_pi_light_pos(self._val(self.pi_light_pos_tag, self.pi_light_pos))
                pi_up_min = float(self._val(self.pi_up_min_tag, self.pi_up_min))
                pi_up_max = float(self._val(self.pi_up_max_tag, self.pi_up_max))
                pi_down_min = float(self._val(self.pi_down_min_tag, self.pi_down_min))
                pi_down_max = float(self._val(self.pi_down_max_tag, self.pi_down_max))
                pi_weight_spd = bool(self._val(self.pi_weight_speed_tag, self.pi_weight_speed))
                pi_val = frame_phototaxis_index(
                    vd.get("angles"),
                    vd.get("magnitudes"),
                    toward_min=pi_up_min,
                    toward_max=pi_up_max,
                    away_min=pi_down_min,
                    away_max=pi_down_max,
                    weight_by_mag=pi_weight_spd,
                )
                self._set_text(self.pi_text_tag, self._format_pi_value(pi_val, pi_light_pos))

            # Update Drift Correction live status display if active
            if dpg.does_item_exist(self.drift_status_text_tag):
                drift_info = vd.get("drift_info")
                if isinstance(drift_info, dict) and drift_info.get("enabled", False):
                    d_spd = drift_info.get("mean_mag_um_s", 0.0)
                    d_ang = drift_info.get("mean_ang_deg", 0.0)
                    d_cnt = drift_info.get("drift_count", 0)
                    if drift_info.get("reused", False):
                        lock_str = " [REUSED]"
                    elif drift_info.get("locked", False):
                        lock_str = " [LOCKED]"
                    else:
                        lock_str = ""
                    dpg.set_value(self.drift_status_text_tag, f"{d_spd:.1f} µm/s @ {d_ang:.1f}° (n={d_cnt}){lock_str}")
                else:
                    dpg.set_value(self.drift_status_text_tag, "--")

            # Update Focus Score display if active
            if dpg.does_item_exist(self.focus_score_tag):
                if vd.get("focus_score") is not None:
                    dpg.set_value(self.focus_score_tag, f"{vd['focus_score']:.1f}")
                elif dpg.does_item_exist(self.enable_focus_picking_tag) and not dpg.get_value(self.enable_focus_picking_tag):
                    dpg.set_value(self.focus_score_tag, "--")

            # Update static removal reference status indicators
            self._set_ref_status(self.r1_status_tag, getattr(self.processor, '_ref_frame_1', None) is not None, 1)
            self._set_ref_status(self.r2_status_tag, getattr(self.processor, '_ref_frame_2', None) is not None, 2)

            # Dispatch other outputs (Frame, Tracking, Contours, Raw Speeds, Vectors)
            source = self._source_name or self.label
            timestamp = time.time()

            for output_key, connected in self.connections.items():
                if not connected:
                    continue
                if output_key == "Frame":
                    for module in connected:
                        module.input_cb(data=frame, data_type=IOTypes.FRAME)
                elif not is_proc_active:
                    continue
                elif output_key == "Tracking" and tracking is not None and tracking != {}:
                    for module in connected:
                        module.input_cb(data=tracking, data_type=IOTypes.TRACKING)
                elif output_key == "Contours" and contours:
                    for module in connected:
                        module.input_cb(data=contours, data_type=IOTypes.CONTOURS)
                elif output_key == "Raw Speeds" and raw_speeds:
                    spd_sample = {"uuid": self._suid("RawSpeed", source), "name": f"RawSpeed_{source}",
                                  "x": [], "y": raw_speeds, "action": "select"}
                    for module in connected:
                        module.input_cb(sample=spd_sample, data_type=IOTypes.SAMPLE)
                elif output_key == "Vectors" and vectors_data is not None:
                    payload = dict(vectors_data, source=source, timestamp=timestamp, frame_idx=frame_idx)
                    for module in connected:
                        module.input_cb(data=payload, data_type=IOTypes.VECTORS)
                elif output_key == "CMD":
                    mean_spd = vd.get("mean_speed", 0.0)
                    raw_spds = vd.get("raw_speeds", raw_speeds or [])
                    std_spd = float(np.std(raw_spds)) if raw_spds and len(raw_spds) > 1 else 0.0

                    mot_pct = vd.get("motility_pct")
                    if mot_pct is None and motility_val is not None:
                        mot_pct = motility_val
                    non_mot_pct = max(0.0, min(100.0, 100.0 - float(mot_pct))) if mot_pct is not None else 0.0

                    d_info = vd.get("drift_info")
                    d_on = bool(d_info and d_info.get("enabled", False) and d_info.get("mean_mag_um_s", 0.0) > 0.0)
                    d_spd = float(d_info.get("mean_mag_um_s", 0.0)) if d_info else 0.0
                    d_screen_ang = float(d_info.get("mean_ang_deg", 270.0)) if d_info else 270.0
                    sim_drift_angle = (360.0 - d_screen_ang) % 360.0

                    calib_val = float(self._val(self.calibration_tag, getattr(self, "calibration", 1.0)) or getattr(self, "calibration", 1.0) or 1.0)
                    fps_val = float(self._val(self.fps_tag, getattr(self, "fps", 30.0)) or getattr(self, "fps", 30.0) or 30.0)
                    n_ind = int(count_val) if isinstance(count_val, (int, float)) and count_val > 0 else 100

                    cmd_payload = {
                        "speed_mean": round(float(mean_spd), 2),
                        "speed_std": round(float(std_spd), 2),
                        "non_motile_pct": round(float(non_mot_pct), 1),
                        "drift_on": d_on,
                        "drift_speed": round(float(d_spd), 2),
                        "drift_angle": round(float(sim_drift_angle), 1),
                        "calibration": calib_val,
                        "fps": fps_val,
                        "n_individuals": n_ind,
                    }
                    for module in connected:
                        module.input_cb(data=cmd_payload, data_type=IOTypes.CMD_DICT)

            # Accumulate and emit full history for Result / Count / Motility only when processing is active
            if not is_proc_active:
                self._skip_first_live_frame = True
            elif self._skip_first_live_frame:
                self._skip_first_live_frame = False
            else:
                push = lambda prefix, name, val, keys: self._push_history(
                    self._suid(prefix, source), name, val, timestamp, frame_idx, keys)
                if actual_mode == 'both':
                    cov, spd, _ = metric_val
                    push("Coverage", f"Coverage_{source}", cov, ["Result"])
                    push("Speed", f"Speed_{source}", spd, ["Result"])
                else:
                    val = metric_val[0] if isinstance(metric_val, (tuple, list)) and len(metric_val) == 2 else metric_val
                    is_speed = (actual_mode == 'denseflow' or (
                        bool(self._val(self.enable_tracking_tag, False)) and bool(self._val(self.use_tracking_speed_tag, False))
                    ))
                    primary_prefix = "Speed" if is_speed else "Coverage"
                    push(primary_prefix, f"{primary_prefix}_{source}", val, ["Result"])

                push("Count", f"Count_{source}", float(count_val), ["Result", "Count"])
                if is_motility_mode and motility_val is not None:
                    push("Motility", f"Motility_{source}", float(motility_val), ["Result", "Motility"])
                emit_pi_live = bool(self._val(self.emit_pi_tag, self.emit_phototaxis_series))
                has_pi_conn = bool(self.connections.get("Phototaxis"))
                if np.isfinite(pi_val) and (emit_pi_live or has_pi_conn):
                    pi_keys = ["Result", "Phototaxis"] if emit_pi_live else ["Phototaxis"]
                    push("Phototaxis", f"Phototaxis_{source}", float(pi_val), pi_keys)

# Exported symbols for the module registry
CounterTrackerWin = counter_tracker_win
EXPORTED_CLASS = counter_tracker_win
EXPORTED_NAME = "Counter Tracker"
