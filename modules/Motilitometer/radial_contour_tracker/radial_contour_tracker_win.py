from typing import Any, Dict, List, Optional, Tuple
import json
import os
import threading
import time
import dearpygui.dearpygui as dpg
import numpy as np

from config.display_scaling import display_scaling
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.app_state import app_state

from .radial_contour_tracker import (
    RadialContourProcessor,
    batch_worker_process_chunk,
    batch_worker_denseflow_chunk,
    batch_worker_both_chunk,
    render_processed_frame,
)


class radial_contour_tracker_win(WindowBase):
    """UI window for the Radial Contour Tracker module.
    Provides controls for monochromatic isolation, circular cropping with offsets,
    gamma correction, separate preprocessing / processing stages, contour
    detection with absolute or relative thresholds, and distance-to-center tracking.
    """

    # 1. Window Configuration
    label = "Radial Contour Tracker"
    description = "Tracks contours and measures radial distance from pellet/center for chemotaxis kymograph analysis."
    win_width = display_scaling.scale(330)
    win_height = display_scaling.scale(740)

    # 2. Input/Output definitions
    accepted_input_types = [IOTypes.FRAME, IOTypes.TRIGGER, IOTypes.FILE_PATH]
    outputs = {
        "Frame": IOTypes.FRAME,
        "Tracking": IOTypes.TRACKING,
        "Contours": IOTypes.CONTOURS,
        "Result": IOTypes.SAMPLE,
        "Raw Speeds": IOTypes.SAMPLE,
        "Trigger": IOTypes.TRIGGER,
    }

    # 3. Persistent fields saved into pipeline JSON
    persistent_fields = [
        "label", "x_axis_mode", "delta_t", "auto_export_csv", "reprocess", "static_speed_thresh",
        "kymograph_bin_width", "kymograph_num_bins", "spatial_scale", "spatial_unit", "adjustment_mode", "invert_image", "crop_type", "crop_radius",
        "crop_width", "crop_height", "show_crop_outline", "crop_use_abs_center",
        "crop_center_x", "crop_center_y", "crop_offset_x", "crop_offset_y",
        "pellet_use_abs_center", "pellet_center_x", "pellet_center_y",
        "pellet_offset_x", "pellet_offset_y", "pellet_radius"
    ]

    # Class-level defaults for persistent fields
    x_axis_mode: str = "Fixed ΔT"
    delta_t: float = 1.0 / 30.0
    auto_export_csv: bool = True
    reprocess: bool = False
    static_speed_thresh: float = 0.5
    kymograph_bin_width: int = 20
    kymograph_num_bins: int = 10
    spatial_scale: float = 1.0
    spatial_unit: str = "µm"
    adjustment_mode: bool = False
    invert_image: bool = False
    crop_type: str = "Circular"
    crop_radius: int = 150
    crop_width: int = 300
    crop_height: int = 300
    show_crop_outline: bool = True
    crop_use_abs_center: bool = False
    crop_center_x: int = 0
    crop_center_y: int = 0
    crop_offset_x: int = 0
    crop_offset_y: int = 0
    pellet_use_abs_center: bool = False
    pellet_center_x: int = 0
    pellet_center_y: int = 0
    pellet_offset_x: int = 0
    pellet_offset_y: int = 0
    pellet_radius: int = 15

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._monitor_thread = None
        self._monitor_thread_running = False
        self.processor = None

        # Load settings from JSON
        settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
        try:
            with open(settings_path, "r") as f:
                self.settings = json.load(f)
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to load settings.json for radial contour tracker: {e}")
            self.settings = {}

        s_cal = self.settings.get("calibration", {})
        s_crop = self.settings.get("crop", {})
        s_mono = self.settings.get("monochrome", {})
        s_glb = self.settings.get("global", {})
        s_bin  = self.settings.get('binarize', {})
        s_cnt  = self.settings.get('contour', {})
        s_trk  = self.settings.get('tracker', {})
        s_prox = self.settings.get('proximity', {})
        s_vid  = self.settings.get('video_export', {})
        s_ret  = self.settings.get('reticules', {})
        s_df   = self.settings.get('denseflow', {})

        def _get_val(key: str, alt_key: Optional[str] = None, fallback: Any = None) -> Any:
            val = getattr(self, key, kwargs.get(key))
            if val is not None:
                return val
            if alt_key:
                val_alt = getattr(self, alt_key, kwargs.get(alt_key))
                if val_alt is not None:
                    return val_alt
            return fallback

        self.x_axis_mode = str(_get_val("x_axis_mode", fallback="Fixed ΔT") or "Fixed ΔT")
        try:
            self.delta_t = float(_get_val("delta_t", fallback=1.0 / 30.0))
        except (ValueError, TypeError):
            self.delta_t = 1.0 / 30.0
        self.auto_export_csv = bool(_get_val("auto_export_csv", fallback=True))
        self.reprocess = bool(_get_val("reprocess", fallback=False))
        try:
            self.static_speed_thresh = float(_get_val("static_speed_thresh", fallback=0.5))
        except (ValueError, TypeError):
            self.static_speed_thresh = 0.5
        try:
            self.kymograph_bin_width = int(_get_val("kymograph_bin_width", fallback=20))
        except (ValueError, TypeError):
            self.kymograph_bin_width = 20
        try:
            self.kymograph_num_bins = int(_get_val("kymograph_num_bins", fallback=10))
        except (ValueError, TypeError):
            self.kymograph_num_bins = 10
        try:
            self.spatial_scale = float(_get_val("spatial_scale", fallback=s_cal.get("spatial_scale", 1.0) or 1.0))
        except (ValueError, TypeError):
            self.spatial_scale = 1.0
        self.spatial_unit = str(_get_val("spatial_unit", fallback=s_cal.get("spatial_unit", "µm") or "µm"))
        self.adjustment_mode = bool(_get_val("adjustment_mode", fallback=s_glb.get("adjustment_mode", False)))
        self.invert_image = bool(_get_val("invert_image", fallback=s_mono.get("invert_image", False)))
        self.crop_type = str(_get_val("crop_type", fallback=s_crop.get("crop_type", "Circular" if s_crop.get("circular_crop", False) else "None")))
        try:
            self.crop_radius = int(_get_val("crop_radius", fallback=s_crop.get("crop_radius", 150) or 150))
        except (ValueError, TypeError):
            self.crop_radius = 150
        try:
            self.crop_width = int(_get_val("crop_width", fallback=s_crop.get("crop_width", 300) or 300))
        except (ValueError, TypeError):
            self.crop_width = 300
        try:
            self.crop_height = int(_get_val("crop_height", fallback=s_crop.get("crop_height", 300) or 300))
        except (ValueError, TypeError):
            self.crop_height = 300
        self.show_crop_outline = bool(_get_val("show_crop_outline", fallback=True))
        
        self.crop_use_abs_center = bool(_get_val("crop_use_abs_center", "use_absolute_center", s_crop.get("use_absolute_center", False)))
        try:
            self.crop_center_x = int(_get_val("crop_center_x", "center_x", s_crop.get("center_x", 0) or 0))
        except (ValueError, TypeError):
            self.crop_center_x = 0
        try:
            self.crop_center_y = int(_get_val("crop_center_y", "center_y", s_crop.get("center_y", 0) or 0))
        except (ValueError, TypeError):
            self.crop_center_y = 0
        try:
            self.crop_offset_x = int(_get_val("crop_offset_x", "offset_x", s_crop.get("offset_x", 0) or 0))
        except (ValueError, TypeError):
            self.crop_offset_x = 0
        try:
            self.crop_offset_y = int(_get_val("crop_offset_y", "offset_y", s_crop.get("offset_y", 0) or 0))
        except (ValueError, TypeError):
            self.crop_offset_y = 0

        self.pellet_use_abs_center = bool(_get_val("pellet_use_abs_center", "use_absolute_center", s_crop.get("use_absolute_center", False)))
        try:
            self.pellet_center_x = int(_get_val("pellet_center_x", "center_x", s_crop.get("center_x", 0) or 0))
        except (ValueError, TypeError):
            self.pellet_center_x = 0
        try:
            self.pellet_center_y = int(_get_val("pellet_center_y", "center_y", s_crop.get("center_y", 0) or 0))
        except (ValueError, TypeError):
            self.pellet_center_y = 0
        try:
            self.pellet_offset_x = int(_get_val("pellet_offset_x", "offset_x", s_crop.get("offset_x", 0) or 0))
        except (ValueError, TypeError):
            self.pellet_offset_x = 0
        try:
            self.pellet_offset_y = int(_get_val("pellet_offset_y", "offset_y", s_crop.get("offset_y", 0) or 0))
        except (ValueError, TypeError):
            self.pellet_offset_y = 0
        try:
            self.pellet_radius = int(_get_val("pellet_radius", fallback=s_crop.get("pellet_radius", 15) or 15))
        except (ValueError, TypeError):
            self.pellet_radius = 15
        self.samples_buffers = {}
        self._skip_first_live_frame = True

        # Initialize processor with default settings
        s_mono = self.settings.get('monochrome', {})
        s_crop = self.settings.get('crop', {})
        s_bin  = self.settings.get('binarize', {})
        s_cnt  = self.settings.get('contour', {})
        s_trk  = self.settings.get('tracker', {})
        s_prox = self.settings.get('proximity', {})
        s_vid  = self.settings.get('video_export', {})
        s_ret  = self.settings.get('reticules', {})
        s_df   = self.settings.get('denseflow', {})
        s_glb  = self.settings.get('global', {})

        self.processor = RadialContourProcessor(params={
            "processing_mode": s_glb.get("processing_mode", "contour"),
            
            # Stages
            "enable_preprocessing": s_glb.get("enable_preprocessing", True),
            "active": s_glb.get("active", True),

            # Preprocessing (Monochrome, Gamma & Crop & Offsets)
            "monochrome_mode": s_mono.get("channel", "Grayscale"),
            "gamma": s_mono.get("gamma", 1.0),
            "circular_crop": s_crop.get("circular_crop", False),
            "crop_radius": s_crop.get("crop_radius", 150),
            "offset_x": s_crop.get("offset_x", 0),
            "offset_y": s_crop.get("offset_y", 0),

            # Static parasite removal defaults
            "remove_static": False,
            "similarity_threshold": 15,
            
            # Reticules
            "show_distance_circles": s_ret.get("show_distance_circles", False),
            "num_distance_circles": s_ret.get("num_distance_circles", 5),

            # Binarize
            "blur": s_bin.get("blur", 3) if 'blur' in s_bin else 3,
            "block_size": s_bin.get("block_size", 25) if 'block_size' in s_bin else 25,
            "bin_thresh": s_bin.get("bin_thresh", 15) if 'bin_thresh' in s_bin else 15,
            "morph_close": s_bin.get("morph_close", 0) if 'morph_close' in s_bin else 0,

            # Contour Detection
            "enable_contour_detection": s_cnt.get("enable_contour_detection", True),
            "area_thresh_mode": s_cnt.get("area_thresh_mode", "Absolute (px^2)"),
            "lower_surface_thresh": s_cnt.get("lower_surface_thresh", 25),
            "upper_surface_thresh": s_cnt.get("upper_surface_thresh", 1000),
            "show_blobs": s_cnt.get("show_blobs", True) if 'show_blobs' in s_cnt else True,
            "isolate_selection": s_cnt.get("isolate_selection", False),
            "show_boxes": s_cnt.get("show_boxes", True),
            "show_centroids": s_cnt.get("show_centroids", True),
            "visu_format": s_cnt.get("visu_format", "Frame"),
            "fill_contours": s_cnt.get("fill_contours", True),

            # Tracker
            "enable_tracking": s_trk.get("enable_tracking", True),
            "distance_function": s_trk.get("distance_function", "euclidean"),
            "distance_threshold": s_trk.get("distance_threshold", 30),
            "trail_length": s_trk.get("trail_length", 30),
            "trail_thickness": s_trk.get("trail_thickness", 2),
            "show_trail": s_trk.get("show_trail", False),
            "show_id": s_trk.get("show_id", False),
            "show_age": s_trk.get("show_age", False),
            "show_distance": s_trk.get("show_distance", False),
            "show_speed": s_trk.get("show_speed", False),
            "show_proximity": s_trk.get("show_proximity", True),

            # Proximity Density Heatmap
            "enable_proximity": s_prox.get("enable_proximity", False),
            "proximity_density_mode": s_prox.get("density_mode", "Centroid"),
            "proximity_scale_mode": s_prox.get("scale_mode", "Absolute"),
            "proximity_palette": s_prox.get("palette", "Inferno"),
            "proximity_opacity": s_prox.get("opacity", 0.5),
            "proximity_show_scale": s_prox.get("show_scale", True),
            "proximity_scale_position": s_prox.get("scale_position", "Top Right"),
            "proximity_scale_size": s_prox.get("scale_size", 150),

            # Video Export
            "export_next_as_video": s_vid.get("export_next_as_video", False),
            "crop_to_selection": s_vid.get("crop_to_selection", False),
            "no_subsampling": s_vid.get("no_subsampling", False),
            "target_duration": s_vid.get("target_duration", 30.0),
            "output_fps": s_vid.get("output_fps", 30),

            # Optical Flow
            "df_pyr_scale": s_df.get("pyr_scale", 0.5),
            "df_levels": s_df.get("levels", 2),
            "df_winsize": s_df.get("winsize", 9),
            "df_iterations": s_df.get("iterations", 2),
            "df_poly_n": s_df.get("poly_n", 5),
            "df_poly_sigma": s_df.get("poly_sigma", 1.1),
            "df_lower_mag_thresh": s_df.get("lower_mag_thresh", 1.5),
            "df_erosion": s_df.get("erosion", 3),
            "df_supersat": s_df.get("supersat", True),
        },
        buffer_size=s_glb.get("buffer_size", 1),
        drop_policy=s_glb.get("drop_policy", "drop_new"))
        self.processor.start()

        # UI Tags
        self.active_tag = f"radial_active_{self.UUID}"
        self.mode_tag = f"radial_mode_{self.UUID}"
        
        # Stages tags
        self.enable_preprocessing_tag = f"radial_enable_prep_{self.UUID}"

        # Preprocessing tags
        self.monochrome_tag = f"radial_monochrome_{self.UUID}"
        self.gamma_tag = f"radial_gamma_{self.UUID}"
        self.circular_crop_tag = f"radial_circular_crop_{self.UUID}"
        self.crop_radius_tag = f"radial_crop_radius_{self.UUID}"
        self.offset_x_tag = f"radial_offset_x_{self.UUID}"
        self.offset_y_tag = f"radial_offset_y_{self.UUID}"

        # Static removal tags
        self.remove_static_tag = f"radial_remove_static_{self.UUID}"
        self.similarity_threshold_tag = f"radial_sim_thresh_{self.UUID}"
        self.r1_status_tag = f"radial_r1_status_{self.UUID}"
        self.r2_status_tag = f"radial_r2_status_{self.UUID}"
        self.static_removal_group_tag = f"radial_static_group_{self.UUID}"
        
        self._last_r1_trigger = 0
        self._last_r2_trigger = 0
        self._last_clear_ref_trigger = 0
        
        # Reticules tags
        self.show_circles_tag = f"radial_show_circles_{self.UUID}"
        self.num_circles_tag = f"radial_num_circles_{self.UUID}"

        # Binarize tags
        self.blur_tag = f"radial_blur_{self.UUID}"
        self.block_size_tag = f"radial_block_size_{self.UUID}"
        self.bin_thresh_tag = f"radial_bin_thresh_{self.UUID}"
        self.morph_close_tag = f"radial_morph_close_{self.UUID}"

        # Contour tags
        self.enable_contour_tag = f"radial_enable_contour_{self.UUID}"
        self.area_thresh_mode_tag = f"radial_area_thresh_mode_{self.UUID}"
        self.lower_str_tag = f"radial_l_str_{self.UUID}"
        self.upper_str_tag = f"radial_u_str_{self.UUID}"
        self.show_blobs_tag = f"radial_show_blobs_{self.UUID}"
        self.show_boxes_tag = f"radial_show_boxes_{self.UUID}"
        self.show_centroids_tag = f"radial_show_centroids_{self.UUID}"
        self.isolate_tag = f"radial_isolate_{self.UUID}"
        self.fill_contours_tag = f"radial_fill_contours_{self.UUID}"

        # Tracker tags
        self.enable_tracking_tag = f"radial_enable_tracking_{self.UUID}"
        self.trail_length_tag = f"radial_trail_length_{self.UUID}"
        self.trail_thickness_tag = f"radial_trail_thickness_{self.UUID}"
        self.show_trail_tag = f"radial_show_trail_{self.UUID}"
        self.show_id_tag = f"radial_show_id_{self.UUID}"
        self.show_age_tag = f"radial_show_age_{self.UUID}"
        self.show_distance_tag = f"radial_show_distance_{self.UUID}"
        self.show_speed_tag = f"radial_show_speed_{self.UUID}"
        self.show_proximity_tag = f"radial_show_proximity_{self.UUID}"
        self.dist_func_tag = f"radial_dist_func_{self.UUID}"
        self.dist_thresh_tag = f"radial_dist_thresh_{self.UUID}"
        self.speed_window_tag = f"radial_speed_window_{self.UUID}"

        # Proximity Density tags
        self.proximity_group_tag = f"radial_proximity_group_{self.UUID}"
        self.enable_proximity_tag = f"radial_enable_proximity_{self.UUID}"
        self.proximity_mode_tag = f"radial_proximity_mode_{self.UUID}"
        self.proximity_scale_mode_tag = f"radial_proximity_scale_mode_{self.UUID}"
        self.proximity_palette_tag = f"radial_proximity_palette_{self.UUID}"
        self.proximity_opacity_tag = f"radial_proximity_opacity_{self.UUID}"
        self.proximity_show_scale_tag = f"radial_proximity_show_scale_{self.UUID}"
        self.proximity_scale_pos_tag = f"radial_proximity_scale_pos_{self.UUID}"
        self.proximity_scale_size_tag = f"radial_proximity_scale_size_{self.UUID}"

        # Video Export tags
        self.export_as_video_tag = f"radial_export_as_video_{self.UUID}"
        self.crop_to_selection_tag = f"radial_crop_to_selection_{self.UUID}"
        self.no_subsampling_tag = f"radial_no_subsampling_{self.UUID}"
        self.video_duration_tag = f"radial_video_duration_{self.UUID}"
        self.video_fps_tag = f"radial_video_fps_{self.UUID}"

        # Optical Flow tags
        self.df_group_tag = f"radial_df_group_{self.UUID}"
        self.df_mag_thresh_tag = f"radial_df_mag_thresh_{self.UUID}"
        self.df_erosion_tag = f"radial_df_erosion_{self.UUID}"
        self.df_supersat_tag = f"radial_df_supersat_{self.UUID}"
        self.df_use_contours_tag = f"radial_df_use_contours_{self.UUID}"
        self.df_show_arrows_tag = f"radial_df_show_arrows_{self.UUID}"
        self.df_isolate_tag = f"radial_df_isolate_{self.UUID}"
        self.df_lower_str_tag = f"radial_df_l_str_{self.UUID}"
        self.df_upper_str_tag = f"radial_df_u_str_{self.UUID}"
        self.df_contour_group_tag = f"radial_df_contour_group_{self.UUID}"
        self.both_vis_tag = f"radial_both_vis_{self.UUID}"
        self.both_group_tag = f"radial_both_group_{self.UUID}"

        # X Axis tags
        self.x_mode_tag = f"radial_x_mode_{self.UUID}"
        self.delta_t_tag = f"radial_delta_t_{self.UUID}"
        self.delta_t_group_tag = f"radial_delta_t_group_{self.UUID}"

        # Preprocessing & Crop & Pellet Center tags
        self.adjustment_mode_tag = f"radial_adj_mode_{self.UUID}"
        self.monochrome_tag = f"radial_monochrome_{self.UUID}"
        self.gamma_tag = f"radial_gamma_{self.UUID}"
        self.invert_image_tag = f"radial_invert_image_{self.UUID}"
        self.crop_type_tag = f"radial_crop_type_{self.UUID}"
        self.crop_radius_tag = f"radial_crop_radius_{self.UUID}"
        self.crop_width_tag = f"radial_crop_width_{self.UUID}"
        self.crop_height_tag = f"radial_crop_height_{self.UUID}"
        self.show_crop_outline_tag = f"radial_show_crop_outline_{self.UUID}"
        self.crop_circular_group_tag = f"radial_crop_circ_grp_{self.UUID}"
        self.crop_rect_group_tag = f"radial_crop_rect_grp_{self.UUID}"
        
        self.crop_use_abs_tag = f"radial_crop_use_abs_{self.UUID}"
        self.crop_center_x_tag = f"radial_crop_center_x_{self.UUID}"
        self.crop_center_y_tag = f"radial_crop_center_y_{self.UUID}"
        self.crop_offset_x_tag = f"radial_crop_offset_x_{self.UUID}"
        self.crop_offset_y_tag = f"radial_crop_offset_y_{self.UUID}"
        self.crop_abs_center_group_tag = f"radial_crop_abs_grp_{self.UUID}"
        self.crop_rel_center_group_tag = f"radial_crop_rel_grp_{self.UUID}"

        self.pellet_radius_tag = f"radial_pellet_radius_{self.UUID}"
        self.pellet_use_abs_tag = f"radial_pellet_use_abs_{self.UUID}"
        self.pellet_center_x_tag = f"radial_pellet_center_x_{self.UUID}"
        self.pellet_center_y_tag = f"radial_pellet_center_y_{self.UUID}"
        self.pellet_offset_x_tag = f"radial_pellet_offset_x_{self.UUID}"
        self.pellet_offset_y_tag = f"radial_pellet_offset_y_{self.UUID}"
        self.pellet_abs_center_group_tag = f"radial_pellet_abs_grp_{self.UUID}"
        self.pellet_rel_center_group_tag = f"radial_pellet_rel_grp_{self.UUID}"

        # Spatial Calibration tags
        self.spatial_scale_tag = f"radial_spatial_scale_{self.UUID}"
        self.spatial_unit_tag = f"radial_spatial_unit_{self.UUID}"

        # Metric tags
        self.metric_label_tag = f"radial_metric_label_{self.UUID}"
        self.metric_text_tag = f"radial_metric_text_{self.UUID}"

        # Batch tags
        self.n_threads_tag = f"radial_n_threads_{self.UUID}"
        self.chunk_size_tag = f"radial_chunk_{self.UUID}"
        self.auto_export_csv_tag = f"radial_auto_export_csv_{self.UUID}"
        self.reprocess_tag = f"radial_reprocess_{self.UUID}"
        self.static_speed_thresh_tag = f"radial_static_speed_thresh_{self.UUID}"
        self.kymograph_bin_width_tag = f"radial_kymograph_bin_width_{self.UUID}"
        self.kymograph_num_bins_tag = f"radial_kymograph_num_bins_{self.UUID}"
        self.batch_progress_tag = f"radial_progress_{self.UUID}"
        self.batch_status_tag = f"radial_status_{self.UUID}"
        self.batch_time_tag = f"radial_batch_time_{self.UUID}"
        self.batch_group_tag = f"radial_batch_group_{self.UUID}"
        self.main_settings_group = f"radial_settings_group_{self.UUID}"

        # Groups tags
        self.preprocessing_group_tag = f"radial_prep_group_{self.UUID}"
        self.processing_main_group_tag = f"radial_proc_main_group_{self.UUID}"

        # Monitor thread
        self._source_name = ""
        self._batch_running = False
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread_running = True
        self._monitor_thread.start()

        # Build Window UI
        scale = display_scaling.scale
        with self.window():
            
            with dpg.group(tag=self.main_settings_group):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Export Settings (JSON)", callback=self._export_settings_cb)
                    dpg.add_button(label="Import Settings (JSON)", callback=self._import_settings_cb)
                dpg.add_checkbox(label="Mode Ajustement (Sans Export / Sans Accumulation)", default_value=self.adjustment_mode, callback=self._update_param_cb, tag=self.adjustment_mode_tag)
                # Preprocessing Toggle & Controls (Always functional, works before Active is checked)
                dpg.add_checkbox(label="Enable Preprocessing", default_value=s_glb.get("enable_preprocessing", True), callback=self._on_preprocessing_toggle, tag=self.enable_preprocessing_tag)
                
                with dpg.group(tag=self.preprocessing_group_tag, show=s_glb.get("enable_preprocessing", True)):
                    dpg.add_combo(["Grayscale", "RGB (Color)", "Red Channel", "Green Channel", "Blue Channel"], default_value=s_mono.get("channel", "Grayscale"), label="Channel", callback=self._update_param_cb, tag=self.monochrome_tag)
                    dpg.add_checkbox(label="Invert Image (Negative)", default_value=self.invert_image, callback=self._update_param_cb, tag=self.invert_image_tag)
                    dpg.add_drag_float(label="Gamma Adjustment", default_value=s_mono.get("gamma", 1.0), min_value=0.1, max_value=5.0, format="%.2f", speed=0.05, callback=self._update_param_cb, tag=self.gamma_tag)
                    
                    # Crop Settings & Center Coordinates
                    dpg.add_combo(["None", "Circular", "Rectangle"], default_value=self.crop_type, label="Crop Mode", callback=self._on_crop_type_changed, tag=self.crop_type_tag)
                    dpg.add_checkbox(label="Show Crop Outline (Post-processing)", default_value=self.show_crop_outline, callback=self._update_param_cb, tag=self.show_crop_outline_tag)
                    
                    with dpg.group(tag=self.crop_circular_group_tag, show=(self.crop_type == "Circular")):
                        dpg.add_drag_int(label="Crop Radius (px)", default_value=self.crop_radius, min_value=1, max_value=100000, callback=self._update_param_cb, tag=self.crop_radius_tag)
                    
                    with dpg.group(tag=self.crop_rect_group_tag, show=(self.crop_type == "Rectangle")):
                        with dpg.group(horizontal=True):
                            dpg.add_drag_int(label="Crop W (px)", default_value=self.crop_width, min_value=1, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.crop_width_tag)
                            dpg.add_drag_int(label="Crop H (px)", default_value=self.crop_height, min_value=1, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.crop_height_tag)

                    dpg.add_checkbox(label="Use Absolute Crop Center (X, Y)", default_value=self.crop_use_abs_center, callback=self._on_crop_center_mode_changed, tag=self.crop_use_abs_tag)
                    with dpg.group(tag=self.crop_abs_center_group_tag, show=self.crop_use_abs_center):
                        with dpg.group(horizontal=True):
                            dpg.add_drag_int(label="Crop Center X (px)", default_value=self.crop_center_x, min_value=-100000, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.crop_center_x_tag)
                            dpg.add_drag_int(label="Crop Center Y (px)", default_value=self.crop_center_y, min_value=-100000, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.crop_center_y_tag)
                    with dpg.group(tag=self.crop_rel_center_group_tag, show=not self.crop_use_abs_center):
                        with dpg.group(horizontal=True):
                            dpg.add_drag_int(label="Crop Offset X", default_value=self.crop_offset_x, min_value=-100000, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.crop_offset_x_tag)
                            dpg.add_drag_int(label="Crop Offset Y", default_value=self.crop_offset_y, min_value=-100000, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.crop_offset_y_tag)

                    # Attractant Pellet Settings & Center Coordinates
                    dpg.add_text("Attractant Pellet Settings")
                    dpg.add_drag_int(label="Pellet Radius (px)", default_value=self.pellet_radius, min_value=0, max_value=100000, callback=self._update_param_cb, tag=self.pellet_radius_tag)
                    dpg.add_checkbox(label="Use Absolute Pellet Center (X, Y)", default_value=self.pellet_use_abs_center, callback=self._on_pellet_center_mode_changed, tag=self.pellet_use_abs_tag)
                    with dpg.group(tag=self.pellet_abs_center_group_tag, show=self.pellet_use_abs_center):
                        with dpg.group(horizontal=True):
                            dpg.add_drag_int(label="Pellet Center X (px)", default_value=self.pellet_center_x, min_value=-100000, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.pellet_center_x_tag)
                            dpg.add_drag_int(label="Pellet Center Y (px)", default_value=self.pellet_center_y, min_value=-100000, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.pellet_center_y_tag)
                    with dpg.group(tag=self.pellet_rel_center_group_tag, show=not self.pellet_use_abs_center):
                        with dpg.group(horizontal=True):
                            dpg.add_drag_int(label="Pellet Offset X", default_value=self.pellet_offset_x, min_value=-100000, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.pellet_offset_x_tag)
                            dpg.add_drag_int(label="Pellet Offset Y", default_value=self.pellet_offset_y, min_value=-100000, max_value=100000, width=scale(100), callback=self._update_param_cb, tag=self.pellet_offset_y_tag)
                    
                    # Static Parasite Removal controls
                    dpg.add_checkbox(label="Remove static noise", default_value=False, callback=self._on_remove_static_toggle, tag=self.remove_static_tag)
                    with dpg.group(tag=self.static_removal_group_tag, show=False):
                        dpg.add_drag_int(label="Similarity Thresh", default_value=15, min_value=0, max_value=255, callback=self._update_param_cb, tag=self.similarity_threshold_tag)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Capture R1", callback=self._capture_r1_cb)
                            dpg.add_button(label="Capture R2", callback=self._capture_r2_cb)
                            dpg.add_button(label="Clear Ref", callback=self._clear_ref_cb)
                        with dpg.group(horizontal=True):
                            dpg.add_text("Frame 1: Empty", tag=self.r1_status_tag, color=(255, 100, 100))
                            dpg.add_text("Frame 2: Empty", tag=self.r2_status_tag, color=(255, 100, 100))
                dpg.add_separator()

                # Spatial Calibration Settings
                dpg.add_text("Spatial Calibration (Distance & Scale)")
                with dpg.group(horizontal=True):
                    dpg.add_drag_float(label="Scale (unit/px)", default_value=self.spatial_scale, min_value=0.0001, max_value=10000.0, format="%.4f", speed=0.01, width=scale(110), callback=self._update_param_cb, tag=self.spatial_scale_tag)
                    dpg.add_combo(["µm", "mm", "px"], default_value=self.spatial_unit, width=scale(60), callback=self._update_param_cb, tag=self.spatial_unit_tag)
                dpg.add_separator()

                # Concentric circles / Reticules
                dpg.add_text("Distance Reticules Settings")
                dpg.add_checkbox(label="Show concentric circles", default_value=s_ret.get("show_distance_circles", False), callback=self._update_param_cb, tag=self.show_circles_tag)
                dpg.add_input_int(label="Reticules count", default_value=s_ret.get("num_distance_circles", 5), min_value=1, max_value=20, callback=self._update_param_cb, tag=self.num_circles_tag)
                dpg.add_separator()

                # Active (Processing activation) & Mode
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Active", default_value=s_glb.get("active", True), callback=self._on_active_toggle, tag=self.active_tag)
                    dpg.add_combo(["contour", "denseflow", "both"], default_value=s_glb.get("processing_mode", "contour"), width=scale(120), callback=self._on_mode_change_cb, tag=self.mode_tag)
                dpg.add_separator()

                # Processing UI controls (toggled by Active checkbox)
                with dpg.group(tag=self.processing_main_group_tag, show=s_glb.get("active", True)):
                    self.mode_contour_group_tag = f"radial_mode_contour_{self.UUID}"
                    mode = s_glb.get("processing_mode", "contour")

                    # Both Mode UI
                    with dpg.group(tag=self.both_group_tag, show=(mode == "both")):
                        dpg.add_text("Combined Analysis (Contours + Flow)")
                        dpg.add_text("Visualization:")
                        dpg.add_combo(["Surface", "Speed", "Selection"],
                                      default_value=s_df.get("both_vis_mode", "Surface"),
                                      callback=self._update_param_cb, tag=self.both_vis_tag)
                        dpg.add_separator()

                    # Dense Flow UI
                    with dpg.group(tag=self.df_group_tag, show=(mode == "denseflow")):
                        dpg.add_text("Dense Optical Flow Options")
                        dpg.add_drag_float(label="Min Velocity Thresh", default_value=s_df.get("lower_mag_thresh", 1.5), min_value=0.0, max_value=20.0, format="%.2f", speed=0.1, callback=self._update_param_cb, tag=self.df_mag_thresh_tag)
                        dpg.add_drag_int(label="Erosion (noise)", default_value=s_df.get("erosion", 3), min_value=0, max_value=10, callback=self._update_param_cb, tag=self.df_erosion_tag)
                        dpg.add_checkbox(label="Supersat Visual", default_value=s_df.get("supersat", True), callback=self._update_param_cb, tag=self.df_supersat_tag)
                        dpg.add_separator()
                        dpg.add_checkbox(label="Use Contours / Vectors", default_value=s_df.get("use_contours", False), callback=self._on_df_contour_toggle_cb, tag=self.df_use_contours_tag)
                        with dpg.group(tag=self.df_contour_group_tag, show=s_df.get("use_contours", False)):
                            dpg.add_checkbox(label="Show Arrows", default_value=s_df.get("show_arrows", True), callback=self._update_param_cb, tag=self.df_show_arrows_tag)
                            dpg.add_checkbox(label="Isolate Selection", default_value=s_df.get("isolate_selection", False), callback=self._update_param_cb, tag=self.df_isolate_tag)
                            dpg.add_drag_int(label="Lower thresh %", default_value=s_df.get("lower_surface_thresh", 25), min_value=0, max_value=100, callback=self._update_param_cb, tag=self.df_lower_str_tag)
                            dpg.add_drag_int(label="Upper thresh %", default_value=s_df.get("upper_surface_thresh", 100), min_value=0, max_value=100, callback=self._update_param_cb, tag=self.df_upper_str_tag)

                    # Contour UI Group
                    with dpg.group(tag=self.mode_contour_group_tag, show=(mode == "contour" or mode == "both")):
                        # Binarize Settings
                        dpg.add_text("Binarize Settings")
                        dpg.add_drag_int(label="Blur", default_value=3, min_value=1, max_value=25, callback=self._update_param_cb, tag=self.blur_tag)
                        dpg.add_drag_int(label="Block size", default_value=25, min_value=3, max_value=51, callback=self._update_param_cb, tag=self.block_size_tag)
                        dpg.add_drag_int(label="Bin thresh", default_value=15, min_value=0, max_value=50, callback=self._update_param_cb, tag=self.bin_thresh_tag)
                        dpg.add_drag_int(label="Close", default_value=0, min_value=0, max_value=5, callback=self._update_param_cb, tag=self.morph_close_tag)
                        dpg.add_checkbox(label="Show blobs", default_value=s_cnt.get("show_blobs", True) if 'show_blobs' in s_cnt else True, callback=self._update_param_cb, tag=self.show_blobs_tag)
                        dpg.add_separator()

                        # Contour detection
                        dpg.add_checkbox(label="Enable Contour Detection", default_value=s_cnt.get("enable_contour_detection", True), callback=self._on_contour_toggle_cb, tag=self.enable_contour_tag)
                        self.contour_group_tag = f"radial_contour_group_{self.UUID}"
                        with dpg.group(tag=self.contour_group_tag, show=s_cnt.get("enable_contour_detection", True)):
                            dpg.add_combo(["Absolute (px^2)", "Relative (%)"], default_value=s_cnt.get("area_thresh_mode", "Absolute (px^2)"), label="Threshold Mode", callback=self._update_param_cb, tag=self.area_thresh_mode_tag)
                            dpg.add_drag_int(label="Lower thresh", default_value=s_cnt.get("lower_surface_thresh", 25), min_value=0, max_value=1000000, callback=self._update_param_cb, tag=self.lower_str_tag)
                            dpg.add_drag_int(label="Upper thresh", default_value=s_cnt.get("upper_surface_thresh", 1000), min_value=0, max_value=1000000, callback=self._update_param_cb, tag=self.upper_str_tag)
                            dpg.add_checkbox(label="Show boxes", default_value=s_cnt.get("show_boxes", True), callback=self._update_param_cb, tag=self.show_boxes_tag)
                            dpg.add_checkbox(label="Show centroids", default_value=s_cnt.get("show_centroids", True), callback=self._update_param_cb, tag=self.show_centroids_tag)
                            dpg.add_checkbox(label="Isolate selection", default_value=s_cnt.get("isolate_selection", False), callback=self._update_param_cb, tag=self.isolate_tag)
                            dpg.add_checkbox(label="Fill contours", default_value=s_cnt.get("fill_contours", True), callback=self._update_param_cb, tag=self.fill_contours_tag)
                        dpg.add_separator()

                        # Tracker Settings
                        dpg.add_text("Tracker Settings")
                        self.tracker_group_tag = f"radial_tracker_group_{self.UUID}"
                        dpg.add_checkbox(label="Enable Tracking", default_value=s_trk.get("enable_tracking", True), callback=self._on_tracker_toggle_cb, tag=self.enable_tracking_tag)
                        with dpg.group(tag=self.tracker_group_tag, show=s_trk.get("enable_tracking", True)):
                            dpg.add_button(label="Reset Tracking", callback=self._reset_tracking_cb)
                            with dpg.group(horizontal=True):
                                dpg.add_text("Dist Func:")
                                dpg.add_combo(["euclidean", "manhattan"], default_value=s_trk.get("distance_function", "euclidean"), tag=self.dist_func_tag, callback=self._update_param_cb, width=scale(100))
                            dpg.add_drag_int(label="Dist Thresh", default_value=s_trk.get("distance_threshold", 30), min_value=1, max_value=200, tag=self.dist_thresh_tag, callback=self._update_param_cb)
                            dpg.add_checkbox(label="Show ID", default_value=s_trk.get("show_id", False), callback=self._update_param_cb, tag=self.show_id_tag)
                            dpg.add_checkbox(label="Show trail", default_value=s_trk.get("show_trail", False), callback=self._update_param_cb, tag=self.show_trail_tag)
                            dpg.add_drag_int(label="Trail length", default_value=s_trk.get("trail_length", 30), min_value=0, max_value=100, callback=self._update_param_cb, tag=self.trail_length_tag)
                            dpg.add_drag_int(label="Trail thickness", default_value=s_trk.get("trail_thickness", 2), min_value=1, max_value=20, callback=self._update_param_cb, tag=self.trail_thickness_tag)
                            dpg.add_checkbox(label="Show age", default_value=s_trk.get("show_age", False), callback=self._update_param_cb, tag=self.show_age_tag)
                            dpg.add_checkbox(label="Show distance", default_value=s_trk.get("show_distance", False), callback=self._update_param_cb, tag=self.show_distance_tag)
                            dpg.add_checkbox(label="Show speed", default_value=s_trk.get("show_speed", False), callback=self._update_param_cb, tag=self.show_speed_tag)
                            dpg.add_checkbox(label="Show proximity", default_value=s_trk.get("show_proximity", True), callback=self._update_param_cb, tag=self.show_proximity_tag)
                        dpg.add_separator()

                        # Proximity Density
                        dpg.add_text("Proximity Density")
                        dpg.add_checkbox(label="Show Proximity Density", default_value=s_prox.get("enable_proximity", False), callback=self._on_proximity_toggle_cb, tag=self.enable_proximity_tag)
                        with dpg.group(tag=self.proximity_group_tag, show=s_prox.get("enable_proximity", False)):
                            dpg.add_combo(["Centroid", "Surface"], default_value=s_prox.get("density_mode", "Centroid"), label="Density Mode", callback=self._update_param_cb, tag=self.proximity_mode_tag)
                            dpg.add_combo(["Absolute", "Relative (%)"], default_value=s_prox.get("scale_mode", "Absolute"), label="Scale Mode", callback=self._update_param_cb, tag=self.proximity_scale_mode_tag)
                            dpg.add_combo(["Inferno", "Jet", "Turbo", "Viridis", "Plasma", "Hot", "HSV"], default_value=s_prox.get("palette", "Inferno"), label="Palette", callback=self._update_param_cb, tag=self.proximity_palette_tag)
                            dpg.add_drag_float(label="Opacity", default_value=s_prox.get("opacity", 0.5), min_value=0.0, max_value=1.0, format="%.2f", callback=self._update_param_cb, tag=self.proximity_opacity_tag)
                            dpg.add_checkbox(label="Show Scale", default_value=s_prox.get("show_scale", True), callback=self._update_param_cb, tag=self.proximity_show_scale_tag)
                            dpg.add_combo(["Top Right", "Bottom Right", "Top Left", "Bottom Left"], default_value=s_prox.get("scale_position", "Top Right"), label="Scale Position", callback=self._update_param_cb, tag=self.proximity_scale_pos_tag)
                            dpg.add_drag_int(label="Scale Height", default_value=s_prox.get("scale_size", 150), min_value=50, max_value=500, callback=self._update_param_cb, tag=self.proximity_scale_size_tag)

                dpg.add_separator()
                # Metric display
                with dpg.group(horizontal=True):
                    dpg.add_text("Metric:", tag=self.metric_label_tag)
                    dpg.add_text("0.00", tag=self.metric_text_tag)

                dpg.add_separator()
                dpg.add_text("X Axis Scale Settings")
                dpg.add_combo(
                    items=["Timestamp", "Index", "Fixed ΔT"],
                    default_value=self.x_axis_mode,
                    tag=self.x_mode_tag,
                    width=-1,
                    callback=self._on_x_mode_changed,
                )
                with dpg.group(tag=self.delta_t_group_tag, show=(self.x_axis_mode == "Fixed ΔT")):
                    dpg.add_input_float(
                        label="ΔT (s/frame)",
                        tag=self.delta_t_tag,
                        default_value=self.delta_t,
                        min_value=1e-6,
                        format="%.4f",
                        width=-1,
                        step=0,
                        callback=self._on_delta_t_changed,
                    )

            dpg.add_separator()
            # Batch Processing
            dpg.add_text("Batch Processing")
            with dpg.group(tag=self.batch_group_tag):
                with dpg.group(tag=f"batch_inputs_{self.UUID}"):
                    dpg.add_input_int(label="Threads", tag=self.n_threads_tag,
                                    default_value=8, min_value=1, max_value=32,
                                    width=scale(80), step=0)
                    dpg.add_input_int(label="Chunk (frames)", tag=self.chunk_size_tag,
                                    default_value=50, min_value=1, max_value=5000,
                                    width=scale(80), step=0)
                    dpg.add_input_float(label="Static Speed Thresh (µm/s)", tag=self.static_speed_thresh_tag,
                                        default_value=self.static_speed_thresh, min_value=0.0, max_value=100.0, format="%.2f",
                                        width=scale(80), step=0, callback=self._update_param_cb)
                    dpg.add_input_int(label="Radial Classes Count (N bins)", tag=self.kymograph_num_bins_tag,
                                      default_value=self.kymograph_num_bins, min_value=1, max_value=100,
                                      width=scale(80), step=0, callback=self._update_param_cb)
                    dpg.add_checkbox(label="Auto Export CSV", tag=self.auto_export_csv_tag,
                                     default_value=self.auto_export_csv, callback=self._on_auto_export_csv_toggled)
                    dpg.add_checkbox(label="Reprocess (from .processed)", tag=self.reprocess_tag,
                                     default_value=self.reprocess, callback=self._on_reprocess_toggled)
                    dpg.add_separator()
                    dpg.add_text("Timelapse Video Export")
                    dpg.add_checkbox(label="Export Next Path as Video", tag=self.export_as_video_tag,
                                     default_value=s_vid.get("export_next_as_video", False), callback=self._update_param_cb)
                    dpg.add_checkbox(label="Crop to Selection", tag=self.crop_to_selection_tag,
                                     default_value=s_vid.get("crop_to_selection", False), callback=self._update_param_cb)
                    dpg.add_checkbox(label="No Subsampling (Full tracking, slower)", tag=self.no_subsampling_tag,
                                     default_value=s_vid.get("no_subsampling", False), callback=self._update_param_cb)
                    dpg.add_input_float(label="Duration (s)", tag=self.video_duration_tag,
                                        default_value=float(s_vid.get("target_duration", 30.0)), min_value=1.0, max_value=3600.0, format="%.1f", step=1.0,
                                        width=scale(150), callback=self._update_param_cb)
                    dpg.add_input_int(label="Output FPS", tag=self.video_fps_tag,
                                      default_value=int(s_vid.get("output_fps", 30)), min_value=1, max_value=120,
                                      width=scale(150), step=0, callback=self._update_param_cb)
                
                with dpg.group(tag=f"batch_progress_group_{self.UUID}", show=False):
                    dpg.add_progress_bar(tag=self.batch_progress_tag,
                                        default_value=0.0, width=-1)
                    dpg.add_text("", tag=self.batch_status_tag)
                    dpg.add_text("", tag=self.batch_time_tag, color=(200, 200, 200))

        self.update_permission()

    def update_permission(self) -> None:
        """Adjust widget permissions based on user mode."""
        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _export_settings_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Opens a native OS file dialog to export current module settings to a JSON file."""
        def run_dialog():
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                file_path = filedialog.asksaveasfilename(
                    title="Export Radial Contour Tracker Settings",
                    defaultextension=".json",
                    filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
                )
                root.destroy()
                if file_path:
                    params = self._get_current_params()
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(params, f, indent=4)
                    from loguru import logger
                    logger.info(f"RadialContourTracker: Settings successfully exported to {file_path}")
            except Exception as e:
                from loguru import logger
                logger.error(f"RadialContourTracker: Export settings error: {e}")

        threading.Thread(target=run_dialog, daemon=True).start()

    def _import_settings_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        """Opens a native OS file dialog to import module settings from a JSON file."""
        def run_dialog():
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                file_path = filedialog.askopenfilename(
                    title="Import Radial Contour Tracker Settings",
                    filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
                )
                root.destroy()
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_params = json.load(f)
                    self._apply_imported_params(loaded_params)
                    from loguru import logger
                    logger.info(f"RadialContourTracker: Settings successfully imported from {file_path}")
            except Exception as e:
                from loguru import logger
                logger.error(f"RadialContourTracker: Import settings error: {e}")

        threading.Thread(target=run_dialog, daemon=True).start()

    def _apply_imported_params(self, p: dict):
        mapping = {
            "processing_mode": self.mode_tag,
            "enable_preprocessing": self.enable_preprocessing_tag,
            "active": self.active_tag,
            "adjustment_mode": self.adjustment_mode_tag,
            "monochrome_mode": self.monochrome_tag,
            "invert_image": self.invert_image_tag,
            "gamma": self.gamma_tag,
            "crop_type": self.crop_type_tag,
            "crop_radius": self.crop_radius_tag,
            "crop_width": self.crop_width_tag,
            "crop_height": self.crop_height_tag,
            "show_crop_outline": self.show_crop_outline_tag,
            "crop_use_abs_center": self.crop_use_abs_tag,
            "crop_center_x": self.crop_center_x_tag,
            "crop_center_y": self.crop_center_y_tag,
            "crop_offset_x": self.crop_offset_x_tag,
            "crop_offset_y": self.crop_offset_y_tag,
            "pellet_use_abs_center": self.pellet_use_abs_tag,
            "pellet_center_x": self.pellet_center_x_tag,
            "pellet_center_y": self.pellet_center_y_tag,
            "pellet_offset_x": self.pellet_offset_x_tag,
            "pellet_offset_y": self.pellet_offset_y_tag,
            "pellet_radius": self.pellet_radius_tag,
            "remove_static": self.remove_static_tag,
            "similarity_threshold": self.similarity_threshold_tag,
            "show_distance_circles": self.show_circles_tag,
            "num_distance_circles": self.num_circles_tag,
            "blur": self.blur_tag,
            "block_size": self.block_size_tag,
            "bin_thresh": self.bin_thresh_tag,
            "morph_close": self.morph_close_tag,
            "enable_contour_detection": self.enable_contour_tag,
            "area_thresh_mode": self.area_thresh_mode_tag,
            "lower_surface_thresh": self.lower_str_tag,
            "upper_surface_thresh": self.upper_str_tag,
            "show_blobs": self.show_blobs_tag,
            "show_boxes": self.show_boxes_tag,
            "show_centroids": self.show_centroids_tag,
            "isolate_selection": self.isolate_tag,
            "fill_contours": self.fill_contours_tag,
            "enable_tracking": self.enable_tracking_tag,
            "distance_function": self.dist_func_tag,
            "distance_threshold": self.dist_thresh_tag,
            "trail_length": self.trail_length_tag,
            "trail_thickness": self.trail_thickness_tag,
            "show_trail": self.show_trail_tag,
            "show_id": self.show_id_tag,
            "show_age": self.show_age_tag,
            "show_distance": self.show_distance_tag,
            "show_speed": self.show_speed_tag,
            "show_proximity": self.show_proximity_tag,
            "enable_proximity": self.enable_proximity_tag,
            "proximity_density_mode": self.proximity_mode_tag,
            "proximity_scale_mode": self.proximity_scale_mode_tag,
            "proximity_palette": self.proximity_palette_tag,
            "proximity_opacity": self.proximity_opacity_tag,
            "proximity_show_scale": self.proximity_show_scale_tag,
            "proximity_scale_position": self.proximity_scale_pos_tag,
            "proximity_scale_size": self.proximity_scale_size_tag,
            "export_next_as_video": self.export_as_video_tag,
            "crop_to_selection": self.crop_to_selection_tag,
            "no_subsampling": self.no_subsampling_tag,
            "target_duration": self.video_duration_tag,
            "output_fps": self.video_fps_tag,
            "df_lower_mag_thresh": self.df_mag_thresh_tag,
            "df_erosion": self.df_erosion_tag,
            "df_supersat": self.df_supersat_tag,
            "df_use_contours": self.df_use_contours_tag,
            "df_show_arrows": self.df_show_arrows_tag,
            "df_isolate_selection": self.df_isolate_tag,
            "df_lower_surface_thresh": self.df_lower_str_tag,
            "df_upper_surface_thresh": self.df_upper_str_tag,
            "both_vis_mode": self.both_vis_tag,
            "static_speed_thresh": self.static_speed_thresh_tag,
            "kymograph_bin_width": self.kymograph_bin_width_tag,
            "kymograph_num_bins": self.kymograph_num_bins_tag,
            "auto_export_csv": self.auto_export_csv_tag,
            "reprocess": self.reprocess_tag,
            "spatial_scale": self.spatial_scale_tag,
            "spatial_unit": self.spatial_unit_tag,
        }

        for param_key, tag in mapping.items():
            if param_key in p and dpg.does_item_exist(tag):
                try:
                    dpg.set_value(tag, p[param_key])
                except Exception:
                    pass

        if "crop_type" in p:
            self._on_crop_type_changed(None, p["crop_type"])
        if "crop_use_abs_center" in p:
            self._on_crop_center_mode_changed(None, p["crop_use_abs_center"])
        if "pellet_use_abs_center" in p:
            self._on_pellet_center_mode_changed(None, p["pellet_use_abs_center"])
        if "enable_preprocessing" in p:
            self._on_preprocessing_toggle(None, p["enable_preprocessing"])
        if "active" in p:
            self._on_active_toggle(None, p["active"])
        if "processing_mode" in p:
            self._on_mode_change_cb(None, p["processing_mode"])

        self._update_param_cb(None, None)

    def _on_preprocessing_toggle(self, sender, app_data, user_data=None, *args, **kwargs):
        dpg.configure_item(self.preprocessing_group_tag, show=app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_crop_type_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        dpg.configure_item(self.crop_circular_group_tag, show=(app_data == "Circular"))
        dpg.configure_item(self.crop_rect_group_tag, show=(app_data == "Rectangle"))
        self._update_param_cb(sender, app_data, user_data)

    def _on_crop_center_mode_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        dpg.configure_item(self.crop_abs_center_group_tag, show=app_data)
        dpg.configure_item(self.crop_rel_center_group_tag, show=not app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_pellet_center_mode_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        dpg.configure_item(self.pellet_abs_center_group_tag, show=app_data)
        dpg.configure_item(self.pellet_rel_center_group_tag, show=not app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_active_toggle(self, sender, app_data, user_data=None, *args, **kwargs):
        """Active checkbox controls the visibility and activation of the Processing group."""
        dpg.configure_item(self.processing_main_group_tag, show=app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _on_remove_static_toggle(self, sender, app_data, user_data=None, *args, **kwargs):
        dpg.configure_item(self.static_removal_group_tag, show=app_data)
        self._update_param_cb(sender, app_data, user_data)

    def _capture_r1_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._last_r1_trigger += 1
        self._update_param_cb(sender, app_data, user_data)

    def _capture_r2_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._last_r2_trigger += 1
        self._update_param_cb(sender, app_data, user_data)

    def _clear_ref_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._last_clear_ref_trigger += 1
        self._update_param_cb(sender, app_data, user_data)

    def _on_x_mode_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.x_axis_mode = app_data
        dpg.configure_item(self.delta_t_group_tag, show=(app_data == "Fixed ΔT"))
        for s_uuid in list(self.samples_buffers.keys()):
            self._emit_full_history(s_uuid)

    def _on_delta_t_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.delta_t = app_data
        for s_uuid in list(self.samples_buffers.keys()):
            self._emit_full_history(s_uuid)

    def _on_auto_export_csv_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.auto_export_csv = app_data
        self._update_param_cb(sender, app_data, user_data)

    def _on_reprocess_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.reprocess = app_data
        self._update_param_cb(sender, app_data, user_data)

    def _build_x_axis(self, raw_x: list[float], has_frame_idx: bool = False) -> list[float]:
        mode = self.x_axis_mode
        n = len(raw_x)
        if has_frame_idx:
            if mode == "Index":
                return list(raw_x)
            elif mode == "Fixed ΔT":
                dt = self.delta_t
                return [x * dt for x in raw_x]
            else:  # Timestamp
                dt = self.delta_t
                return [x * dt for x in raw_x]
        else:
            if mode == "Index":
                return list(range(n))
            elif mode == "Fixed ΔT":
                dt = self.delta_t
                return [i * dt for i in range(n)]
            else:  # Timestamp
                return list(raw_x)

    def _clear_downstream_plots(self):
        for s_uuid in list(self.samples_buffers.keys()):
            out_sample = {
                "name": self.samples_buffers[s_uuid]["name"],
                "uuid": s_uuid,
                "x": [],
                "y": [],
                "action": "delete",
            }
            for module in self.connections.get("Result", []):
                try:
                    module.input_cb(sample=out_sample, data_type=IOTypes.SAMPLE)
                except Exception:
                    pass
        clear_sample = {
            "action": "clear",
            "clear": True
        }
        for module in self.connections.get("Result", []):
            try:
                module.input_cb(sample=clear_sample, data_type=IOTypes.SAMPLE)
            except Exception:
                pass

    def _accumulate_point(self, s_uuid: str, name: str, val: float, timestamp: float, frame_idx: int | None = None):
        if (dpg.does_item_exist(self.adjustment_mode_tag) and dpg.get_value(self.adjustment_mode_tag)) or getattr(self, 'adjustment_mode', False):
            return
        if s_uuid not in self.samples_buffers:
            self.samples_buffers[s_uuid] = {
                "x": [],
                "y": [],
                "name": name,
                "t0": None,
                "has_frame_idx": False
            }
        buff = self.samples_buffers[s_uuid]
        if buff["t0"] is None:
            buff["t0"] = timestamp
        
        if frame_idx is not None:
            buff["has_frame_idx"] = True
            try:
                idx = buff["x"].index(frame_idx)
                buff["y"][idx] = val
            except ValueError:
                buff["x"].append(frame_idx)
                buff["y"].append(val)
                zipped = sorted(zip(buff["x"], buff["y"]))
                buff["x"] = [x for x, y in zipped]
                buff["y"] = [y for x, y in zipped]
        else:
            buff["x"].append(timestamp - buff["t0"])
            buff["y"].append(val)

    def _emit_full_history(self, s_uuid: str):
        buff = self.samples_buffers.get(s_uuid)
        if not buff:
            return
        x_scaled = self._build_x_axis(buff["x"], has_frame_idx=buff.get("has_frame_idx", False))
        out_sample = {
            "name": buff["name"],
            "uuid": s_uuid,
            "x": x_scaled,
            "y": list(buff["y"]),
            "action": "select",
        }
        for module in self.connections.get("Result", []):
            try:
                module.input_cb(sample=out_sample, data_type=IOTypes.SAMPLE)
            except Exception as e:
                from loguru import logger
                logger.error(f"Error emitting sample history to module: {e}")

    def _on_contour_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        enabled = dpg.get_value(self.enable_contour_tag)
        dpg.configure_item(self.contour_group_tag, show=enabled)
        self._update_param_cb(sender, app_data, user_data)

    def _on_tracker_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        enabled = dpg.get_value(self.enable_tracking_tag)
        dpg.configure_item(self.tracker_group_tag, show=enabled)
        self._update_param_cb(sender, app_data, user_data)

    def _on_proximity_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        enabled = dpg.get_value(self.enable_proximity_tag)
        dpg.configure_item(self.proximity_group_tag, show=enabled)
        self._update_param_cb(sender, app_data, user_data)

    def _on_df_contour_toggle_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        enabled = dpg.get_value(self.df_use_contours_tag)
        dpg.configure_item(self.df_contour_group_tag, show=enabled)
        self._update_param_cb(sender, app_data, user_data)

    def _on_mode_change_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        mode = dpg.get_value(self.mode_tag)
        dpg.configure_item(self.mode_contour_group_tag, show=(mode == "contour" or mode == "both"))
        dpg.configure_item(self.df_group_tag, show=(mode == "denseflow"))
        dpg.configure_item(self.both_group_tag, show=(mode == "both"))
        if mode == "denseflow":
            dpg.set_value(self.metric_label_tag, "Mean Speed (µm/s):")
        elif mode == "both":
            dpg.set_value(self.metric_label_tag, "Contours | Flow Speed:")
        else:
            dpg.set_value(self.metric_label_tag, "Metric:")
        try:
            self.processor._reset_denseflow_state()
        except AttributeError:
            pass
        self._update_param_cb(sender, app_data, user_data)

    def _reset_tracking_cb(self, sender, app_data, user_data=None, *args, **kwargs):
        self._reset_trigger_count = getattr(self, '_reset_trigger_count', 0) + 1
        self.processor.update_params(reset_trigger=self._reset_trigger_count)
        self._clear_downstream_plots()
        self.samples_buffers.clear()
        self._skip_first_live_frame = True

    def _update_param_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        s_df = self.settings.get('denseflow', {})
        if dpg.does_item_exist(self.x_mode_tag):
            self.x_axis_mode = dpg.get_value(self.x_mode_tag)
        if dpg.does_item_exist(self.delta_t_tag):
            self.delta_t = dpg.get_value(self.delta_t_tag)
            
        params = {
            "processing_mode": dpg.get_value(self.mode_tag),
            "enable_preprocessing": dpg.get_value(self.enable_preprocessing_tag),
            "active": dpg.get_value(self.active_tag),
            
            "adjustment_mode": dpg.get_value(self.adjustment_mode_tag) if dpg.does_item_exist(self.adjustment_mode_tag) else self.adjustment_mode,
            "monochrome_mode": dpg.get_value(self.monochrome_tag),
            "invert_image": dpg.get_value(self.invert_image_tag) if dpg.does_item_exist(self.invert_image_tag) else self.invert_image,
            "gamma": float(dpg.get_value(self.gamma_tag)),
            "crop_type": dpg.get_value(self.crop_type_tag) if dpg.does_item_exist(self.crop_type_tag) else self.crop_type,
            "circular_crop": (dpg.get_value(self.crop_type_tag) == "Circular") if dpg.does_item_exist(self.crop_type_tag) else (self.crop_type == "Circular"),
            "crop_radius": dpg.get_value(self.crop_radius_tag) if dpg.does_item_exist(self.crop_radius_tag) else self.crop_radius,
            "crop_width": dpg.get_value(self.crop_width_tag) if dpg.does_item_exist(self.crop_width_tag) else self.crop_width,
            "crop_height": dpg.get_value(self.crop_height_tag) if dpg.does_item_exist(self.crop_height_tag) else self.crop_height,
            "show_crop_outline": dpg.get_value(self.show_crop_outline_tag) if dpg.does_item_exist(self.show_crop_outline_tag) else self.show_crop_outline,
            
            "crop_use_abs_center": dpg.get_value(self.crop_use_abs_tag) if dpg.does_item_exist(self.crop_use_abs_tag) else self.crop_use_abs_center,
            "crop_center_x": dpg.get_value(self.crop_center_x_tag) if dpg.does_item_exist(self.crop_center_x_tag) else self.crop_center_x,
            "crop_center_y": dpg.get_value(self.crop_center_y_tag) if dpg.does_item_exist(self.crop_center_y_tag) else self.crop_center_y,
            "crop_offset_x": dpg.get_value(self.crop_offset_x_tag) if dpg.does_item_exist(self.crop_offset_x_tag) else self.crop_offset_x,
            "crop_offset_y": dpg.get_value(self.crop_offset_y_tag) if dpg.does_item_exist(self.crop_offset_y_tag) else self.crop_offset_y,

            "pellet_use_abs_center": dpg.get_value(self.pellet_use_abs_tag) if dpg.does_item_exist(self.pellet_use_abs_tag) else self.pellet_use_abs_center,
            "pellet_center_x": dpg.get_value(self.pellet_center_x_tag) if dpg.does_item_exist(self.pellet_center_x_tag) else self.pellet_center_x,
            "pellet_center_y": dpg.get_value(self.pellet_center_y_tag) if dpg.does_item_exist(self.pellet_center_y_tag) else self.pellet_center_y,
            "pellet_offset_x": dpg.get_value(self.pellet_offset_x_tag) if dpg.does_item_exist(self.pellet_offset_x_tag) else self.pellet_offset_x,
            "pellet_offset_y": dpg.get_value(self.pellet_offset_y_tag) if dpg.does_item_exist(self.pellet_offset_y_tag) else self.pellet_offset_y,
            "pellet_radius": dpg.get_value(self.pellet_radius_tag) if dpg.does_item_exist(self.pellet_radius_tag) else self.pellet_radius,

            "remove_static": dpg.get_value(self.remove_static_tag),
            "similarity_threshold": dpg.get_value(self.similarity_threshold_tag),
            "capture_r1_trigger": self._last_r1_trigger,
            "capture_r2_trigger": self._last_r2_trigger,
            "clear_ref_trigger": self._last_clear_ref_trigger,
            
            "show_distance_circles": dpg.get_value(self.show_circles_tag),
            "num_distance_circles": dpg.get_value(self.num_circles_tag),

            "blur": dpg.get_value(self.blur_tag),
            "block_size": dpg.get_value(self.block_size_tag),
            "bin_thresh": dpg.get_value(self.bin_thresh_tag),
            "morph_close": dpg.get_value(self.morph_close_tag),

            "enable_contour_detection": dpg.get_value(self.enable_contour_tag),
            "area_thresh_mode": dpg.get_value(self.area_thresh_mode_tag),
            "lower_surface_thresh": dpg.get_value(self.lower_str_tag),
            "upper_surface_thresh": dpg.get_value(self.upper_str_tag),
            "show_blobs": dpg.get_value(self.show_blobs_tag),
            "show_boxes": dpg.get_value(self.show_boxes_tag),
            "show_centroids": dpg.get_value(self.show_centroids_tag),
            "isolate_selection": dpg.get_value(self.isolate_tag),
            "fill_contours": dpg.get_value(self.fill_contours_tag),

            "enable_tracking": dpg.get_value(self.enable_tracking_tag),
            "distance_function": dpg.get_value(self.dist_func_tag),
            "distance_threshold": dpg.get_value(self.dist_thresh_tag),
            "trail_length": int(dpg.get_value(self.trail_length_tag)),
            "trail_thickness": int(dpg.get_value(self.trail_thickness_tag)),
            "show_trail": dpg.get_value(self.show_trail_tag),
            "show_id": dpg.get_value(self.show_id_tag),
            "show_age": dpg.get_value(self.show_age_tag),
            "show_distance": dpg.get_value(self.show_distance_tag),
            "show_speed": dpg.get_value(self.show_speed_tag),
            "speed_window_frames": int(dpg.get_value(self.speed_window_tag)) if dpg.does_item_exist(self.speed_window_tag) else 5,
            "show_proximity": dpg.get_value(self.show_proximity_tag),
            "enable_proximity": dpg.get_value(self.enable_proximity_tag),
            "proximity_density_mode": dpg.get_value(self.proximity_mode_tag),
            "proximity_scale_mode": dpg.get_value(self.proximity_scale_mode_tag),
            "proximity_palette": dpg.get_value(self.proximity_palette_tag),
            "proximity_opacity": float(dpg.get_value(self.proximity_opacity_tag)),
            "proximity_show_scale": dpg.get_value(self.proximity_show_scale_tag),
            "proximity_scale_position": dpg.get_value(self.proximity_scale_pos_tag),
            "proximity_scale_size": int(dpg.get_value(self.proximity_scale_size_tag)),

            "export_next_as_video": dpg.get_value(self.export_as_video_tag),
            "crop_to_selection": dpg.get_value(self.crop_to_selection_tag),
            "no_subsampling": dpg.get_value(self.no_subsampling_tag),
            "target_duration": float(dpg.get_value(self.video_duration_tag)),
            "output_fps": int(dpg.get_value(self.video_fps_tag)),

            "df_calibration": s_df.get("calibration", 1.0),
            "df_fps": s_df.get("fps", 30.0),
            "df_pyr_scale": s_df.get("pyr_scale", 0.5),
            "df_levels": s_df.get("levels", 2),
            "df_winsize": s_df.get("winsize", 9),
            "df_iterations": s_df.get("iterations", 2),
            "df_poly_n": s_df.get("poly_n", 5),
            "df_poly_sigma": s_df.get("poly_sigma", 1.1),
            "df_lower_mag_thresh": dpg.get_value(self.df_mag_thresh_tag),
            "df_erosion": dpg.get_value(self.df_erosion_tag),
            "df_supersat": dpg.get_value(self.df_supersat_tag),
            "df_use_contours": dpg.get_value(self.df_use_contours_tag),
            "df_show_arrows": dpg.get_value(self.df_show_arrows_tag),
            "df_isolate_selection": dpg.get_value(self.df_isolate_tag),
            "df_lower_surface_thresh": dpg.get_value(self.df_lower_str_tag),
            "df_upper_surface_thresh": dpg.get_value(self.df_upper_str_tag),
            "both_vis_mode": dpg.get_value(self.both_vis_tag),

            "active": dpg.get_value(self.active_tag),
            "collect_raw_speeds": len(self.connections.get("Raw Speeds", [])) > 0,
            "static_speed_thresh": float(dpg.get_value(self.static_speed_thresh_tag)) if dpg.does_item_exist(self.static_speed_thresh_tag) else self.static_speed_thresh,
            "kymograph_bin_width": int(dpg.get_value(self.kymograph_bin_width_tag)) if dpg.does_item_exist(self.kymograph_bin_width_tag) else self.kymograph_bin_width,
            "kymograph_num_bins": int(dpg.get_value(self.kymograph_num_bins_tag)) if dpg.does_item_exist(self.kymograph_num_bins_tag) else self.kymograph_num_bins,
            "auto_export_csv": dpg.get_value(self.auto_export_csv_tag) if dpg.does_item_exist(self.auto_export_csv_tag) else self.auto_export_csv,
            "reprocess": dpg.get_value(self.reprocess_tag) if dpg.does_item_exist(self.reprocess_tag) else self.reprocess,
            "spatial_scale": float(dpg.get_value(self.spatial_scale_tag)) if dpg.does_item_exist(self.spatial_scale_tag) else self.spatial_scale,
            "spatial_unit": str(dpg.get_value(self.spatial_unit_tag)) if dpg.does_item_exist(self.spatial_unit_tag) else self.spatial_unit,
        }
        self.processor.update_params(**params)

    def on_close(self) -> None:
        """Cleanup on window close."""
        self._monitor_thread_running = False
        if getattr(self, "_monitor_thread", None) is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        if getattr(self, "processor", None) is not None and hasattr(self.processor, "stop"):
            try:
                self.processor.stop()
            except Exception:
                pass
        super().on_close()

    def close(self) -> None:
        """Explicit close handler."""
        self.on_close()

    def destroy(self) -> None:
        """Complete resource deallocation."""
        self.on_close()

    def is_ready(self) -> bool:
        """Check if module is ready to receive input data."""
        if self._batch_running:
            return False
        if getattr(self, "processor", None) is not None and hasattr(self.processor, "is_ready"):
            return bool(self.processor.is_ready())
        return True

    def input_cb(self, *args: Any, **kwargs: Any) -> bool:
        data      = kwargs.get("data")      if "data"      in kwargs else (args[0] if args else None)
        data_type = kwargs.get("data_type") if "data_type" in kwargs else None

        is_start_trigger = False
        if data_type == IOTypes.TRIGGER:
            if isinstance(data, str) and data.lower() in ["stop", "done", "aborted", "killed"]:
                is_start_trigger = False
            else:
                is_start_trigger = True
        elif isinstance(data, str) and data.upper() == "START":
            is_start_trigger = True

        if is_start_trigger:
            self._clear_downstream_plots()
            self.samples_buffers.clear()
            self._skip_first_live_frame = True
            if isinstance(data, str) and data.upper() == "START":
                self._reset_tracking_cb(None, None)
                try:
                    self.processor._reset_denseflow_state()
                except AttributeError:
                    pass
            return True

        if data_type == IOTypes.FILE_PATH:
            if isinstance(data, tuple) and len(data) == 2:
                path, name = data
            else:
                path, name = data, ""
            self._clear_downstream_plots()
            self.samples_buffers.clear()
            self._skip_first_live_frame = True
            if path and not self._batch_running:
                reprocess_active = dpg.get_value(self.reprocess_tag) if dpg.does_item_exist(self.reprocess_tag) else self.reprocess
                if reprocess_active:
                    processed_path = os.path.splitext(path)[0] + ".processed"
                    if os.path.exists(processed_path):
                        try:
                            with open(processed_path, "r", encoding="utf-8") as f:
                                loaded_meta = json.load(f)
                            loaded_params = loaded_meta.get("parameters", loaded_meta)
                            self._apply_imported_params(loaded_params)
                            from loguru import logger
                            logger.info(f"RadialContourTracker: Reprocess enabled Loaded settings from {processed_path}")
                        except Exception as e:
                            from loguru import logger
                            logger.error(f"RadialContourTracker: Failed to load {processed_path} for reprocessing: {e}")

                params_snapshot = self._get_current_params()
                if params_snapshot.get("export_next_as_video", False):
                    if dpg.does_item_exist(self.export_as_video_tag):
                        dpg.set_value(self.export_as_video_tag, False)
                    threading.Thread(
                        target=self._export_timelapse_video,
                        args=(path, name, params_snapshot),
                        daemon=True
                    ).start()
                else:
                    threading.Thread(
                        target=self._batch_process_video,
                        args=(path, name, params_snapshot),
                        daemon=True
                    ).start()
            return True

        if self._batch_running:
            return False

        frame_idx = None
        if isinstance(data, tuple) and len(data) >= 2:
            if len(data) == 3:
                frame, name, frame_idx = data
            else:
                frame, name = data
            if name:
                if self._source_name != name:
                    self._source_name = name
                    self._clear_downstream_plots()
                    self.samples_buffers.clear()
                    self._skip_first_live_frame = True
        else:
            frame = data

        if frame is None or not isinstance(frame, np.ndarray):
            return False
        if frame_idx is not None:
            return self.processor.submit((frame, frame_idx))
        return self.processor.submit(frame)

    def _get_current_params(self) -> dict:
        s_df = self.settings.get('denseflow', {})
        return {
            "processing_mode": dpg.get_value(self.mode_tag),
            "enable_preprocessing": dpg.get_value(self.enable_preprocessing_tag),
            "active": dpg.get_value(self.active_tag),
            
            "adjustment_mode": dpg.get_value(self.adjustment_mode_tag) if dpg.does_item_exist(self.adjustment_mode_tag) else self.adjustment_mode,
            "monochrome_mode": dpg.get_value(self.monochrome_tag),
            "invert_image": dpg.get_value(self.invert_image_tag) if dpg.does_item_exist(self.invert_image_tag) else self.invert_image,
            "gamma": float(dpg.get_value(self.gamma_tag)),
            "crop_type": dpg.get_value(self.crop_type_tag) if dpg.does_item_exist(self.crop_type_tag) else self.crop_type,
            "circular_crop": (dpg.get_value(self.crop_type_tag) == "Circular") if dpg.does_item_exist(self.crop_type_tag) else (self.crop_type == "Circular"),
            "crop_radius": dpg.get_value(self.crop_radius_tag) if dpg.does_item_exist(self.crop_radius_tag) else self.crop_radius,
            "crop_width": dpg.get_value(self.crop_width_tag) if dpg.does_item_exist(self.crop_width_tag) else self.crop_width,
            "crop_height": dpg.get_value(self.crop_height_tag) if dpg.does_item_exist(self.crop_height_tag) else self.crop_height,
            "show_crop_outline": dpg.get_value(self.show_crop_outline_tag) if dpg.does_item_exist(self.show_crop_outline_tag) else self.show_crop_outline,
            
            "crop_use_abs_center": dpg.get_value(self.crop_use_abs_tag) if dpg.does_item_exist(self.crop_use_abs_tag) else self.crop_use_abs_center,
            "crop_center_x": dpg.get_value(self.crop_center_x_tag) if dpg.does_item_exist(self.crop_center_x_tag) else self.crop_center_x,
            "crop_center_y": dpg.get_value(self.crop_center_y_tag) if dpg.does_item_exist(self.crop_center_y_tag) else self.crop_center_y,
            "crop_offset_x": dpg.get_value(self.crop_offset_x_tag) if dpg.does_item_exist(self.crop_offset_x_tag) else self.crop_offset_x,
            "crop_offset_y": dpg.get_value(self.crop_offset_y_tag) if dpg.does_item_exist(self.crop_offset_y_tag) else self.crop_offset_y,

            "pellet_use_abs_center": dpg.get_value(self.pellet_use_abs_tag) if dpg.does_item_exist(self.pellet_use_abs_tag) else self.pellet_use_abs_center,
            "pellet_center_x": dpg.get_value(self.pellet_center_x_tag) if dpg.does_item_exist(self.pellet_center_x_tag) else self.pellet_center_x,
            "pellet_center_y": dpg.get_value(self.pellet_center_y_tag) if dpg.does_item_exist(self.pellet_center_y_tag) else self.pellet_center_y,
            "pellet_offset_x": dpg.get_value(self.pellet_offset_x_tag) if dpg.does_item_exist(self.pellet_offset_x_tag) else self.pellet_offset_x,
            "pellet_offset_y": dpg.get_value(self.pellet_offset_y_tag) if dpg.does_item_exist(self.pellet_offset_y_tag) else self.pellet_offset_y,
            "pellet_radius": dpg.get_value(self.pellet_radius_tag) if dpg.does_item_exist(self.pellet_radius_tag) else self.pellet_radius,

            "remove_static": dpg.get_value(self.remove_static_tag),
            "similarity_threshold": dpg.get_value(self.similarity_threshold_tag),
            
            "show_distance_circles": dpg.get_value(self.show_circles_tag),
            "num_distance_circles": dpg.get_value(self.num_circles_tag),

            "blur": dpg.get_value(self.blur_tag),
            "block_size": dpg.get_value(self.block_size_tag),
            "bin_thresh": dpg.get_value(self.bin_thresh_tag),
            "morph_close": dpg.get_value(self.morph_close_tag),

            "enable_contour_detection": dpg.get_value(self.enable_contour_tag),
            "area_thresh_mode": dpg.get_value(self.area_thresh_mode_tag),
            "lower_surface_thresh": dpg.get_value(self.lower_str_tag),
            "upper_surface_thresh": dpg.get_value(self.upper_str_tag),
            "show_blobs": dpg.get_value(self.show_blobs_tag),
            "show_boxes": dpg.get_value(self.show_boxes_tag),
            "show_centroids": dpg.get_value(self.show_centroids_tag),
            "isolate_selection": dpg.get_value(self.isolate_tag),
            "fill_contours": dpg.get_value(self.fill_contours_tag),

            "enable_tracking": dpg.get_value(self.enable_tracking_tag),
            "distance_function": dpg.get_value(self.dist_func_tag),
            "distance_threshold": dpg.get_value(self.dist_thresh_tag),
            "trail_length": int(dpg.get_value(self.trail_length_tag)),
            "trail_thickness": int(dpg.get_value(self.trail_thickness_tag)),
            "show_trail": dpg.get_value(self.show_trail_tag),
            "show_id": dpg.get_value(self.show_id_tag),
            "show_age": dpg.get_value(self.show_age_tag),
            "show_distance": dpg.get_value(self.show_distance_tag),
            "show_speed": dpg.get_value(self.show_speed_tag),
            "speed_window_frames": int(dpg.get_value(self.speed_window_tag)) if dpg.does_item_exist(self.speed_window_tag) else 5,
            "show_proximity": dpg.get_value(self.show_proximity_tag),
            "enable_proximity": dpg.get_value(self.enable_proximity_tag),
            "proximity_density_mode": dpg.get_value(self.proximity_mode_tag),
            "proximity_scale_mode": dpg.get_value(self.proximity_scale_mode_tag),
            "proximity_palette": dpg.get_value(self.proximity_palette_tag),
            "proximity_opacity": float(dpg.get_value(self.proximity_opacity_tag)),
            "proximity_show_scale": dpg.get_value(self.proximity_show_scale_tag),
            "proximity_scale_position": dpg.get_value(self.proximity_scale_pos_tag),
            "proximity_scale_size": int(dpg.get_value(self.proximity_scale_size_tag)),

            "export_next_as_video": dpg.get_value(self.export_as_video_tag),
            "crop_to_selection": dpg.get_value(self.crop_to_selection_tag),
            "no_subsampling": dpg.get_value(self.no_subsampling_tag),
            "target_duration": float(dpg.get_value(self.video_duration_tag)),
            "output_fps": int(dpg.get_value(self.video_fps_tag)),

            "df_calibration": s_df.get("calibration", 1.0),
            "df_fps": s_df.get("fps", 30.0),
            "df_pyr_scale": s_df.get("pyr_scale", 0.5),
            "df_levels": s_df.get("levels", 2),
            "df_winsize": s_df.get("winsize", 9),
            "df_iterations": s_df.get("iterations", 2),
            "df_poly_n": s_df.get("poly_n", 5),
            "df_poly_sigma": s_df.get("poly_sigma", 1.1),
            "df_lower_mag_thresh": dpg.get_value(self.df_mag_thresh_tag),
            "df_erosion": dpg.get_value(self.df_erosion_tag),
            "df_supersat": dpg.get_value(self.df_supersat_tag),
            "df_use_contours": dpg.get_value(self.df_use_contours_tag),
            "df_show_arrows": dpg.get_value(self.df_show_arrows_tag),
            "df_isolate_selection": dpg.get_value(self.df_isolate_tag),
            "df_lower_surface_thresh": dpg.get_value(self.df_lower_str_tag),
            "df_upper_surface_thresh": dpg.get_value(self.df_upper_str_tag),
            "both_vis_mode": dpg.get_value(self.both_vis_tag),

            "active": dpg.get_value(self.active_tag),
            "collect_raw_speeds": len(self.connections.get("Raw Speeds", [])) > 0,
            "static_speed_thresh": float(dpg.get_value(self.static_speed_thresh_tag)) if dpg.does_item_exist(self.static_speed_thresh_tag) else self.static_speed_thresh,
            "kymograph_bin_width": int(dpg.get_value(self.kymograph_bin_width_tag)) if dpg.does_item_exist(self.kymograph_bin_width_tag) else self.kymograph_bin_width,
            "kymograph_num_bins": int(dpg.get_value(self.kymograph_num_bins_tag)) if dpg.does_item_exist(self.kymograph_num_bins_tag) else self.kymograph_num_bins,
            "auto_export_csv": dpg.get_value(self.auto_export_csv_tag) if dpg.does_item_exist(self.auto_export_csv_tag) else self.auto_export_csv,
            "reprocess": dpg.get_value(self.reprocess_tag) if dpg.does_item_exist(self.reprocess_tag) else self.reprocess,
            "spatial_scale": float(dpg.get_value(self.spatial_scale_tag)) if dpg.does_item_exist(self.spatial_scale_tag) else self.spatial_scale,
            "spatial_unit": str(dpg.get_value(self.spatial_unit_tag)) if dpg.does_item_exist(self.spatial_unit_tag) else self.spatial_unit,
        }

    def _set_batch_lock(self, locked: bool):
        self._batch_running = locked
        try:
            dpg.configure_item(self.main_settings_group, show=not locked)
            dpg.configure_item(f"batch_inputs_{self.UUID}", show=not locked)
            dpg.configure_item(f"batch_progress_group_{self.UUID}", show=locked)
        except Exception:
            pass

    def _batch_process_video(self, video_path: str, video_name: str, params: dict):
        import cv2
        import uuid as _uuid
        from concurrent.futures import ProcessPoolExecutor, as_completed

        source = video_name or self.label
        self._source_name = source
        mode = params.get('processing_mode', 'contour')

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total_frames <= 0:
            from loguru import logger
            logger.error(f"RadialContourTracker batch: could not read {video_path}")
            trigger_modules = self.connections.get("Trigger", [])
            for module in trigger_modules:
                try:
                    module.input_cb(data=video_path, data_type=IOTypes.TRIGGER)
                except Exception:
                    pass
            return

        chunk_size = max(1, dpg.get_value(self.chunk_size_tag))
        n_threads  = max(1, dpg.get_value(self.n_threads_tag))
        chunks     = [(s, min(s + chunk_size, total_frames))
                      for s in range(0, total_frames, chunk_size)]
        n_chunks   = len(chunks)

        self._set_batch_lock(True)
        dpg.set_value(self.batch_progress_tag, 0.0)
        dpg.set_value(self.batch_status_tag, f"Starting… {total_frames} frames")
        dpg.set_value(self.batch_time_tag, "Time: 0s | ETA: --")
        
        start_time_process = time.time()
        ordered_results: list = [None] * n_chunks
        completed = 0

        # Inject references to batch params for static removal support
        params["_ref_frame_1"] = self.processor._ref_frame_1
        params["_ref_frame_2"] = self.processor._ref_frame_2

        with ProcessPoolExecutor(max_workers=n_threads) as exe:
            worker_fn = (
                batch_worker_denseflow_chunk if mode == 'denseflow'
                else batch_worker_both_chunk if mode == 'both'
                else batch_worker_process_chunk
            )
            futures = {
                exe.submit(worker_fn, video_path, s, e, params): idx
                for idx, (s, e) in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as exc:
                    from loguru import logger
                    logger.error(f"RadialContourTracker batch chunk {idx} error: {exc}")
                    ordered_results[idx] = []
                
                completed += 1
                now = time.time()
                elapsed = now - start_time_process
                progress = completed / n_chunks
                
                if progress > 0:
                    total_estimated = elapsed / progress
                    remaining = total_estimated - elapsed
                    def fmt_time(s):
                        if s < 60: return f"{int(s)}s"
                        return f"{int(s//60)}m {int(s%60)}s"
                    dpg.set_value(self.batch_time_tag, f"Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(remaining)}")

                dpg.set_value(self.batch_progress_tag, progress)
                dpg.set_value(self.batch_status_tag,
                              f"Processing: {completed}/{n_chunks} chunks ({int(progress*100)}%)")

        flat = [item for chunk in ordered_results if chunk for item in chunk]
        dpg.set_value(self.batch_status_tag, "Saving and emitting results…")

        result_modules = self.connections.get("Result", [])

        if not params.get('active', True):
            dpg.set_value(self.batch_progress_tag, 1.0)
            dpg.set_value(self.batch_status_tag, "Done – processing was disabled")
            self._set_batch_lock(False)
            trigger_modules = self.connections.get("Trigger", [])
            for module in trigger_modules:
                try:
                    module.input_cb(data=video_path, data_type=IOTypes.TRIGGER)
                except Exception:
                    pass
            return

        # Collect per-frame object detections: {frame_idx: [{'id': id, 'x': x, 'y': y, 'dist': dist, 'speed': speed}, ...]}
        frame_detections = {}
        calib = float(params.get('df_calibration', 1.0))
        spatial_scale = float(params.get('spatial_scale', 1.0))
        spatial_unit = str(params.get('spatial_unit', 'µm'))
        static_speed_thresh = float(params.get('static_speed_thresh', 0.5))

        if params.get('pellet_use_abs_center', params.get('use_absolute_center', False)):
            pellet_center_x = int(params.get('pellet_center_x', params.get('center_x', width // 2)))
            pellet_center_y = int(params.get('pellet_center_y', params.get('center_y', height // 2)))
        else:
            offset_x = int(params.get('pellet_offset_x', params.get('offset_x', 0)))
            offset_y = int(params.get('pellet_offset_y', params.get('offset_y', 0)))
            pellet_center_x = width // 2 + offset_x
            pellet_center_y = height // 2 + offset_y

        pellet_radius = int(params.get('pellet_radius', 0))

        if mode == 'contour':
            from norfair import Tracker, Detection
            tracker = Tracker(
                distance_function=params.get('distance_function', 'euclidean'),
                distance_threshold=params.get('distance_threshold', 30)
            )
            
            curves = {}
            track_history = {}  # {obj_id: [(frame_idx, cx, cy), ...]}
            speed_window = max(2, int(params.get('trail_length', 30)))

            dpg.set_value(self.batch_progress_tag, 0.0)
            dpg.set_value(self.batch_status_tag, f"Tracking objects across {len(flat)} frames…")
            total_flat = max(1, len(flat))
            for frame_idx, centroids in enumerate(flat):
                track_progress = (frame_idx + 1) / total_flat
                dpg.set_value(self.batch_progress_tag, track_progress)
                if frame_idx % 50 == 0 or frame_idx == total_flat - 1:
                    dpg.set_value(self.batch_status_tag, f"Tracking objects: frame {frame_idx + 1}/{total_flat} ({int(track_progress * 100)}% | {len(curves)} tracks)")
                norfair_detections = [Detection(np.array(pt)) for pt in centroids]
                tracked_objects = tracker.update(norfair_detections)
                
                frame_dets = []
                for obj in tracked_objects:
                    cx_obj, cy_obj = map(int, obj.estimate[0])
                    dist_center = float(np.sqrt((cx_obj - pellet_center_x) ** 2 + (cy_obj - pellet_center_y) ** 2))
                    dist = max(0.0, dist_center - pellet_radius)
                    
                    if obj.id not in track_history:
                        track_history[obj.id] = []
                    track_history[obj.id].append((frame_idx, cx_obj, cy_obj))

                    # Calculate smoothed speed over the last speed_window frames along the actual trajectory path
                    pts = track_history[obj.id][-speed_window:]
                    if len(pts) >= 2:
                        d_frames = pts[-1][0] - pts[0][0]
                        if d_frames > 0:
                            path_len_px = sum(
                                float(np.sqrt((pts[k][1] - pts[k-1][1])**2 + (pts[k][2] - pts[k-1][2])**2))
                                for k in range(1, len(pts))
                            )
                            speed_val = round((path_len_px * calib * video_fps) / d_frames, 3)
                        else:
                            speed_val = 0.0
                    else:
                        speed_val = 0.0

                    if obj.id not in curves:
                        curves[obj.id] = ([], [])
                    curves[obj.id][0].append(frame_idx)
                    curves[obj.id][1].append(dist)

                    frame_dets.append({
                        'id': obj.id,
                        'x': cx_obj,
                        'y': cy_obj,
                        'dist': round(dist, 2),
                        'speed': speed_val
                    })
                frame_detections[frame_idx] = frame_dets

            is_tuning = params.get('adjustment_mode', False)

            if not is_tuning:
                dpg.set_value(self.batch_status_tag, "Emitting tracked samples to downstream modules…")
                for obj_id, (frame_idxs, dists) in curves.items():
                    x_values = self._build_x_axis(frame_idxs, has_frame_idx=True)
                    sample_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"{self.UUID}_{source}_{obj_id}"))
                    sample = {
                        "uuid": sample_uuid,
                        "name": f"Contour_{obj_id}_{source}",
                        "x": x_values,
                        "y": dists,
                        "action": "select",
                    }
                    for module in result_modules:
                        module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)

            dpg.set_value(self.metric_text_tag, f"{len(curves)} contours tracked")

        elif mode == 'both':
            x_values = self._build_x_axis([float(item[0]) / video_fps for item in flat])
            y_values = [float(item[1]) for item in flat]
            
            for item in flat:
                if len(item) >= 4:
                    frame_idx, _, _, objects_data = item[:4]
                    frame_dets = []
                    for idx, (cx_obj, cy_obj, dist, spd) in enumerate(objects_data):
                        frame_dets.append({
                            'id': idx,
                            'x': cx_obj,
                            'y': cy_obj,
                            'dist': round(dist, 2),
                            'speed': round(spd, 3)
                        })
                    frame_detections[frame_idx] = frame_dets

            is_tuning = params.get('adjustment_mode', False)
            if not is_tuning:
                sample_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"{self.UUID}_{source}"))
                sample = {
                    "uuid": sample_uuid,
                    "name": source,
                    "x": x_values,
                    "y": y_values,
                    "action": "select",
                }
                for module in result_modules:
                    module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)

            if flat:
                dpg.set_value(self.metric_text_tag, f"{y_values[-1]:.3f} µm/s")

        elif mode == 'denseflow':
            x_values = self._build_x_axis([float(item[0]) / video_fps for item in flat])
            y_values = [float(item[1]) for item in flat]
            
            is_tuning = params.get('adjustment_mode', False)
            if not is_tuning:
                sample_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"{self.UUID}_{source}"))
                sample = {
                    "uuid": sample_uuid,
                    "name": source,
                    "x": x_values,
                    "y": y_values,
                    "action": "select",
                }
                for module in result_modules:
                    module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)

            if flat:
                dpg.set_value(self.metric_text_tag, f"{y_values[-1]:.3f} µm/s")

            auto_export = bool(params.get('auto_export_csv', self.auto_export_csv))
            if auto_export:
                csv_path = os.path.splitext(video_path)[0] + "_radial.csv"
                try:
                    import csv
                    headers = [self.x_axis_mode, f"Speed_{source}"]
                    rows = zip(x_values, y_values)
                    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerows(rows)
                    from loguru import logger
                    logger.info(f"RadialContourTracker: Exported CSV to {csv_path}")
                except Exception as e:
                    from loguru import logger
                    logger.error(f"RadialContourTracker: CSV write error: {e}")

        # --- Enhanced Radial Detections CSV Export ---
        auto_export = bool(params.get('auto_export_csv', self.auto_export_csv))
        if auto_export and frame_detections:
            dt = 1.0 / video_fps

            # Export *_radial_detections.csv (Single source of truth for detections and kymographs)
            det_csv_path = os.path.splitext(video_path)[0] + "_radial_detections.csv"
            try:
                import csv
                det_headers = [
                    "Frame", "Time_s", "Track_ID",
                    "X_px", "Y_px",
                    "Distance_px", f"Distance_{spatial_unit}",
                    f"Speed_{spatial_unit}_s",
                    f"Scale_{spatial_unit}_per_px",
                    "Pellet_Center_X_px", "Pellet_Center_Y_px", "Pellet_Radius_px"
                ]
                det_rows = []
                for f_idx in sorted(frame_detections.keys()):
                    t_sec = round(f_idx * dt, 4)
                    for det in frame_detections[f_idx]:
                        dist_unit = round(det['dist'] * spatial_scale, 2)
                        det_rows.append([
                            f_idx, t_sec, det['id'],
                            det['x'], det['y'],
                            det['dist'], dist_unit,
                            det['speed'],
                            spatial_scale,
                            pellet_center_x, pellet_center_y, pellet_radius
                        ])
                
                with open(det_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(det_headers)
                    writer.writerows(det_rows)
                from loguru import logger
                logger.info(f"RadialContourTracker: Exported detections CSV to {det_csv_path}")
            except Exception as e:
                from loguru import logger
                logger.error(f"RadialContourTracker: Detections CSV write error: {e}")

        processed_path = os.path.splitext(video_path)[0] + ".processed"
        try:
            from datetime import datetime
            metadata = {
                "mode": mode,
                "parameters": params,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
        except Exception:
            pass

        dpg.set_value(self.batch_progress_tag, 1.0)
        if params.get('adjustment_mode', False):
            dpg.set_value(self.batch_status_tag, "Done – Mode Ajustement (Envoi aux modules désactivé)")
        else:
            dpg.set_value(self.batch_status_tag, f"Done – {len(flat)} frames processed")
        self._set_batch_lock(False)

        trigger_modules = self.connections.get("Trigger", [])
        for module in trigger_modules:
            try:
                module.input_cb(data=video_path, data_type=IOTypes.TRIGGER)
            except Exception:
                pass

    def _export_timelapse_video(self, video_path: str, video_name: str, params: dict):
        import cv2
        import time

        source = video_name or self.label
        self._source_name = source

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or width <= 0 or height <= 0:
            cap.release()
            from loguru import logger
            logger.error(f"RadialContourTracker Timelapse: could not read video {video_path}")
            trigger_modules = self.connections.get("Trigger", [])
            for module in trigger_modules:
                try:
                    module.input_cb(data=video_path, data_type=IOTypes.TRIGGER)
                except Exception:
                    pass
            return

        target_duration = max(1.0, float(params.get("target_duration", 30.0)))
        out_fps = max(1, int(params.get("output_fps", 30)))
        total_out_frames = max(1, int(round(target_duration * out_fps)))

        if total_frames > total_out_frames:
            sampled_indices = np.linspace(0, total_frames - 1, total_out_frames, dtype=int)
        else:
            sampled_indices = np.arange(total_frames, dtype=int)
            total_out_frames = len(sampled_indices)

        base, _ = os.path.splitext(video_path)
        output_video_path = f"{base}_timelapse.mp4"

        is_crop_to_selection = bool(params.get("crop_to_selection", False))

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sampled_indices[0]))
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, test_frame = cap.read()

        params_render = dict(params)
        params_render["_ref_frame_1"] = self.processor._ref_frame_1
        params_render["_ref_frame_2"] = self.processor._ref_frame_2

        if params.get('enable_tracking', True):
            from norfair import Tracker
            video_tracker = Tracker(
                distance_function=params.get('distance_function', 'euclidean'),
                distance_threshold=params.get('distance_threshold', 30)
            )
            video_tracking_dict = {}
        else:
            video_tracker = None
            video_tracking_dict = None

        sample_out = render_processed_frame(test_frame, params_render, is_crop_to_selection=is_crop_to_selection)
        out_h, out_w = sample_out.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, (out_w, out_h))

        self._set_batch_lock(True)
        dpg.set_value(self.batch_progress_tag, 0.0)
        dpg.set_value(self.batch_status_tag, f"Exporting timelapse: 0/{total_out_frames} frames…")
        dpg.set_value(self.batch_time_tag, "Time: 0s | ETA: --")

        start_time_export = time.time()

        no_subsampling = bool(params.get("no_subsampling", False))

        if no_subsampling:
            save_frame_set = set(sampled_indices)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            written_count = 0
            for f_idx in range(total_frames):
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    break

                should_save = f_idx in save_frame_set
                if should_save:
                    processed_frame = render_processed_frame(
                        raw_frame, params_render,
                        is_crop_to_selection=is_crop_to_selection,
                        tracker=video_tracker,
                        tracking_dict=video_tracking_dict
                    )
                    if processed_frame is not None:
                        if processed_frame.shape[:2] != (out_h, out_w):
                            processed_frame = cv2.resize(processed_frame, (out_w, out_h))
                        writer.write(processed_frame)
                        written_count += 1
                else:
                    if video_tracker is not None:
                        centroids = RadialContourProcessor.process_frame_stateless(raw_frame, params_render)
                        if centroids:
                            from norfair import Detection
                            norfair_dets = [Detection(np.array(pt)) for pt in centroids]
                            tracked = video_tracker.update(norfair_dets)
                            for obj in tracked:
                                cx_o, cy_o = map(int, obj.estimate[0])
                                if obj.id not in video_tracking_dict:
                                    video_tracking_dict[obj.id] = {
                                        "color": tuple(int(x) for x in np.random.randint(50, 255, size=3)),
                                        "points": [],
                                        "distance": 0.0,
                                        "age": obj.age
                                    }
                                video_tracking_dict[obj.id]["points"].append((cx_o, cy_o))
                                video_tracking_dict[obj.id]["distance"] += getattr(obj, "last_distance", 0.0)
                                video_tracking_dict[obj.id]["age"] = obj.age

                progress = (f_idx + 1) / total_frames
                now = time.time()
                elapsed = now - start_time_export
                total_est = elapsed / max(1e-5, progress)
                remaining = max(0.0, total_est - elapsed)
                def fmt_time(s):
                    if s < 60: return f"{int(s)}s"
                    return f"{int(s//60)}m {int(s%60)}s"

                if f_idx % 10 == 0 or f_idx == total_frames - 1:
                    dpg.set_value(self.batch_progress_tag, progress)
                    dpg.set_value(self.batch_status_tag, f"Exporting timelapse (Full Tracking): {f_idx + 1}/{total_frames} (saved {written_count}/{total_out_frames})")
                    dpg.set_value(self.batch_time_tag, f"Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(remaining)}")
        else:
            for out_idx, frame_idx in enumerate(sampled_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    continue

                processed_frame = render_processed_frame(
                    raw_frame, params_render,
                    is_crop_to_selection=is_crop_to_selection,
                    tracker=video_tracker,
                    tracking_dict=video_tracking_dict
                )
                if processed_frame is not None:
                    if processed_frame.shape[:2] != (out_h, out_w):
                        processed_frame = cv2.resize(processed_frame, (out_w, out_h))
                    writer.write(processed_frame)

                progress = (out_idx + 1) / total_out_frames
                now = time.time()
                elapsed = now - start_time_export
                total_est = elapsed / max(1e-5, progress)
                remaining = max(0.0, total_est - elapsed)
                def fmt_time(s):
                    if s < 60: return f"{int(s)}s"
                    return f"{int(s//60)}m {int(s%60)}s"

                dpg.set_value(self.batch_progress_tag, progress)
                dpg.set_value(self.batch_status_tag, f"Exporting timelapse: {out_idx + 1}/{total_out_frames} frames ({int(progress * 100)}%)")
                dpg.set_value(self.batch_time_tag, f"Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(remaining)}")

        writer.release()
        cap.release()

        dpg.set_value(self.batch_progress_tag, 1.0)
        dpg.set_value(self.batch_status_tag, f"Timelapse exported: {os.path.basename(output_video_path)}")
        self._set_batch_lock(False)

        from loguru import logger
        logger.success(f"Timelapse video exported successfully to {output_video_path}")

        trigger_modules = self.connections.get("Trigger", [])
        for module in trigger_modules:
            try:
                module.input_cb(data=output_video_path, data_type=IOTypes.TRIGGER)
            except Exception:
                pass

    def _monitor_loop(self):
        while self._monitor_thread_running:
            result = self.processor.get()
            if result is not None:
                if isinstance(result, tuple) and len(result) == 2 and not isinstance(result[0], np.ndarray):
                    res, frame_idx = result
                else:
                    res, frame_idx = result, None

                frame, tracking, contours, mask, distances = res
                
                # Update status of reference frames
                r1_exists = self.processor._ref_frame_1 is not None
                r2_exists = self.processor._ref_frame_2 is not None
                r1_text = "Frame 1: Captured" if r1_exists else "Frame 1: Empty"
                r2_text = "Frame 2: Captured" if r2_exists else "Frame 2: Empty"
                
                dpg.set_value(self.r1_status_tag, r1_text)
                dpg.configure_item(self.r1_status_tag, color=(100, 255, 100) if r1_exists else (255, 100, 100))
                dpg.set_value(self.r2_status_tag, r2_text)
                dpg.configure_item(self.r2_status_tag, color=(100, 255, 100) if r2_exists else (255, 100, 100))

                # Update metrics in UI
                mode = dpg.get_value(self.mode_tag)
                active_processing = dpg.get_value(self.active_tag)
                if not active_processing:
                    dpg.set_value(self.metric_text_tag, "Processing Disabled")
                elif mode == 'contour':
                    if distances:
                        avg_dist = np.mean([d[1] for d in distances])
                        dpg.set_value(self.metric_text_tag, f"{len(distances)} contours | Avg: {avg_dist:.1f} px")
                    else:
                        dpg.set_value(self.metric_text_tag, "0 contours")
                elif mode == 'denseflow' and distances:
                    dpg.set_value(self.metric_text_tag, f"{distances[0][1]:.3f} µm/s")
                elif mode == 'both':
                    cnt_count = len(distances)
                    dpg.set_value(self.metric_text_tag, f"{cnt_count} contours")

                # Dispatch outputs
                import uuid as _uuid
                source = self._source_name or self.label
                timestamp = time.time()

                for idx, output_key in enumerate(self.outputs):
                    connected = self.connections.get(output_key, [])
                    if not connected:
                        continue
                    for module in connected:
                        if idx == 0:
                            module.input_cb(data=frame, data_type=IOTypes.FRAME)
                        elif idx == 1:
                            if tracking is not None and tracking != {}:
                                module.input_cb(data=tracking, data_type=IOTypes.TRACKING)
                        elif idx == 2:
                            if contours is not None:
                                module.input_cb(data=contours, data_type=IOTypes.CONTOURS)

                # Accumulate and emit Result samples
                if self._skip_first_live_frame:
                    self._skip_first_live_frame = False
                elif active_processing and not dpg.get_value(self.adjustment_mode_tag):
                    if mode == 'contour':
                        for obj_id, dist in distances:
                            sample_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"{self.UUID}_{source}_{obj_id}"))
                            self._accumulate_point(sample_uuid, f"Contour_{obj_id}_{source}", dist, timestamp, frame_idx)
                            self._emit_full_history(sample_uuid)
                    elif mode == 'denseflow' and distances:
                        speed_val = distances[0][1]
                        sample_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"{self.UUID}_{source}"))
                        self._accumulate_point(sample_uuid, source, speed_val, timestamp, frame_idx)
                        self._emit_full_history(sample_uuid)
                    elif mode == 'both':
                        for obj_id, dist in distances:
                            sample_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"{self.UUID}_{source}_{obj_id}"))
                            self._accumulate_point(sample_uuid, f"Contour_{obj_id}_{source}", dist, timestamp, frame_idx)
                            self._emit_full_history(sample_uuid)
            else:
                time.sleep(0.005)


# Registry metadata
RadialContourTrackerWin = radial_contour_tracker_win
EXPORTED_CLASS = radial_contour_tracker_win
EXPORTED_NAME = "Radial Contour Tracker"
