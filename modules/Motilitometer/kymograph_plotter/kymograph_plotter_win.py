import os
import threading
from typing import Any
import numpy as np
import dearpygui.dearpygui as dpg
from loguru import logger

from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.file_explorer import file_explorer
from core.app_state import app_state
from config.display_scaling import display_scaling

AVAILABLE_CMAPS = [
    "magma", "viridis", "plasma", "inferno", "cividis", "turbo",
    "coolwarm", "jet", "Blues", "Reds", "Greens", "hot", "gray", "copper", "gist_earth"
]


def parse_detections_data(file_path: str, num_bins: int = 50, static_thresh: float = 0.5,
                          smooth_time: float = 3.0, smooth_space: float = 2.0,
                          n_time_steps: int = 40, fps: float = 30.0):
    """
    Parses detection CSV and produces:
    - 2D continuous matrices and 3D surface grid matrices for Total, Moving, and Static subsets.
    - Raw speeds array for histogram distribution.
    - Speed vs Time series (mean speed per frame/time step).
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception:
        import csv
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader)]
            rows = list(reader)
        data = []
        for r in rows:
            if r:
                try:
                    data.append([float(x) if x != '' else 0.0 for x in r])
                except ValueError:
                    pass
        data_arr = np.array(data)
        df = {h: data_arr[:, i] for i, h in enumerate(headers)}

    cols = {str(col).lower().strip(): col for col in df.keys()}
    
    frame_col = next((cols[c] for c in cols if 'frame' in c), None)
    time_col = next((cols[c] for c in cols if 'time' in c), None)
    if time_col is None and frame_col is not None:
        time_col = frame_col

    dist_col = next((cols[c] for c in cols if 'distance_µm' in c or 'distance_um' in c or 'distance_px' in c or 'distance' in c), None)
    speed_col = next((cols[c] for c in cols if 'speed' in c), None)
    static_col = next((cols[c] for c in cols if 'is_static' in c or 'static' in c), None)

    if dist_col is None:
        raise ValueError("Could not find Distance column in CSV.")

    unit_name = "px"
    if "µm" in str(dist_col) or "um" in str(dist_col):
        unit_name = "µm"

    if time_col is not None:
        raw_times = df[time_col].to_numpy() if hasattr(df[time_col], 'to_numpy') else np.array(df[time_col])
        if 'frame' in str(time_col).lower() and 'time' not in str(time_col).lower() and fps > 0:
            times = raw_times / float(fps)
        else:
            times = raw_times
    else:
        times = np.arange(len(df[dist_col])) / float(fps if fps > 0 else 30.0)

    dists = df[dist_col].to_numpy() if hasattr(df[dist_col], 'to_numpy') else np.array(df[dist_col])
    
    if speed_col is not None:
        speeds = df[speed_col].to_numpy() if hasattr(df[speed_col], 'to_numpy') else np.array(df[speed_col])
        is_static = (speeds < static_thresh)
        raw_speeds = speeds[~np.isnan(speeds)]
    elif static_col is not None:
        is_static = (df[static_col].to_numpy() == 1) if hasattr(df[static_col], 'to_numpy') else (np.array(df[static_col]) == 1)
        raw_speeds = np.array([])
    else:
        is_static = np.zeros_like(dists, dtype=bool)
        raw_speeds = np.array([])

    max_dist = max(1.0, float(np.nanmax(dists)))
    min_time = float(np.nanmin(times))
    max_time = float(np.nanmax(times))

    # --- 1. Continuous 2D Matrices (Frame/Time granularity) ---
    unique_times, time_indices = np.unique(times, return_inverse=True)
    num_times = len(unique_times)
    time_bins = np.arange(num_times + 1) - 0.5
    dist_bins = np.linspace(0, max_dist, num_bins + 1)

    tot_mat_2d, _, _ = np.histogram2d(time_indices, dists, bins=[time_bins, dist_bins])
    mov_mat_2d, _, _ = np.histogram2d(time_indices[~is_static], dists[~is_static], bins=[time_bins, dist_bins])
    stat_mat_2d, _, _ = np.histogram2d(time_indices[is_static], dists[is_static], bins=[time_bins, dist_bins])

    # --- 2. 3D Sampled Surface Matrices ---
    n_time_steps = max(5, int(n_time_steps))
    sampled_times_3d = np.linspace(min_time, max_time, n_time_steps)
    dist_centers_3d = 0.5 * (dist_bins[:-1] + dist_bins[1:])
    time_bin_edges_3d = np.linspace(min_time, max_time, n_time_steps + 1)

    total_unique_frames = len(unique_times)
    frames_per_window = max(1.0, float(total_unique_frames) / float(n_time_steps))

    tot_mat_3d, _, _ = np.histogram2d(times, dists, bins=[time_bin_edges_3d, dist_bins])
    mov_mat_3d, _, _ = np.histogram2d(times[~is_static], dists[~is_static], bins=[time_bin_edges_3d, dist_bins])
    stat_mat_3d, _, _ = np.histogram2d(times[is_static], dists[is_static], bins=[time_bin_edges_3d, dist_bins])

    tot_mat_3d = tot_mat_3d / frames_per_window
    mov_mat_3d = mov_mat_3d / frames_per_window
    stat_mat_3d = stat_mat_3d / frames_per_window

    # --- 3. 2D Gaussian Kernel Smoothing ---
    if smooth_time > 0 or smooth_space > 0:
        try:
            from scipy.ndimage import gaussian_filter
            tot_mat_2d = gaussian_filter(tot_mat_2d, sigma=(smooth_time, smooth_space))
            mov_mat_2d = gaussian_filter(mov_mat_2d, sigma=(smooth_time, smooth_space))
            stat_mat_2d = gaussian_filter(stat_mat_2d, sigma=(smooth_time, smooth_space))

            tot_mat_3d = gaussian_filter(tot_mat_3d, sigma=(smooth_time, smooth_space))
            mov_mat_3d = gaussian_filter(mov_mat_3d, sigma=(smooth_time, smooth_space))
            stat_mat_3d = gaussian_filter(stat_mat_3d, sigma=(smooth_time, smooth_space))
        except ImportError:
            pass

    # --- 4. Speed vs Time Series (Mean speed per frame/time) ---
    time_series_x = unique_times
    time_series_y = np.zeros(len(unique_times), dtype=float)
    if speed_col is not None and len(speeds) > 0:
        for idx in range(len(unique_times)):
            mask = (time_indices == idx)
            s_frame = speeds[mask]
            valid_s = s_frame[~np.isnan(s_frame)]
            if len(valid_s) > 0:
                time_series_y[idx] = float(np.mean(valid_s))

    # --- 5. Raw Time & Speed Pairs for Boxplot ---
    if speed_col is not None and len(speeds) > 0:
        valid_mask = ~np.isnan(speeds)
        box_times = times[valid_mask]
        box_speeds = speeds[valid_mask]
    else:
        box_times = np.array([])
        box_speeds = np.array([])

    return {
        "unit_name": unit_name,
        "max_dist": max_dist,
        "times_2d": unique_times,
        "tot_mat_2d": tot_mat_2d,
        "mov_mat_2d": mov_mat_2d,
        "stat_mat_2d": stat_mat_2d,
        "sampled_times_3d": sampled_times_3d,
        "dist_centers_3d": dist_centers_3d,
        "tot_mat_3d": tot_mat_3d,
        "mov_mat_3d": mov_mat_3d,
        "stat_mat_3d": stat_mat_3d,
        "raw_speeds": raw_speeds,
        "time_series_x": time_series_x,
        "time_series_y": time_series_y,
        "box_times": box_times,
        "box_speeds": box_speeds,
    }


class Kymograph_plotter_win(WindowBase):
    """
    Kymograph ploter for radial analysis
    """
    # 1. Window Configuration
    label = "Kymograph Plotter"
    description = "Generates 2D and 3D spatio-temporal kymographs, speed histograms, and distributions from motilitometer tracking data."
    win_width = display_scaling.scale(470)
    win_height = display_scaling.scale(310)

    # 2. Input/Output definitions
    accepted_input_types = [IOTypes.FILE_PATH, IOTypes.TRIGGER]
    outputs = {
        "Frame": IOTypes.FRAME,
        "Speed Histogram": IOTypes.SAMPLE,
        "Speed vs Time": IOTypes.SAMPLE,
        "Speed Boxplot Data": IOTypes.SAMPLE,
        "Trigger": IOTypes.TRIGGER,
    }

    # 3. Persistent fields saved into pipeline JSON
    persistent_fields = [
        "label", "csv_path", "plot_title", "plot_mode", "num_bins", "static_thresh",
        "smooth_time", "smooth_space", "interpolation",
        "plot_total", "plot_moving", "plot_static",
        "cmap_total", "cmap_moving", "cmap_static", "same_scale", "fps",
        "n_time_steps_3d", "camera_elev", "camera_azim", "export_html_3d",
        "auto_save_png", "dpi",
    ]

    # Class-level defaults for persistent fields
    csv_path: str = ""
    plot_title: str = ""
    plot_mode: str = "2D Continuous Heatmap"
    num_bins: int = 50
    static_thresh: float = 0.5
    smooth_time: float = 3.0
    smooth_space: float = 2.0
    interpolation: str = "bicubic"
    plot_total: bool = True
    plot_moving: bool = True
    plot_static: bool = True
    cmap_total: str = "magma"
    cmap_moving: str = "viridis"
    cmap_static: str = "cividis"
    same_scale: bool = False
    fps: float = 30.0
    n_time_steps_3d: int = 40
    camera_elev: float = 30.0
    camera_azim: float = -60.0
    export_html_3d: bool = False
    auto_save_png: bool = True
    dpi: int = 200

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        scale = display_scaling.scale

        # Defensive extraction of persistent fields
        self.csv_path = str(getattr(self, "csv_path", kwargs.get("csv_path", "")) or "")
        self.plot_title = str(getattr(self, "plot_title", kwargs.get("plot_title", "")) or "")
        self.plot_mode = str(getattr(self, "plot_mode", kwargs.get("plot_mode", "2D Continuous Heatmap")) or "2D Continuous Heatmap")
        try:
            self.num_bins = int(getattr(self, "num_bins", kwargs.get("num_bins", 50)))
        except (ValueError, TypeError):
            self.num_bins = 50
        try:
            self.static_thresh = float(getattr(self, "static_thresh", kwargs.get("static_thresh", 0.5)))
        except (ValueError, TypeError):
            self.static_thresh = 0.5
        try:
            self.smooth_time = float(getattr(self, "smooth_time", kwargs.get("smooth_time", 3.0)))
        except (ValueError, TypeError):
            self.smooth_time = 3.0
        try:
            self.smooth_space = float(getattr(self, "smooth_space", kwargs.get("smooth_space", 2.0)))
        except (ValueError, TypeError):
            self.smooth_space = 2.0
        self.interpolation = str(getattr(self, "interpolation", kwargs.get("interpolation", "bicubic")) or "bicubic")
        self.plot_total = bool(getattr(self, "plot_total", kwargs.get("plot_total", True)))
        self.plot_moving = bool(getattr(self, "plot_moving", kwargs.get("plot_moving", True)))
        self.plot_static = bool(getattr(self, "plot_static", kwargs.get("plot_static", True)))
        self.cmap_total = str(getattr(self, "cmap_total", kwargs.get("cmap_total", "magma")) or "magma")
        self.cmap_moving = str(getattr(self, "cmap_moving", kwargs.get("cmap_moving", "viridis")) or "viridis")
        self.cmap_static = str(getattr(self, "cmap_static", kwargs.get("cmap_static", "cividis")) or "cividis")
        self.same_scale = bool(getattr(self, "same_scale", kwargs.get("same_scale", False)))
        try:
            self.fps = float(getattr(self, "fps", kwargs.get("fps", 30.0)))
        except (ValueError, TypeError):
            self.fps = 30.0
        try:
            self.n_time_steps_3d = int(getattr(self, "n_time_steps_3d", kwargs.get("n_time_steps_3d", 40)))
        except (ValueError, TypeError):
            self.n_time_steps_3d = 40
        try:
            self.camera_elev = float(getattr(self, "camera_elev", kwargs.get("camera_elev", 30.0)))
        except (ValueError, TypeError):
            self.camera_elev = 30.0
        try:
            self.camera_azim = float(getattr(self, "camera_azim", kwargs.get("camera_azim", -60.0)))
        except (ValueError, TypeError):
            self.camera_azim = -60.0
        self.export_html_3d = bool(getattr(self, "export_html_3d", kwargs.get("export_html_3d", False)))
        self.auto_save_png = bool(getattr(self, "auto_save_png", kwargs.get("auto_save_png", True)))
        try:
            self.dpi = int(getattr(self, "dpi", kwargs.get("dpi", 200)))
        except (ValueError, TypeError):
            self.dpi = 200

        # Internal state
        self._is_processing = False
        self._last_exported_png = ""

        # UI Tags
        self.csv_path_tag = f"kymo_csv_path_{self.UUID}"
        self.status_text_tag = f"kymo_status_{self.UUID}"
        self.plot_title_tag = f"kymo_title_{self.UUID}"
        self.plot_mode_tag = f"kymo_mode_{self.UUID}"
        self.num_bins_tag = f"kymo_bins_{self.UUID}"
        self.static_thresh_tag = f"kymo_thresh_{self.UUID}"
        self.smooth_time_tag = f"kymo_sm_time_{self.UUID}"
        self.smooth_space_tag = f"kymo_sm_space_{self.UUID}"
        self.interpolation_tag = f"kymo_interp_{self.UUID}"
        self.plot_total_tag = f"kymo_plot_tot_{self.UUID}"
        self.plot_moving_tag = f"kymo_plot_mov_{self.UUID}"
        self.plot_static_tag = f"kymo_plot_stat_{self.UUID}"
        self.cmap_total_tag = f"kymo_cmap_tot_{self.UUID}"
        self.cmap_moving_tag = f"kymo_cmap_mov_{self.UUID}"
        self.cmap_static_tag = f"kymo_cmap_stat_{self.UUID}"
        self.same_scale_tag = f"kymo_same_scale_{self.UUID}"
        self.fps_tag = f"kymo_fps_{self.UUID}"
        self.n_time_steps_3d_tag = f"kymo_time_steps_3d_{self.UUID}"
        self.camera_elev_tag = f"kymo_elev_{self.UUID}"
        self.camera_azim_tag = f"kymo_azim_{self.UUID}"
        self.export_html_3d_tag = f"kymo_html_3d_{self.UUID}"
        self.auto_save_png_tag = f"kymo_save_png_{self.UUID}"
        self.dpi_tag = f"kymo_dpi_{self.UUID}"
        self.group_3d_params_tag = f"kymo_3d_group_{self.UUID}"

        with self.window():
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    label="CSV",
                    tag=self.csv_path_tag,
                    default_value=self.csv_path,
                    width=scale(230),
                    callback=self._on_csv_path_changed
                )
                dpg.add_button(label="\uf07c Browse", callback=self._on_browse_clicked)
                dpg.add_button(label="\uf04b Generate", callback=self._on_generate_clicked)

            dpg.add_text("Ready", tag=self.status_text_tag, color=(160, 200, 255))
            dpg.add_separator()

            with dpg.collapsing_header(label="Plot Settings", default_open=True):
                dpg.add_input_text(
                    label="Title",
                    tag=self.plot_title_tag,
                    default_value=self.plot_title,
                    hint="Optional plot title (leave empty for none)",
                    width=scale(230),
                    callback=self._update_params
                )
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        label="Mode",
                        tag=self.plot_mode_tag,
                        items=["2D Continuous Heatmap", "3D Surface Elevation"],
                        default_value=self.plot_mode,
                        width=scale(160),
                        callback=self._on_plot_mode_changed
                    )
                    dpg.add_combo(
                        label="Interp",
                        tag=self.interpolation_tag,
                        items=["bicubic", "gaussian", "bilinear", "nearest", "spline16"],
                        default_value=self.interpolation,
                        width=scale(85),
                        callback=self._update_params
                    )
                    dpg.add_input_int(
                        label="DPI",
                        tag=self.dpi_tag,
                        default_value=self.dpi,
                        min_value=72, max_value=600,
                        width=scale(55), step=0,
                        callback=self._update_params
                    )

                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="Bins",
                        tag=self.num_bins_tag,
                        default_value=self.num_bins,
                        min_value=5, max_value=200,
                        width=scale(60), step=0,
                        callback=self._update_params
                    )
                    dpg.add_input_float(
                        label="Thresh (um/s)",
                        tag=self.static_thresh_tag,
                        default_value=self.static_thresh,
                        min_value=0.0, max_value=100.0, format="%.2f",
                        width=scale(65), step=0,
                        callback=self._update_params
                    )
                    dpg.add_input_float(
                        label="FPS",
                        tag=self.fps_tag,
                        default_value=self.fps,
                        min_value=0.1, max_value=240.0, format="%.1f",
                        width=scale(75), step=0,
                        callback=self._update_params
                    )

                with dpg.group(horizontal=True):
                    dpg.add_input_float(
                        label="Sigma Time",
                        tag=self.smooth_time_tag,
                        default_value=self.smooth_time,
                        min_value=0.0, max_value=20.0, format="%.1f",
                        width=scale(55), step=0,
                        callback=self._update_params
                    )
                    dpg.add_input_float(
                        label="Sigma Space",
                        tag=self.smooth_space_tag,
                        default_value=self.smooth_space,
                        min_value=0.0, max_value=20.0, format="%.1f",
                        width=scale(55), step=0,
                        callback=self._update_params
                    )

                with dpg.group(tag=self.group_3d_params_tag, show=(self.plot_mode == "3D Surface Elevation")):
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(
                            label="3D Steps",
                            tag=self.n_time_steps_3d_tag,
                            default_value=self.n_time_steps_3d,
                            min_value=5, max_value=200,
                            width=scale(60), step=0,
                            callback=self._update_params
                        )
                        dpg.add_input_float(
                            label="Elev (deg)",
                            tag=self.camera_elev_tag,
                            default_value=self.camera_elev,
                            min_value=-90.0, max_value=90.0, format="%.0f",
                            width=scale(50), step=0,
                            callback=self._update_params
                        )
                        dpg.add_input_float(
                            label="Azim (deg)",
                            tag=self.camera_azim_tag,
                            default_value=self.camera_azim,
                            min_value=-180.0, max_value=180.0, format="%.0f",
                            width=scale(50), step=0,
                            callback=self._update_params
                        )
                        dpg.add_checkbox(
                            label="3D HTML (Plotly)",
                            tag=self.export_html_3d_tag,
                            default_value=self.export_html_3d,
                            callback=self._update_params
                        )

                dpg.add_separator()
                dpg.add_text("Panels Selection & Colormaps:")
                
                # Total Panel & Colormap
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Total Population",
                        tag=self.plot_total_tag,
                        default_value=self.plot_total,
                        callback=self._on_panel_toggled
                    )
                    dpg.add_combo(
                        items=AVAILABLE_CMAPS,
                        default_value=self.cmap_total,
                        tag=self.cmap_total_tag,
                        width=scale(90),
                        show=self.plot_total,
                        callback=self._update_params
                    )

                # Moving Panel & Colormap
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Moving Only",
                        tag=self.plot_moving_tag,
                        default_value=self.plot_moving,
                        callback=self._on_panel_toggled
                    )
                    dpg.add_combo(
                        items=AVAILABLE_CMAPS,
                        default_value=self.cmap_moving,
                        tag=self.cmap_moving_tag,
                        width=scale(90),
                        show=self.plot_moving,
                        callback=self._update_params
                    )

                # Static Panel & Colormap
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Static Only",
                        tag=self.plot_static_tag,
                        default_value=self.plot_static,
                        callback=self._on_panel_toggled
                    )
                    dpg.add_combo(
                        items=AVAILABLE_CMAPS,
                        default_value=self.cmap_static,
                        tag=self.cmap_static_tag,
                        width=scale(90),
                        show=self.plot_static,
                        callback=self._update_params
                    )

                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Same scale (shared color & value range)",
                        tag=self.same_scale_tag,
                        default_value=self.same_scale,
                        callback=self._update_params
                    )

                dpg.add_checkbox(
                    label="Auto Save PNG alongside CSV",
                    tag=self.auto_save_png_tag,
                    default_value=self.auto_save_png,
                    callback=self._update_params
                )

        self.update_permission()

    def update_permission(self) -> None:
        """Adjust widget permissions based on user mode."""
        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def is_ready(self) -> bool:
        """Check if module is ready to process data."""
        return not self._is_processing

    def on_close(self) -> None:
        """Cleanup on window close."""
        self._is_processing = False
        super().on_close()

    def close(self) -> None:
        """Explicit close handler."""
        self.on_close()

    def destroy(self) -> None:
        """Complete resource deallocation."""
        self.on_close()

    def _on_panel_toggled(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        self._update_params()
        if dpg.does_item_exist(self.cmap_total_tag):
            dpg.configure_item(self.cmap_total_tag, show=self.plot_total)
        if dpg.does_item_exist(self.cmap_moving_tag):
            dpg.configure_item(self.cmap_moving_tag, show=self.plot_moving)
        if dpg.does_item_exist(self.cmap_static_tag):
            dpg.configure_item(self.cmap_static_tag, show=self.plot_static)

    def _on_plot_mode_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        self.plot_mode = str(app_data)
        is_3d = (self.plot_mode == "3D Surface Elevation")
        if dpg.does_item_exist(self.group_3d_params_tag):
            dpg.configure_item(self.group_3d_params_tag, show=is_3d)
        self._update_params()

    def _update_params(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        if dpg.does_item_exist(self.plot_title_tag):
            self.plot_title = str(dpg.get_value(self.plot_title_tag))
        if dpg.does_item_exist(self.plot_mode_tag):
            self.plot_mode = str(dpg.get_value(self.plot_mode_tag))
        if dpg.does_item_exist(self.num_bins_tag):
            self.num_bins = int(dpg.get_value(self.num_bins_tag))
        if dpg.does_item_exist(self.static_thresh_tag):
            self.static_thresh = float(dpg.get_value(self.static_thresh_tag))
        if dpg.does_item_exist(self.fps_tag):
            self.fps = float(dpg.get_value(self.fps_tag))
        if dpg.does_item_exist(self.smooth_time_tag):
            self.smooth_time = float(dpg.get_value(self.smooth_time_tag))
        if dpg.does_item_exist(self.smooth_space_tag):
            self.smooth_space = float(dpg.get_value(self.smooth_space_tag))
        if dpg.does_item_exist(self.interpolation_tag):
            self.interpolation = str(dpg.get_value(self.interpolation_tag))
        if dpg.does_item_exist(self.plot_total_tag):
            self.plot_total = bool(dpg.get_value(self.plot_total_tag))
        if dpg.does_item_exist(self.plot_moving_tag):
            self.plot_moving = bool(dpg.get_value(self.plot_moving_tag))
        if dpg.does_item_exist(self.plot_static_tag):
            self.plot_static = bool(dpg.get_value(self.plot_static_tag))
        if dpg.does_item_exist(self.cmap_total_tag):
            self.cmap_total = str(dpg.get_value(self.cmap_total_tag))
        if dpg.does_item_exist(self.cmap_moving_tag):
            self.cmap_moving = str(dpg.get_value(self.cmap_moving_tag))
        if dpg.does_item_exist(self.cmap_static_tag):
            self.cmap_static = str(dpg.get_value(self.cmap_static_tag))
        if dpg.does_item_exist(self.same_scale_tag):
            self.same_scale = bool(dpg.get_value(self.same_scale_tag))
        if dpg.does_item_exist(self.n_time_steps_3d_tag):
            self.n_time_steps_3d = int(dpg.get_value(self.n_time_steps_3d_tag))
        if dpg.does_item_exist(self.camera_elev_tag):
            self.camera_elev = float(dpg.get_value(self.camera_elev_tag))
        if dpg.does_item_exist(self.camera_azim_tag):
            self.camera_azim = float(dpg.get_value(self.camera_azim_tag))
        if dpg.does_item_exist(self.export_html_3d_tag):
            self.export_html_3d = bool(dpg.get_value(self.export_html_3d_tag))
        if dpg.does_item_exist(self.auto_save_png_tag):
            self.auto_save_png = bool(dpg.get_value(self.auto_save_png_tag))
        if dpg.does_item_exist(self.dpi_tag):
            self.dpi = int(dpg.get_value(self.dpi_tag))

    def _on_csv_path_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        self.csv_path = str(app_data or "")

    def _on_browse_clicked(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        default_dir = os.path.dirname(self.csv_path) if self.csv_path and os.path.exists(self.csv_path) else ""
        file_explorer.select_file(
            callback=self._on_csv_selected,
            default_path=default_dir,
            extensions=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

    def _on_csv_selected(self, path: str, ext: str = "", *args: Any, **kwargs: Any) -> None:
        if path:
            self.csv_path = path
            if dpg.does_item_exist(self.csv_path_tag):
                dpg.set_value(self.csv_path_tag, path)
            self._start_processing(path)

    def _on_generate_clicked(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        path = dpg.get_value(self.csv_path_tag) if dpg.does_item_exist(self.csv_path_tag) else self.csv_path
        if path:
            self._start_processing(path)

    def _resolve_csv_path(self, raw_path: str) -> str:
        """Resolves raw input (video, .processed, or csv) to target CSV file."""
        if not raw_path:
            return ""
        if raw_path.lower().endswith(".csv") and os.path.exists(raw_path):
            return raw_path

        base_no_ext = os.path.splitext(raw_path)[0]
        candidate_det = base_no_ext + "_radial_detections.csv"
        if os.path.exists(candidate_det):
            return candidate_det

        candidate_rad = base_no_ext + "_radial.csv"
        if os.path.exists(candidate_rad):
            return candidate_rad

        candidate_csv = base_no_ext + ".csv"
        if os.path.exists(candidate_csv):
            return candidate_csv

        # Check directory
        if os.path.isdir(raw_path):
            for f in os.listdir(raw_path):
                if f.endswith("_radial_detections.csv"):
                    return os.path.join(raw_path, f)
            for f in os.listdir(raw_path):
                if f.endswith(".csv"):
                    return os.path.join(raw_path, f)

        return raw_path if os.path.exists(raw_path) else ""

    def _start_processing(self, raw_path: str):
        if self._is_processing:
            return
        target_csv = self._resolve_csv_path(raw_path)
        if not target_csv or not os.path.exists(target_csv):
            if dpg.does_item_exist(self.status_text_tag):
                dpg.set_value(self.status_text_tag, f"Error: CSV not found for {os.path.basename(raw_path)}")
                dpg.configure_item(self.status_text_tag, color=(255, 100, 100))
            return

        self._is_processing = True
        if dpg.does_item_exist(self.status_text_tag):
            dpg.set_value(self.status_text_tag, f"Generating {self.plot_mode} for {os.path.basename(target_csv)}...")
            dpg.configure_item(self.status_text_tag, color=(255, 220, 100))

        threading.Thread(target=self._process_worker, args=(target_csv,), daemon=True).start()

    def _process_worker(self, csv_file: str):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            custom_title = self.plot_title.strip()
            num_bins = self.num_bins
            static_thresh = self.static_thresh
            fps_val = max(0.1, self.fps)
            smooth_time = self.smooth_time
            smooth_space = self.smooth_space
            interp = self.interpolation
            plot_tot = self.plot_total
            plot_mov = self.plot_moving
            plot_stat = self.plot_static
            cmap_tot = self.cmap_total or "magma"
            cmap_mov = self.cmap_moving or "viridis"
            cmap_stat = self.cmap_static or "cividis"
            same_scale = self.same_scale
            save_png = self.auto_save_png
            dpi = max(72, self.dpi)
            is_3d = (self.plot_mode == "3D Surface Elevation")

            # Ensure at least 1 panel is active
            if not (plot_tot or plot_mov or plot_stat):
                plot_tot = True

            data = parse_detections_data(
                csv_file, num_bins=num_bins, static_thresh=static_thresh,
                smooth_time=smooth_time, smooth_space=smooth_space,
                n_time_steps=self.n_time_steps_3d, fps=fps_val
            )

            unit_name = data["unit_name"]
            max_dist = data["max_dist"]

            panels = []
            if is_3d:
                x_centers = data["dist_centers_3d"]
                y_times = data["sampled_times_3d"]
                X_grid, Y_grid = np.meshgrid(x_centers, y_times)
                if plot_tot:
                    panels.append(("Total Population", data["tot_mat_3d"], cmap_tot))
                if plot_mov:
                    panels.append((f"Moving (Speed >= {static_thresh})", data["mov_mat_3d"], cmap_mov))
                if plot_stat:
                    panels.append((f"Static (Speed < {static_thresh})", data["stat_mat_3d"], cmap_stat))
            else:
                times_2d = data["times_2d"]
                if plot_tot:
                    panels.append(("Total Population", data["tot_mat_2d"], cmap_tot))
                if plot_mov:
                    panels.append((f"Moving (Speed >= {static_thresh})", data["mov_mat_2d"], cmap_mov))
                if plot_stat:
                    panels.append((f"Static (Speed < {static_thresh})", data["stat_mat_2d"], cmap_stat))

            n_panels = len(panels)
            fig_w = 6.2 * n_panels
            fig_h = 7.0

            # Calculate shared scale limits if same_scale is enabled
            if same_scale and len(panels) > 0:
                global_vmax = max(float(np.nanmax(mat)) for _, mat, _ in panels)
                if global_vmax <= 0:
                    global_vmax = 1.0
                global_vmin = 0.0
            else:
                global_vmin = None
                global_vmax = None

            if is_3d:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(fig_w, fig_h))
                for idx, (title, matrix, cmap) in enumerate(panels):
                    ax = fig.add_subplot(1, n_panels, idx + 1, projection='3d')
                    kwargs_surf = {}
                    if same_scale and global_vmax is not None:
                        kwargs_surf["vmin"] = global_vmin
                        kwargs_surf["vmax"] = global_vmax
                        ax.set_zlim(0, global_vmax)

                    surf = ax.plot_surface(
                        X_grid, Y_grid, matrix,
                        cmap=cmap, rstride=1, cstride=1,
                        linewidth=0.1, antialiased=True, alpha=0.92,
                        **kwargs_surf
                    )
                    ax.view_init(elev=self.camera_elev, azim=self.camera_azim)
                    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
                    ax.set_xlabel(f"Distance ({unit_name})", fontsize=9, fontweight='bold', labelpad=8)
                    ax.set_ylabel("Time (s)", fontsize=9, fontweight='bold', labelpad=8)
                    ax.set_zlabel("Worms / Frame", fontsize=9, fontweight='bold', labelpad=8)
                    cbar = fig.colorbar(surf, ax=ax, pad=0.1, shrink=0.6)
                    cbar.set_label("Density", fontsize=8)

                if custom_title:
                    fig.suptitle(custom_title, fontsize=12, fontweight='bold')
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                else:
                    plt.tight_layout()
                out_png = os.path.splitext(csv_file)[0] + "_3d_kymograph.png"
            else:
                fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True, squeeze=False)
                axes_flat = axes[0]
                extent = [0, max_dist, times_2d[0], times_2d[-1]]

                for idx, (title, matrix, cmap) in enumerate(panels):
                    ax = axes_flat[idx]
                    kwargs_im = {}
                    if same_scale and global_vmax is not None:
                        kwargs_im["vmin"] = global_vmin
                        kwargs_im["vmax"] = global_vmax

                    im = ax.imshow(matrix, aspect='auto', cmap=cmap, extent=extent, origin='lower',
                                   interpolation=interp, **kwargs_im)
                    ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.set_xlabel(f"Distance from Pellet ({unit_name})", fontsize=10, fontweight='bold')
                    if idx == 0:
                        ax.set_ylabel("Time (s)", fontsize=10, fontweight='bold')
                    cbar = fig.colorbar(im, ax=ax, pad=0.02)
                    cbar.set_label("Density", fontsize=9)

                if custom_title:
                    fig.suptitle(custom_title, fontsize=12, fontweight='bold')
                    plt.tight_layout(rect=[0, 0, 1, 0.94])
                else:
                    plt.tight_layout()
                out_png = os.path.splitext(csv_file)[0] + "_continuous_kymograph.png"

            if save_png:
                fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
                logger.info(f"KymographPlotter: Saved PNG to {out_png}")
                self._last_exported_png = out_png

            # Optional 3D Interactive HTML export via Plotly
            if is_3d and self.export_html_3d:
                try:
                    import plotly.graph_objects as go
                    html_path = os.path.splitext(csv_file)[0] + "_3d_kymograph.html"
                    Z_ext = panels[0][1]  # primary selected panel
                    n_y, n_x = Z_ext.shape
                    x_ext = np.concatenate([[x_centers[0]], x_centers, [x_centers[-1]]])
                    y_ext = np.concatenate([[y_times[0]], y_times, [y_times[-1]]])
                    Z_solid = np.zeros((n_y + 2, n_x + 2))
                    Z_solid[1:-1, 1:-1] = Z_ext

                    surf_trace = go.Surface(
                        z=Z_solid, x=x_ext, y=y_ext,
                        colorscale="Viridis",
                        name="Chemotaxis 3D Volume",
                        colorbar=dict(title=dict(text="Worms/Frame", font=dict(size=12)))
                    )
                    fig_plotly = go.Figure(data=[surf_trace])
                    fig_plotly.update_layout(
                        title=f"3D Chemotaxis Population Surface - {os.path.basename(csv_file)}",
                        scene=dict(
                            xaxis_title=f"Distance ({unit_name})",
                            yaxis_title="Time (s)",
                            zaxis_title="Worms/Frame",
                            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.4))
                        )
                    )
                    fig_plotly.write_html(html_path)
                    logger.info(f"KymographPlotter: Exported interactive 3D HTML to {html_path}")
                except Exception as exc:
                    logger.debug(f"Plotly 3D HTML export failed: {exc}")

            # Render to standard OpenCV BGR frame
            import cv2
            fig.canvas.draw()
            rgba_buf = np.asarray(fig.canvas.buffer_rgba())
            rgb_frame = rgba_buf[:, :, :3]
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            plt.close(fig)

            # 1. Emit Frame to downstream modules (e.g. Image Viewer, Video Viewer)
            frame_modules = self.connections.get("Frame", [])
            for module in frame_modules:
                try:
                    module.input_cb(data=(bgr_frame, os.path.basename(csv_file)), data_type=IOTypes.FRAME)
                except Exception as exc:
                    logger.debug(f"KymographPlotter Frame emission failed: {exc}")

            # 2. Emit Speed Histogram (IOTypes.SAMPLE) for Histogram_win
            hist_modules = self.connections.get("Speed Histogram", [])
            raw_speeds = data.get("raw_speeds", np.array([]))
            if hist_modules and len(raw_speeds) > 0:
                hist_sample = {
                    "uuid": f"kymo_hist_{self.UUID}_{os.path.splitext(os.path.basename(csv_file))[0]}",
                    "name": f"Speed - {os.path.basename(csv_file)}",
                    "y": raw_speeds,
                    "action": "select"
                }
                for module in hist_modules:
                    try:
                        module.input_cb(sample=hist_sample, data_type=IOTypes.SAMPLE)
                    except Exception as exc:
                        logger.debug(f"KymographPlotter Speed Histogram emission failed: {exc}")

            # 3. Emit Speed vs Time (IOTypes.SAMPLE)
            time_modules = self.connections.get("Speed vs Time", [])
            time_x = data.get("time_series_x", np.array([]))
            time_y = data.get("time_series_y", np.array([]))
            if time_modules and len(time_x) > 0:
                time_sample = {
                    "uuid": f"kymo_speed_time_{self.UUID}_{os.path.splitext(os.path.basename(csv_file))[0]}",
                    "name": f"Mean Speed/Time - {os.path.basename(csv_file)}",
                    "x": time_x,
                    "y": time_y,
                    "action": "select"
                }
                for module in time_modules:
                    try:
                        module.input_cb(sample=time_sample, data_type=IOTypes.SAMPLE)
                    except Exception as exc:
                        logger.debug(f"KymographPlotter Speed vs Time emission failed: {exc}")

            # 4. Emit Speed Boxplot Data (IOTypes.SAMPLE) for Boxplot_win
            box_modules = self.connections.get("Speed Boxplot Data", [])
            box_times = data.get("box_times", np.array([]))
            box_speeds = data.get("box_speeds", np.array([]))
            if box_modules and len(box_times) > 0 and len(box_speeds) > 0:
                box_sample = {
                    "uuid": f"kymo_box_{self.UUID}_{os.path.splitext(os.path.basename(csv_file))[0]}",
                    "name": f"Speed Dist - {os.path.basename(csv_file)}",
                    "x": box_times,
                    "y": box_speeds,
                    "xlabel": "Time (s)",
                    "ylabel": f"Speed ({unit_name}/s)",
                    "action": "select"
                }
                for module in box_modules:
                    try:
                        module.input_cb(sample=box_sample, data_type=IOTypes.SAMPLE)
                    except Exception as exc:
                        logger.debug(f"KymographPlotter Speed Boxplot emission failed: {exc}")

            if dpg.does_item_exist(self.status_text_tag):
                dpg.set_value(self.status_text_tag, f"Done ({self.plot_mode}, {n_panels} panel{'s' if n_panels > 1 else ''}) - {os.path.basename(csv_file)}")
                dpg.configure_item(self.status_text_tag, color=(100, 255, 120))

            # Mark processing finished before emitting completion trigger
            self._is_processing = False

            # 4. Emit Trigger to downstream modules (e.g. File Scanner Emitter queue coordinator)
            trigger_modules = self.connections.get("Trigger", [])
            for module in trigger_modules:
                try:
                    trigger_data = out_png if save_png and os.path.exists(out_png) else csv_file
                    module.input_cb(data=trigger_data, data_type=IOTypes.TRIGGER)
                except Exception as exc:
                    logger.debug(f"KymographPlotter Trigger emission failed: {exc}")

        except Exception as e:
            logger.error(f"KymographPlotter error: {e}")
            if dpg.does_item_exist(self.status_text_tag):
                dpg.set_value(self.status_text_tag, f"Error: {e}")
                dpg.configure_item(self.status_text_tag, color=(255, 80, 80))
        finally:
            self._is_processing = False

    def input_cb(self, *args: Any, **kwargs: Any) -> bool:
        data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
        data_type = kwargs.get("data_type") if "data_type" in kwargs else None

        if data_type == IOTypes.FILE_PATH:
            if isinstance(data, tuple) and len(data) == 2:
                path, _ = data
            else:
                path = data or ""
            if path:
                self.csv_path = str(path)
                if dpg.does_item_exist(self.csv_path_tag):
                    dpg.set_value(self.csv_path_tag, self.csv_path)
                self._start_processing(self.csv_path)
            return True

        if data_type == IOTypes.TRIGGER:
            if isinstance(data, str) and (data.lower().endswith(".csv") or data.lower().endswith(".mp4") or data.lower().endswith(".processed")):
                self.csv_path = data
                if dpg.does_item_exist(self.csv_path_tag):
                    dpg.set_value(self.csv_path_tag, self.csv_path)
                self._start_processing(self.csv_path)
            else:
                # Trigger without explicit path -> re-run on current csv_path
                if self.csv_path:
                    self._start_processing(self.csv_path)
            return True

        return False


KymographPlotterWin = Kymograph_plotter_win
EXPORTED_CLASS = Kymograph_plotter_win
EXPORTED_NAME = "Kymograph Plotter"
