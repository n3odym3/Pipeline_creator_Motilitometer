from __future__ import annotations

import collections
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import dearpygui.dearpygui as dpg
from loguru import logger
import numpy as np

from config.display_scaling import display_scaling
from core.input_output_types import IOTypes
from core.tools.clipboard_injector import clipboardinjector
from core.window_base import WindowBase
from modules.plot.polar_spectrogram import PolarSpectrogramProcessor


class PolarSpectrogramWin(WindowBase):
    """
    Polar Spectrogram window module for visualizing temporal distributions of vector angles
    and orientation fields over time.

    Maps angular distributions across successive video frames or time steps, revealing
    behavioral trends, periodic directional oscillations, and orientation transitions.
    """

    label: str = "Polar Spectrogram"
    win_width: int = display_scaling.scale(720)
    win_height: int = display_scaling.scale(700)

    accepted_input_types: List[Any] = [
        IOTypes.VECTORS,
        IOTypes.SAMPLE,
        IOTypes.DATALIST,
        IOTypes.CMD_DICT,
        IOTypes.TRIGGER,
    ]
    outputs: Dict[str, Any] = {
        "Frame": IOTypes.FRAME,
        "CMD": IOTypes.CMD_DICT,
    }

    persistent_fields: List[str] = [
        "label",
        "history_size",
        "fps",
        "auto_delta_t",
        "num_bins",
        "orientation",
        "weight_by_magnitude",
        "temporal_window",
        "smooth_time",
        "smooth_angle",
        "interpolation",
        "mean_curve_smooth",
        "show_mean_curve",
        "invert_y",
        "color_mode",
        "colormap",
        "normalization",
        "color_scale_mode",
        "color_vmin",
        "color_vmax",
        "theme_mode",
        "plot_title",
        "rolling_mode",
        "is_paused",
        "canvas_width",
        "canvas_height",
        "dpi",
    ]

    history_size: int = 300
    fps: float = 30.0
    auto_delta_t: bool = False
    num_bins: int = 72
    orientation: str = "Horizontal (Time ->)"
    weight_by_magnitude: bool = True
    temporal_window: int = 1
    smooth_time: float = 1.0
    smooth_angle: float = 1.0
    interpolation: str = "bicubic"
    mean_curve_smooth: int = 3
    show_mean_curve: bool = True
    invert_y: bool = True
    color_mode: str = "Colormap"
    colormap: str = "turbo"
    normalization: str = "Frame Peak (Max = 100%)"
    color_scale_mode: str = "Auto"
    color_vmin: float = 0.0
    color_vmax: float = 50.0
    theme_mode: str = "Auto"
    plot_title: str = ""
    rolling_mode: bool = False
    is_paused: bool = False
    canvas_width: int = 1100
    canvas_height: int = 650
    dpi: int = 100

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._is_closing: bool = False

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_thread_running: bool = False
        self.processor: Optional[PolarSpectrogramProcessor] = None

        self.history_size = max(10, int(self.history_size if self.history_size is not None else 300))
        self.fps = max(0.1, float(self.fps if self.fps is not None else 30.0))
        self.delta_t: float = 1.0 / self.fps
        self.auto_delta_t = bool(self.auto_delta_t if self.auto_delta_t is not None else False)
        self.num_bins = max(8, min(360, int(self.num_bins if self.num_bins is not None else 72)))
        self.orientation = str(self.orientation or "Horizontal (Time ->)")
        self.weight_by_magnitude = bool(self.weight_by_magnitude if self.weight_by_magnitude is not None else True)
        self.temporal_window = max(1, int(self.temporal_window if self.temporal_window is not None else 1))
        self.smooth_time = max(0.0, float(self.smooth_time if self.smooth_time is not None else 1.0))
        self.smooth_angle = max(0.0, float(self.smooth_angle if self.smooth_angle is not None else 1.0))
        self.interpolation = str(self.interpolation or "bicubic")
        self.mean_curve_smooth = max(1, int(self.mean_curve_smooth if self.mean_curve_smooth is not None else 3))
        self.show_mean_curve = bool(self.show_mean_curve if self.show_mean_curve is not None else True)
        self.invert_y = bool(self.invert_y if self.invert_y is not None else True)
        self.color_mode = str(self.color_mode or "Colormap")
        self.colormap = str(self.colormap or "turbo")
        self.normalization = str(self.normalization or "Frame Peak (Max = 100%)")
        self.color_scale_mode = str(self.color_scale_mode or "Auto")
        self.color_vmin = float(self.color_vmin if self.color_vmin is not None else 0.0)
        self.color_vmax = float(self.color_vmax if self.color_vmax is not None else 50.0)
        self.theme_mode = str(self.theme_mode or "Auto")
        self.plot_title = str(self.plot_title or "")
        self.rolling_mode = bool(self.rolling_mode if self.rolling_mode is not None else False)
        self.is_paused = bool(self.is_paused if self.is_paused is not None else False)
        self.canvas_width = max(200, int(self.canvas_width if self.canvas_width is not None else 1100))
        self.canvas_height = max(200, int(self.canvas_height if self.canvas_height is not None else 650))
        self.dpi = max(50, int(self.dpi if self.dpi is not None else 100))

        max_l = self.history_size if self.rolling_mode else None
        self.raw_history: Deque[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, float, int]] = collections.deque(
            maxlen=max_l
        )
        self._matrix_rows: Deque[np.ndarray] = collections.deque(maxlen=max_l)
        self._mean_angles: Deque[float] = collections.deque(maxlen=max_l)
        self._directionality: Deque[float] = collections.deque(maxlen=max_l)
        self._mean_speeds: Deque[float] = collections.deque(maxlen=max_l)
        self._total_vectors: Deque[int] = collections.deque(maxlen=max_l)
        self._timestamps: Deque[float] = collections.deque(maxlen=max_l)
        self._frame_indices: Deque[int] = collections.deque(maxlen=max_l)

        self._buffer_lock = threading.RLock()
        self._frame_counter: int = 0
        self.current_t: float = 0.0
        self._last_arrival_time: Optional[float] = None
        self._last_render_submit_time: float = 0.0
        self._pending_render: bool = False
        self._worker_busy: bool = False
        self._batch_loaded: bool = False
        self._last_source: str = ""
        self._last_frame_idx: Optional[int] = None

        self.texture_registry_tag: str = f"polar_spectro_registry_{self.UUID}"
        self.texture_tag: str = f"polar_spectro_texture_{self.UUID}"
        self._tex_buffer: np.ndarray = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.float32)
        self.last_bgr_frame: Optional[np.ndarray] = None
        self.last_stats: Dict[str, Any] = {
            "total_frames": 0,
            "current_frame": 0,
            "elapsed_time": 0.0,
            "latest_mean_angle_deg": 0.0,
            "latest_directionality_r": 0.0,
            "latest_mean_speed": 0.0,
            "latest_total_vectors": 0,
        }

        self.options_header_tag: str = f"spectro_options_{self.UUID}"
        self.fps_input_tag: str = f"spectro_fps_{self.UUID}"
        self.auto_dt_check_tag: str = f"spectro_auto_dt_{self.UUID}"
        self.bins_slider_tag: str = f"spectro_bins_{self.UUID}"
        self.history_input_tag: str = f"spectro_hist_{self.UUID}"
        self.rolling_check_tag: str = f"spectro_rolling_{self.UUID}"
        self.pause_check_tag: str = f"spectro_pause_{self.UUID}"
        self.weight_check_tag: str = f"spectro_weight_{self.UUID}"
        self.invert_y_check_tag: str = f"spectro_inverty_{self.UUID}"
        self.mean_curve_check_tag: str = f"spectro_mean_curve_{self.UUID}"
        self.orientation_combo_tag: str = f"spectro_orient_{self.UUID}"
        self.cmap_combo_tag: str = f"spectro_cmap_{self.UUID}"
        self.color_mode_combo_tag: str = f"spectro_colormode_{self.UUID}"
        self.norm_combo_tag: str = f"spectro_norm_{self.UUID}"
        self.color_scale_mode_tag: str = f"spectro_cscale_mode_{self.UUID}"
        self.color_vmin_tag: str = f"spectro_cvmin_{self.UUID}"
        self.color_vmax_tag: str = f"spectro_cvmax_{self.UUID}"
        self.color_scale_group_tag: str = f"spectro_cscale_grp_{self.UUID}"
        self.theme_combo_tag: str = f"spectro_theme_{self.UUID}"
        self.title_input_tag: str = f"spectro_title_{self.UUID}"
        self.stats_text_tag: str = f"spectro_stats_{self.UUID}"
        self.temp_window_input_tag: str = f"spectro_temp_win_{self.UUID}"
        self.smooth_time_tag: str = f"spectro_sm_time_{self.UUID}"
        self.smooth_angle_tag: str = f"spectro_sm_angle_{self.UUID}"
        self.interpolation_combo_tag: str = f"spectro_interp_{self.UUID}"
        self.curve_smooth_tag: str = f"spectro_curve_smooth_{self.UUID}"

        with dpg.texture_registry(tag=self.texture_registry_tag, show=False):
            dpg.add_raw_texture(
                width=self.canvas_width,
                height=self.canvas_height,
                default_value=self._tex_buffer.ravel(),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb,
            )

        self._build_ui()
        self.update_permission()

        self.processor = PolarSpectrogramProcessor(
            params=self._get_processor_params(),
            buffer_size=1,
            drop_policy="drop_oldest",
            daemon=True,
        )
        self.processor.start()

        self._monitor_thread_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self.processor.submit([])

    def _is_light_theme(self) -> bool:
        """Determine whether the light theme is active."""
        if self.theme_mode == "Light":
            return True
        if self.theme_mode == "Dark":
            return False
        try:
            from config.theme_manager import theme_manager
            return bool(theme_manager.active_palette.get("is_light", False))
        except Exception:
            return False

    def _get_stats_color(self) -> Tuple[int, int, int]:
        """Return theme-dependent RGB color for the live statistics text."""
        return (30, 80, 150) if self._is_light_theme() else (160, 220, 255)

    def _get_processor_params(self) -> Dict[str, Any]:
        """Return parameter dictionary for PolarSpectrogramProcessor."""
        return {
            "num_bins": self.num_bins,
            "weight_by_magnitude": self.weight_by_magnitude,
            "invert_y": self.invert_y,
            "orientation": self.orientation,
            "color_mode": self.color_mode,
            "colormap": self.colormap,
            "normalization": self.normalization,
            "color_scale_mode": self.color_scale_mode,
            "color_vmin": self.color_vmin,
            "color_vmax": self.color_vmax,
            "temporal_window": self.temporal_window,
            "smooth_time": self.smooth_time,
            "smooth_angle": self.smooth_angle,
            "interpolation": self.interpolation,
            "mean_curve_smooth": self.mean_curve_smooth,
            "show_mean_curve": self.show_mean_curve,
            "plot_title": self.plot_title,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "dpi": self.dpi,
            "is_light": self._is_light_theme(),
            "fps": self.fps,
            "delta_t": 1.0 / max(0.1, self.fps),
        }

    def _update_processor_params(self) -> None:
        """Send updated parameters to background processor."""
        if self.processor is not None:
            self.processor.update_params(**self._get_processor_params())

    def update_permission(self) -> None:
        """Adjust module permissions based on app_state.mode."""
        from core.app_state import app_state
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=(app_state.mode == "user"))

    def _build_ui(self) -> None:
        scale = display_scaling.scale
        with self.window():
            with dpg.collapsing_header(label="Options & Export", default_open=False, tag=self.options_header_tag):
                with dpg.group(horizontal=True):
                    btn_copy = dpg.add_button(label="Copy Image", callback=self.copy_to_clipboard)
                    with dpg.tooltip(btn_copy):
                        dpg.add_text("Copy the current spectrogram image to the system clipboard.")

                    btn_save_img = dpg.add_button(label="Save PNG", callback=self.save_image_to_file)
                    with dpg.tooltip(btn_save_img):
                        dpg.add_text("Save spectrogram as high-resolution PNG image file.")

                    btn_copy_tsv = dpg.add_button(label="Copy TSV", callback=self.export_tsv_to_clipboard)
                    with dpg.tooltip(btn_copy_tsv):
                        dpg.add_text("Export full 2D time-angle spectrogram matrix as TSV to clipboard.")

                    btn_save_csv = dpg.add_button(label="Save CSV/TSV", callback=self.save_csv_to_file)
                    with dpg.tooltip(btn_save_csv):
                        dpg.add_text("Save full spectrogram matrix and circular statistics to a CSV/TSV file.")

                    btn_save_npz = dpg.add_button(label="Save .NPZ", callback=self.save_npz_to_file)
                    with dpg.tooltip(btn_save_npz):
                        dpg.add_text("Save raw matrix, timestamps, angles, and statistics as NumPy (.npz) file.")

                    dpg.add_checkbox(
                        label="Pause",
                        default_value=self.is_paused,
                        callback=self._on_pause_toggled,
                        tag=self.pause_check_tag,
                    )
                    with dpg.tooltip(self.pause_check_tag):
                        dpg.add_text("Freeze live spectrogram accumulation to inspect or export data.")

                    btn_clear = dpg.add_button(label="Reset / Clear", callback=self.clear_buffer)
                    with dpg.tooltip(btn_clear):
                        dpg.add_text("Clear all accumulated frames and reset the spectrogram.")

                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_input_float(
                        label="FPS",
                        default_value=self.fps,
                        min_value=0.1,
                        max_value=1000.0,
                        step=0.0,
                        step_fast=0.0,
                        format="%.1f",
                        width=scale(110),
                        callback=self._on_fps_changed,
                        tag=self.fps_input_tag,
                    )
                    with dpg.tooltip(self.fps_input_tag):
                        dpg.add_text(
                            "Expected video frame rate in frames per second (e.g. 30 FPS).\n"
                            "Automatically determines Delta T = 1 / FPS."
                        )

                    dpg.add_checkbox(
                        label="Auto Δt",
                        default_value=self.auto_delta_t,
                        callback=self._on_auto_delta_t_toggled,
                        tag=self.auto_dt_check_tag,
                    )
                    with dpg.tooltip(self.auto_dt_check_tag):
                        dpg.add_text(
                            "Measure real wall-clock elapsed time between incoming frames.\n"
                            "Unchecked: Uses fixed Δt = 1 / FPS."
                        )

                    dpg.add_checkbox(
                        label="Rolling",
                        default_value=self.rolling_mode,
                        callback=self._on_rolling_toggled,
                        tag=self.rolling_check_tag,
                    )
                    with dpg.tooltip(self.rolling_check_tag):
                        dpg.add_text(
                            "Unchecked (Default): Infinite accumulation (keeps all frames without limit).\n"
                            "Checked: Sliding window (keeps only the last Window frames)."
                        )

                    dpg.add_input_int(
                        label="Rolling Win",
                        default_value=self.history_size,
                        min_value=10,
                        max_value=50000,
                        step=0,
                        step_fast=0,
                        width=scale(120),
                        show=self.rolling_mode,
                        callback=self._on_history_size_changed,
                        tag=self.history_input_tag,
                    )
                    with dpg.tooltip(self.history_input_tag):
                        dpg.add_text(
                            "Number of frames displayed in the rolling window buffer (e.g. 300 frames = 10s at 30 FPS)."
                        )

                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="Avg win",
                        default_value=self.temporal_window,
                        min_value=1,
                        max_value=50,
                        step=0,
                        step_fast=0,
                        width=scale(110),
                        callback=self._on_temporal_window_changed,
                        tag=self.temp_window_input_tag,
                    )
                    with dpg.tooltip(self.temp_window_input_tag):
                        dpg.add_text(
                            "Temporal Moving Average Window (number of frames).\n"
                            "1 = Raw frames (no temporal window averaging).\n"
                            "3, 5, 10 = Averages over sliding window of X frames to reduce frame noise."
                        )

                    dpg.add_input_float(
                        label="Sigma Time",
                        default_value=self.smooth_time,
                        min_value=0.0,
                        max_value=20.0,
                        step=0.0,
                        step_fast=0.0,
                        format="%.1f",
                        width=scale(110),
                        callback=self._on_smooth_time_changed,
                        tag=self.smooth_time_tag,
                    )
                    with dpg.tooltip(self.smooth_time_tag):
                        dpg.add_text(
                            "Gaussian filter standard deviation (sigma) along the time axis.\n"
                            "0.0 = No Gaussian temporal filtering.\n"
                            "Smooths transitions over time continuously."
                        )

                    dpg.add_input_float(
                        label="Sigma Angle",
                        default_value=self.smooth_angle,
                        min_value=0.0,
                        max_value=20.0,
                        step=0.0,
                        step_fast=0.0,
                        format="%.1f",
                        width=scale(110),
                        callback=self._on_smooth_angle_changed,
                        tag=self.smooth_angle_tag,
                    )
                    with dpg.tooltip(self.smooth_angle_tag):
                        dpg.add_text(
                            "Gaussian filter standard deviation (sigma) along the angular axis.\n"
                            "Employs circular periodic wrapping (0° <-> 360°) to avoid boundary edge artifacts.\n"
                            "0.0 = No Gaussian angular filtering."
                        )

                    dpg.add_combo(
                        items=["bicubic", "bilinear", "gaussian", "nearest", "spline16"],
                        label="Interp",
                        default_value=self.interpolation,
                        width=scale(110),
                        callback=self._on_interpolation_changed,
                        tag=self.interpolation_combo_tag,
                    )
                    with dpg.tooltip(self.interpolation_combo_tag):
                        dpg.add_text(
                            "Image interpolation algorithm:\n"
                            "- bicubic / bilinear: Smooth continuous visual gradients (recommended)\n"
                            "- nearest: Sharp discrete bins\n"
                            "- gaussian: Gaussian smoothed pixels"
                        )

                    dpg.add_drag_int(
                        label="Curve Smooth",
                        default_value=self.mean_curve_smooth,
                        min_value=1,
                        max_value=61,
                        width=scale(110),
                        callback=self._on_curve_smooth_changed,
                        tag=self.curve_smooth_tag,
                    )
                    with dpg.tooltip(self.curve_smooth_tag):
                        dpg.add_text(
                            "Circular moving average window for smoothing the mean resultant angle trajectory line.\n"
                            "1 = Raw trajectory, 3+ = Smooth trend."
                        )

                with dpg.group(horizontal=True):
                    dpg.add_drag_int(
                        label="Bins",
                        default_value=self.num_bins,
                        min_value=8,
                        max_value=360,
                        width=scale(110),
                        callback=self._on_bins_changed,
                        tag=self.bins_slider_tag,
                    )
                    with dpg.tooltip(self.bins_slider_tag):
                        dpg.add_text("Number of angular sectors (e.g. 72 bins = 5° resolution, 36 bins = 10°).")

                    dpg.add_combo(
                        items=["Horizontal (Time ->)", "Vertical (Time ^)"],
                        label="Orientation",
                        default_value=self.orientation,
                        width=scale(180),
                        callback=self._on_orientation_changed,
                        tag=self.orientation_combo_tag,
                    )
                    with dpg.tooltip(self.orientation_combo_tag):
                        dpg.add_text(
                            "Horizontal: Time advances to the right, Y-axis is Angles [0°, 360°].\n"
                            "Vertical: Time rises upwards, X-axis is Angles [0°, 360°]."
                        )

                    dpg.add_combo(
                        items=[
                            "Frame Peak (Max = 100%)",
                            "Frame Density (%)",
                            "Global Max",
                            "Raw / None",
                            "Log Scale",
                        ],
                        label="Norm",
                        default_value=self.normalization,
                        width=scale(185),
                        callback=self._on_normalization_changed,
                        tag=self.norm_combo_tag,
                    )
                    with dpg.tooltip(self.norm_combo_tag):
                        dpg.add_text(
                            "Intensity Normalization:\n"
                            "- Frame Peak (100%): Scales each frame to its own peak (clear angular distribution even with few vectors).\n"
                            "- Frame Density (%): Normalizes each frame sum to 100%.\n"
                            "- Global Max: Scales relative to max count across entire history.\n"
                            "- Raw: Unnormalized counts/magnitudes.\n"
                            "- Log Scale: Logarithmic dynamic range compression."
                        )

                    dpg.add_combo(
                        items=["Auto", "Manual"],
                        label="Color Range",
                        default_value=self.color_scale_mode,
                        width=scale(110),
                        callback=self._on_color_scale_mode_changed,
                        tag=self.color_scale_mode_tag,
                    )
                    with dpg.tooltip(self.color_scale_mode_tag):
                        dpg.add_text(
                            "Color scale normalization:\n"
                            "- Auto: Dynamically stretches palette to fit the frame or matrix max.\n"
                            "- Manual: Fixes Min/Max color limits so diffuse/uniform motion stays faint and directional bursts stand out."
                        )

                    with dpg.group(
                        horizontal=True,
                        tag=self.color_scale_group_tag,
                        show=(self.color_scale_mode == "Manual"),
                    ):
                        dpg.add_input_float(
                            label="Min",
                            default_value=self.color_vmin,
                            step=0.0,
                            step_fast=0.0,
                            format="%.1f",
                            width=scale(85),
                            callback=self._on_color_vmin_changed,
                            tag=self.color_vmin_tag,
                        )
                        dpg.add_input_float(
                            label="Max",
                            default_value=self.color_vmax,
                            step=0.0,
                            step_fast=0.0,
                            format="%.1f",
                            width=scale(85),
                            callback=self._on_color_vmax_changed,
                            tag=self.color_vmax_tag,
                        )

                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        items=["Colormap", "Angle (HSV Flow)"],
                        label="Color Mode",
                        default_value=self.color_mode,
                        width=scale(150),
                        callback=self._on_color_mode_changed,
                        tag=self.color_mode_combo_tag,
                    )
                    with dpg.tooltip(self.color_mode_combo_tag):
                        dpg.add_text(
                            "Colormap: Uses chosen gradient (Turbo, Viridis, etc.)\n"
                            "Angle (HSV Flow): Angle determines Hue (East=Red, North=Green, West=Cyan, South=Blue), intensity modulates brightness."
                        )

                    dpg.add_combo(
                        items=[
                            "turbo",
                            "viridis",
                            "plasma",
                            "inferno",
                            "magma",
                            "cividis",
                            "twilight",
                            "twilight_shifted",
                            "coolwarm",
                            "hsv",
                            "Greys",
                        ],
                        label="Cmap",
                        default_value=self.colormap,
                        width=scale(120),
                        callback=self._on_colormap_changed,
                        tag=self.cmap_combo_tag,
                    )

                    dpg.add_combo(
                        items=["Auto", "Dark", "Light"],
                        label="Theme",
                        default_value=self.theme_mode,
                        width=scale(100),
                        callback=self._on_theme_changed,
                        tag=self.theme_combo_tag,
                    )

                    dpg.add_checkbox(
                        label="Weight Speed",
                        default_value=self.weight_by_magnitude,
                        callback=self._on_weight_toggled,
                        tag=self.weight_check_tag,
                    )
                    with dpg.tooltip(self.weight_check_tag):
                        dpg.add_text("Weight angular sectors by vector speed / magnitude.")

                    dpg.add_checkbox(
                        label="Invert Y",
                        default_value=self.invert_y,
                        callback=self._on_invert_y_toggled,
                        tag=self.invert_y_check_tag,
                    )
                    with dpg.tooltip(self.invert_y_check_tag):
                        dpg.add_text(
                            "Invert Y axis so video motion upwards points UP (90°) at the top of the plot,\n"
                            "and video motion downwards points DOWN (270°) at the bottom of the plot."
                        )

                    dpg.add_checkbox(
                        label="Mean Curve",
                        default_value=self.show_mean_curve,
                        callback=self._on_mean_curve_toggled,
                        tag=self.mean_curve_check_tag,
                    )
                    with dpg.tooltip(self.mean_curve_check_tag):
                        dpg.add_text("Overlay mean circular angle trajectory across time.")

                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        label="Title",
                        default_value=self.plot_title,
                        hint="Custom plot title (leave blank for automatic summary)",
                        width=scale(380),
                        callback=self._on_title_changed,
                        tag=self.title_input_tag,
                    )

                with dpg.tree_node(label="Canvas Dimensions & DPI", default_open=False):
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(
                            label="Width",
                            default_value=self.canvas_width,
                            min_value=200,
                            min_clamped=True,
                            step=0,
                            step_fast=0,
                            width=scale(100),
                            callback=lambda s, a, *args, **kwargs: setattr(self, "canvas_width", max(200, int(a))),
                        )
                        dpg.add_input_int(
                            label="Height",
                            default_value=self.canvas_height,
                            min_value=200,
                            min_clamped=True,
                            step=0,
                            step_fast=0,
                            width=scale(100),
                            callback=lambda s, a, *args, **kwargs: setattr(self, "canvas_height", max(200, int(a))),
                        )
                        dpg.add_input_int(
                            label="DPI",
                            default_value=self.dpi,
                            min_value=50,
                            max_value=600,
                            min_clamped=True,
                            max_clamped=True,
                            step=0,
                            step_fast=0,
                            width=scale(100),
                            callback=lambda s, a, *args, **kwargs: setattr(self, "dpi", max(50, int(a))),
                        )
                        dpg.add_button(label="Apply Dimensions", callback=self._apply_options)

            dpg.add_text(
                "Frames: 0/300 (0.00s) | Current Mean: 0.0° (Right) | R: 0.00 | Speed: 0.00 µm/s | Vectors: 0",
                tag=self.stats_text_tag,
                color=self._get_stats_color(),
            )

            dpg.add_separator()

            with dpg.group(tag=f"spectro_plot_wrapper_{self.UUID}"):
                self._build_plot_widget()

    def _build_plot_widget(self) -> None:
        with dpg.plot(
            height=-1,
            width=-1,
            no_menus=True,
            tag=f"spectro_plot_widget_{self.UUID}",
            parent=f"spectro_plot_wrapper_{self.UUID}",
        ):
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True)
            with dpg.plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True) as yaxis:
                dpg.add_image_series(self.texture_tag, [0, 0], [self.canvas_width, self.canvas_height])
                dpg.fit_axis_data(xaxis)
                dpg.fit_axis_data(yaxis)

    def _apply_options(self, *args: Any, **kwargs: Any) -> None:
        """Apply canvas dimension changes and recreate textures."""
        self.canvas_width = max(200, int(self.canvas_width))
        self.canvas_height = max(200, int(self.canvas_height))
        self.dpi = max(50, int(self.dpi))
        self._update_processor_params()

        tex_cfg = dpg.get_item_configuration(self.texture_tag)
        if self.canvas_width != tex_cfg.get("width") or self.canvas_height != tex_cfg.get("height"):
            self._tex_buffer = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.float32)
            if dpg.does_item_exist(self.texture_tag):
                dpg.delete_item(self.texture_tag)
            if dpg.does_item_exist(f"spectro_plot_widget_{self.UUID}"):
                dpg.delete_item(f"spectro_plot_widget_{self.UUID}")

            dpg.add_raw_texture(
                width=self.canvas_width,
                height=self.canvas_height,
                default_value=self._tex_buffer.ravel(),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb,
                parent=self.texture_registry_tag,
            )
            self._build_plot_widget()

        self._trigger_render(force=True)

    def _resize_buffers(self) -> None:
        """Resize all history deques according to rolling_mode and history_size."""
        max_l = self.history_size if self.rolling_mode else None
        with self._buffer_lock:
            self.raw_history = collections.deque(self.raw_history, maxlen=max_l)
            self._matrix_rows = collections.deque(self._matrix_rows, maxlen=max_l)
            self._mean_angles = collections.deque(self._mean_angles, maxlen=max_l)
            self._directionality = collections.deque(self._directionality, maxlen=max_l)
            self._mean_speeds = collections.deque(self._mean_speeds, maxlen=max_l)
            self._total_vectors = collections.deque(self._total_vectors, maxlen=max_l)
            self._timestamps = collections.deque(self._timestamps, maxlen=max_l)
            self._frame_indices = collections.deque(self._frame_indices, maxlen=max_l)

    def _on_fps_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.fps = max(0.1, float(app_data))
        self.delta_t = 1.0 / self.fps
        self._recompute_timestamps()
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_auto_delta_t_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.auto_delta_t = bool(app_data)
        self._last_arrival_time = None

    def _on_bins_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.num_bins = max(8, min(360, int(app_data)))
        self._rebin_all()
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_history_size_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        try:
            self.history_size = max(10, int(app_data))
            if self.rolling_mode:
                self._resize_buffers()
                self._update_processor_params()
                self._trigger_render(force=True)
        except Exception:
            pass

    def _on_rolling_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.rolling_mode = bool(app_data)
        if dpg.does_item_exist(self.history_input_tag):
            dpg.configure_item(self.history_input_tag, show=self.rolling_mode)
        self._resize_buffers()
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_pause_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.is_paused = bool(app_data)

    def _on_weight_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.weight_by_magnitude = bool(app_data)
        self._rebin_all()
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_temporal_window_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.temporal_window = max(1, int(app_data))
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_smooth_time_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.smooth_time = max(0.0, float(app_data))
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_smooth_angle_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.smooth_angle = max(0.0, float(app_data))
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_interpolation_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.interpolation = str(app_data)
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_curve_smooth_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.mean_curve_smooth = max(1, int(app_data))
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_invert_y_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.invert_y = bool(app_data)
        self._rebin_all()
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_mean_curve_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.show_mean_curve = bool(app_data)
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_color_scale_mode_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.color_scale_mode = str(app_data)
        if dpg.does_item_exist(self.color_scale_group_tag):
            dpg.configure_item(self.color_scale_group_tag, show=(self.color_scale_mode == "Manual"))
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_color_vmin_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        try:
            self.color_vmin = float(app_data)
            self._update_processor_params()
            self._trigger_render(force=True)
        except Exception:
            pass

    def _on_color_vmax_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        try:
            self.color_vmax = float(app_data)
            self._update_processor_params()
            self._trigger_render(force=True)
        except Exception:
            pass

    def _on_orientation_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.orientation = str(app_data)
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_color_mode_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.color_mode = str(app_data)
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_colormap_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.colormap = str(app_data)
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_normalization_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.normalization = str(app_data)
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_theme_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.theme_mode = str(app_data)
        if dpg.does_item_exist(self.stats_text_tag):
            dpg.configure_item(self.stats_text_tag, color=self._get_stats_color())
        self._update_processor_params()
        self._trigger_render(force=True)

    def _on_title_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.plot_title = str(app_data)
        self._update_processor_params()
        self._trigger_render(force=True)

    def _recompute_timestamps(self) -> None:
        """Recompute cumulative timestamps across buffer when delta_t is updated."""
        with self._buffer_lock:
            for i in range(len(self._timestamps)):
                self._timestamps[i] = i * self.delta_t
            for i in range(len(self.raw_history)):
                ang, cnt_w, mag, _, f_idx = self.raw_history[i]
                self.raw_history[i] = (ang, cnt_w, mag, i * self.delta_t, f_idx)

    def _rebin_all(self) -> None:
        """
        Recompute per-frame histograms and circular statistics from raw history
        when num_bins, weight_by_magnitude, or invert_y is modified.
        """
        with self._buffer_lock:
            snapshot = list(self.raw_history)
            num_bins = self.num_bins
            w_mag = self.weight_by_magnitude
            inv_y = self.invert_y
            max_l = None if self._batch_loaded else (self.history_size if self.rolling_mode else None)

        new_matrix_rows: List[np.ndarray] = []
        new_mean_angles: List[float] = []
        new_directionality: List[float] = []
        new_mean_speeds: List[float] = []
        new_total_vectors: List[int] = []
        new_timestamps: List[float] = []
        new_frame_indices: List[int] = []

        for angles_rep, cnt_w, mags_rep, t_val, f_idx in snapshot:
            row, mean_ang, r_val, spd, n_vec = PolarSpectrogramProcessor.bin_single_frame(
                angles_rep, mags_rep, num_bins, w_mag, inv_y, count_weights=cnt_w
            )
            new_matrix_rows.append(row)
            new_mean_angles.append(mean_ang)
            new_directionality.append(r_val)
            new_mean_speeds.append(spd)
            new_total_vectors.append(n_vec)
            new_timestamps.append(t_val)
            new_frame_indices.append(f_idx)

        with self._buffer_lock:
            self._matrix_rows = collections.deque(new_matrix_rows, maxlen=max_l)
            self._mean_angles = collections.deque(new_mean_angles, maxlen=max_l)
            self._directionality = collections.deque(new_directionality, maxlen=max_l)
            self._mean_speeds = collections.deque(new_mean_speeds, maxlen=max_l)
            self._total_vectors = collections.deque(new_total_vectors, maxlen=max_l)
            self._timestamps = collections.deque(new_timestamps, maxlen=max_l)
            self._frame_indices = collections.deque(new_frame_indices, maxlen=max_l)

    def _reset_buffers_locked(self) -> None:
        """Reset internal vector and matrix buffers (caller must hold or acquire _buffer_lock)."""
        max_l = self.history_size if self.rolling_mode else None
        with self._buffer_lock:
            self.raw_history = collections.deque(maxlen=max_l)
            self._matrix_rows = collections.deque(maxlen=max_l)
            self._mean_angles = collections.deque(maxlen=max_l)
            self._directionality = collections.deque(maxlen=max_l)
            self._mean_speeds = collections.deque(maxlen=max_l)
            self._total_vectors = collections.deque(maxlen=max_l)
            self._timestamps = collections.deque(maxlen=max_l)
            self._frame_indices = collections.deque(maxlen=max_l)
            self._frame_counter = 0
            self.current_t = 0.0
            self._last_arrival_time = None
            self._pending_render = False
            self._worker_busy = False
            self._batch_loaded = False
            self._last_frame_idx = None
            self.last_stats = {
                "total_frames": 0,
                "current_frame": 0,
                "elapsed_time": 0.0,
                "latest_mean_angle_deg": 0.0,
                "latest_directionality_r": 0.0,
                "latest_mean_speed": 0.0,
                "latest_total_vectors": 0,
            }

    def clear_buffer(self, *args: Any, **kwargs: Any) -> None:
        """Clear accumulated vector buffer and reset spectrogram."""
        self._reset_buffers_locked()
        if self.processor is not None:
            self.processor.clear()
            self._worker_busy = True
            self.processor.submit([])
        if dpg.does_item_exist(self.stats_text_tag):
            frame_str = f"Frames: 0/{self.history_size}" if self.rolling_mode else "Frames: 0"
            dpg.set_value(
                self.stats_text_tag,
                f"{frame_str} (0.00s) | Current Mean: 0.0° (Right) | R: 0.00 | Speed: 0.00 µm/s | Vectors: 0",
            )

    def _trigger_render(self, force: bool = False) -> None:
        """Submit current compact matrix payload to background processor for rendering."""
        if self.processor is None:
            return

        with self._buffer_lock:
            if len(self._matrix_rows) == 0:
                self._worker_busy = True
                self.processor.submit([])
                return

            now = time.perf_counter()
            if not force:
                busy_recently = self._worker_busy and (now - self._last_render_submit_time < 0.75)
                if busy_recently or not self.processor.is_ready() or (now - self._last_render_submit_time < 0.055):
                    self._pending_render = True
                    return

            self._pending_render = False
            self._last_render_submit_time = now
            self._worker_busy = True

            payload: Dict[str, Any] = {
                "matrix": np.array(self._matrix_rows, dtype=np.float32),
                "mean_angles_deg": np.array(self._mean_angles, dtype=np.float32),
                "directionality_r": np.array(self._directionality, dtype=np.float32),
                "mean_speeds": np.array(self._mean_speeds, dtype=np.float32),
                "total_vectors_arr": np.array(self._total_vectors, dtype=np.int32),
                "timestamps": np.array(self._timestamps, dtype=np.float32),
                "frame_indices": np.array(self._frame_indices, dtype=np.int32),
            }
        self.processor.submit(payload)

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> bool:
        """Handle incoming vector streams, samples, and pipeline command triggers."""
        data_type = kwargs.get("data_type")
        payload = (
            data
            if data is not None
            else (kwargs.get("sample") or kwargs.get("cmd") or (args[0] if args else None))
        )

        if data_type == IOTypes.TRIGGER:
            if isinstance(payload, str):
                cmd_upper = payload.upper()
                if cmd_upper in ("START", "RESET", "CLEAR"):
                    self.clear_buffer()
                elif cmd_upper == "PAUSE":
                    self.is_paused = True
                    if dpg.does_item_exist(self.pause_check_tag):
                        dpg.set_value(self.pause_check_tag, True)
                elif cmd_upper == "RESUME":
                    self.is_paused = False
                    if dpg.does_item_exist(self.pause_check_tag):
                        dpg.set_value(self.pause_check_tag, False)
                elif cmd_upper == "EXPORT":
                    self.copy_to_clipboard()
            return True

        if data_type == IOTypes.CMD_DICT and isinstance(payload, dict):
            action = str(payload.get("action", "")).lower()
            if action in ("clear", "reset"):
                self.clear_buffer()
                return True
            if action == "pause":
                self.is_paused = True
                if dpg.does_item_exist(self.pause_check_tag):
                    dpg.set_value(self.pause_check_tag, True)
                return True
            if action == "resume":
                self.is_paused = False
                if dpg.does_item_exist(self.pause_check_tag):
                    dpg.set_value(self.pause_check_tag, False)
                return True
            if action == "export_image":
                self.save_image_to_file()
                return True
            if action == "export_data":
                self.save_csv_to_file()
                return True

        if self.is_paused:
            return True

        # Batch vectors import
        if isinstance(payload, dict) and (
            payload.get("action") == "batch_vectors"
            or ("frames" in payload and isinstance(payload.get("frames"), (list, tuple)))
        ):
            self._reset_buffers_locked()
            if self.processor is not None:
                self.processor.clear()

            if "fps" in payload:
                try:
                    self.fps = max(0.1, float(payload["fps"]))
                    self.delta_t = float(payload.get("delta_t", 1.0 / self.fps))
                    if dpg.does_item_exist(self.fps_input_tag):
                        dpg.set_value(self.fps_input_tag, self.fps)
                except Exception:
                    pass

            frames_list = payload.get("frames", [])
            if not frames_list:
                self.clear_buffer()
                return True

            b_raw: List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, float, int]] = []
            b_mat: List[np.ndarray] = []
            b_mean: List[float] = []
            b_dir: List[float] = []
            b_spd: List[float] = []
            b_tot: List[int] = []
            b_ts: List[float] = []
            b_idx: List[int] = []

            curr_t = 0.0
            frame_cnt = 0

            for frame_item in frames_list:
                if not isinstance(frame_item, dict):
                    continue
                angles_clean, mags_clean = PolarSpectrogramProcessor.clean_vector_inputs(
                    frame_item.get("angles"), frame_item.get("magnitudes")
                )
                ang_rep, cnt_w, mag_rep = PolarSpectrogramProcessor.compress_frame_vectors(angles_clean, mags_clean)

                f_idx = int(frame_item.get("frame_idx", frame_cnt))
                t_val = float(frame_item.get("timestamp", curr_t))
                curr_t = t_val + self.delta_t
                frame_cnt = f_idx + 1

                b_raw.append((ang_rep, cnt_w, mag_rep, t_val, f_idx))

                row, mean_ang, r_val, spd, n_vec = PolarSpectrogramProcessor.bin_single_frame(
                    ang_rep, mag_rep, self.num_bins, self.weight_by_magnitude, self.invert_y, count_weights=cnt_w
                )

                b_mat.append(row)
                b_mean.append(mean_ang)
                b_dir.append(r_val)
                b_spd.append(spd)
                b_tot.append(n_vec)
                b_ts.append(t_val)
                b_idx.append(f_idx)

            with self._buffer_lock:
                self.raw_history = collections.deque(b_raw, maxlen=None)
                self._matrix_rows = collections.deque(b_mat, maxlen=None)
                self._mean_angles = collections.deque(b_mean, maxlen=None)
                self._directionality = collections.deque(b_dir, maxlen=None)
                self._mean_speeds = collections.deque(b_spd, maxlen=None)
                self._total_vectors = collections.deque(b_tot, maxlen=None)
                self._timestamps = collections.deque(b_ts, maxlen=None)
                self._frame_indices = collections.deque(b_idx, maxlen=None)
                self.current_t = curr_t
                self._frame_counter = frame_cnt
                self._batch_loaded = True
                self._last_source = str(payload.get("source", ""))
                self._last_frame_idx = b_idx[-1] if b_idx else None

            self._update_processor_params()
            self._trigger_render(force=True)
            logger.success(f"Polar Spectrogram batch imported: {len(frames_list)} frames processed in one shot.")
            return True

        angles: Any = None
        magnitudes: Any = None
        pkt_frame_idx: Optional[int] = None
        pkt_source: str = ""

        if isinstance(payload, dict):
            if "frame_idx" in payload and payload["frame_idx"] is not None:
                try:
                    pkt_frame_idx = int(payload["frame_idx"])
                except Exception:
                    pkt_frame_idx = None
            if "source" in payload and payload["source"] is not None:
                pkt_source = str(payload["source"])

            if "angles" in payload:
                angles = payload.get("angles")
                magnitudes = payload.get("magnitudes")
            elif "y" in payload:
                angles = payload.get("y")
                magnitudes = payload.get("x")
        elif isinstance(payload, (list, tuple, np.ndarray)):
            angles = payload

        if angles is not None and len(angles) > 0:
            with self._buffer_lock:
                should_reset = (
                    self._batch_loaded
                    or (bool(pkt_source) and bool(self._last_source) and pkt_source != self._last_source)
                    or (
                        pkt_frame_idx is not None
                        and self._last_frame_idx is not None
                        and pkt_frame_idx < self._last_frame_idx
                    )
                )
                if should_reset:
                    self._reset_buffers_locked()
                    if self.processor is not None:
                        self.processor.clear()

                if pkt_source:
                    self._last_source = pkt_source
                if pkt_frame_idx is not None:
                    self._last_frame_idx = pkt_frame_idx

            angles_clean, mags_clean = PolarSpectrogramProcessor.clean_vector_inputs(angles, magnitudes)
            ang_rep, cnt_w, mag_rep = PolarSpectrogramProcessor.compress_frame_vectors(angles_clean, mags_clean)

            now = time.perf_counter()
            if self.auto_delta_t:
                dt = (now - self._last_arrival_time) if self._last_arrival_time is not None else self.delta_t
                self._last_arrival_time = now
                self.current_t += dt
                f_idx = pkt_frame_idx if pkt_frame_idx is not None else self._frame_counter
                self._frame_counter = f_idx + 1
            elif pkt_frame_idx is not None:
                f_idx = pkt_frame_idx
                self.current_t = f_idx * self.delta_t
                self._frame_counter = f_idx + 1
            else:
                self.current_t += self.delta_t
                f_idx = self._frame_counter
                self._frame_counter += 1

            row, mean_ang, r_val, spd, n_vec = PolarSpectrogramProcessor.bin_single_frame(
                ang_rep, mag_rep, self.num_bins, self.weight_by_magnitude, self.invert_y, count_weights=cnt_w
            )

            with self._buffer_lock:
                self.raw_history.append((ang_rep, cnt_w, mag_rep, self.current_t, f_idx))
                self._matrix_rows.append(row)
                self._mean_angles.append(mean_ang)
                self._directionality.append(r_val)
                self._mean_speeds.append(spd)
                self._total_vectors.append(n_vec)
                self._timestamps.append(self.current_t)
                self._frame_indices.append(f_idx)

            self._trigger_render()

        return True

    def _monitor_loop(self) -> None:
        """
        Background polling loop retrieving rendered spectrogram frames from
        PolarSpectrogramProcessor and updating the DearPyGui texture.
        """
        while self._monitor_thread_running:
            try:
                res = self.processor.get() if self.processor is not None else None
                if res is not None:
                    while self.processor is not None:
                        newer = self.processor.get()
                        if newer is None:
                            break
                        res = newer

                    self._worker_busy = False
                    img_bgr, stats = res
                    self.last_bgr_frame = img_bgr
                    self.last_stats = stats

                    if self.connections.get("Frame"):
                        self._emit_frame(img_bgr)

                    if (
                        dpg.does_item_exist(self.texture_tag)
                        and img_bgr.shape[0] == self._tex_buffer.shape[0]
                        and img_bgr.shape[1] == self._tex_buffer.shape[1]
                    ):
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        np.multiply(img_rgb, 1.0 / 255.0, out=self._tex_buffer, casting="unsafe")
                        dpg.set_value(self.texture_tag, self._tex_buffer.ravel())

                    if dpg.does_item_exist(self.stats_text_tag):
                        tot_f = stats.get("total_frames", 0)
                        el_t = stats.get("elapsed_time", 0.0)
                        mean_ang = stats.get("latest_mean_angle_deg", 0.0)
                        dir_r = stats.get("latest_directionality_r", 0.0)
                        spd = stats.get("latest_mean_speed", 0.0)
                        vecs = stats.get("latest_total_vectors", 0)

                        cardinal = "Up ↑" if 0.0 <= mean_ang <= 180.0 else "Down ↓"
                        frame_str = f"Frames: {tot_f}/{self.history_size}" if self.rolling_mode else f"Frames: {tot_f}"

                        dpg.set_value(
                            self.stats_text_tag,
                            f"{frame_str} ({el_t:.2f}s) | "
                            f"Current Mean: {mean_ang:.1f}° ({cardinal}) | "
                            f"R: {dir_r:.2f} | Speed: {spd:.2f} µm/s | Vectors: {vecs}",
                        )

                    if self._pending_render:
                        self._trigger_render()
                else:
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"Error in PolarSpectrogramWin monitor loop: {e}")
                time.sleep(0.02)

    def _emit_frame(self, frame: np.ndarray) -> None:
        """Emit BGR frame to connected downstream modules."""
        for module in self.connections.get("Frame", []):
            if hasattr(module, "input_cb"):
                try:
                    module.input_cb(data=frame, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.debug(f"Polar Spectrogram frame emit error: {e}")

    def copy_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """Copy the latest polar spectrogram image to the system clipboard."""
        if self.last_bgr_frame is not None:
            clipboardinjector.send_image(self.last_bgr_frame)
            logger.success("Polar Spectrogram image copied to clipboard.")
        else:
            logger.warning("No spectrogram frame available to copy.")

    def save_image_to_file(self, *args: Any, **kwargs: Any) -> None:
        """Save current polar spectrogram image to a PNG file."""
        if self.last_bgr_frame is None:
            logger.warning("No spectrogram frame available to save.")
            return

        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"polar_spectrogram_{timestamp_str}.png"
            filepath = Path(filename).resolve()
            cv2.imwrite(str(filepath), self.last_bgr_frame)
            logger.success(f"Spectrogram image saved: [{filename}](file:///{filepath.as_posix()})")
        except Exception as e:
            logger.error(f"Failed to save spectrogram image: {e}")

    def export_tsv_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """Export full 2D spectrogram matrix and circular statistics as TSV to the clipboard."""
        tsv_content = self._build_tsv_content()
        if tsv_content:
            clipboardinjector.send_text(tsv_content)
            logger.success(
                f"Polar Spectrogram ({self.num_bins} bins, {len(self.raw_history)} frames) copied to clipboard as TSV."
            )
        else:
            logger.warning("No vector data to export.")

    def save_csv_to_file(self, *args: Any, **kwargs: Any) -> None:
        """Save full 2D spectrogram matrix and circular statistics to a CSV/TSV file."""
        tsv_content = self._build_tsv_content()
        if not tsv_content:
            logger.warning("No vector data to export.")
            return

        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"polar_spectrogram_{timestamp_str}.tsv"
            filepath = Path(filename).resolve()
            filepath.write_text(tsv_content, encoding="utf-8")
            logger.success(f"Spectrogram data saved: [{filename}](file:///{filepath.as_posix()})")
        except Exception as e:
            logger.error(f"Failed to save spectrogram TSV: {e}")

    def save_npz_to_file(self, *args: Any, **kwargs: Any) -> None:
        """Save spectrogram matrix, timestamps, bin centers, and circular statistics as a NumPy .NPZ file."""
        stats = self.last_stats
        matrix = stats.get("matrix")
        if matrix is None or len(matrix) == 0:
            logger.warning("No spectrogram matrix data to save.")
            return

        with self._buffer_lock:
            raw_mat = stats.get(
                "raw_matrix",
                np.array(self._matrix_rows, dtype=np.float32) if self._matrix_rows else matrix,
            )
            timestamps = stats.get("timestamps", np.array(self._timestamps, dtype=np.float32))
            frame_indices = stats.get("frame_indices", np.array(self._frame_indices, dtype=np.int32))
            mean_angles_deg = stats.get("mean_angles_deg", np.array(self._mean_angles, dtype=np.float32))
            directionality_r = stats.get("directionality_r", np.array(self._directionality, dtype=np.float32))
            mean_speeds = stats.get("mean_speeds", np.array(self._mean_speeds, dtype=np.float32))
            total_vectors = stats.get("total_vectors_arr", np.array(self._total_vectors, dtype=np.int32))

        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"polar_spectrogram_{timestamp_str}.npz"
            filepath = Path(filename).resolve()

            np.savez_compressed(
                str(filepath),
                matrix=matrix,
                raw_matrix=raw_mat,
                timestamps=timestamps,
                frame_indices=frame_indices,
                bin_centers_deg=stats.get("bin_centers_deg", np.array([])),
                mean_angles_deg=mean_angles_deg,
                directionality_r=directionality_r,
                mean_speeds=mean_speeds,
                total_vectors=total_vectors,
                fps=self.fps,
                delta_t=self.delta_t,
                num_bins=self.num_bins,
                temporal_window=self.temporal_window,
                smooth_time=self.smooth_time,
                smooth_angle=self.smooth_angle,
                interpolation=self.interpolation,
                normalization=self.normalization,
                color_scale_mode=self.color_scale_mode,
                color_vmin=self.color_vmin,
                color_vmax=self.color_vmax,
                weight_by_magnitude=self.weight_by_magnitude,
                invert_y=self.invert_y,
            )
            logger.success(f"Spectrogram NumPy archive saved: [{filename}](file:///{filepath.as_posix()})")
        except Exception as e:
            logger.error(f"Failed to save spectrogram NPZ: {e}")

    def _build_tsv_content(self) -> str:
        """Build TSV representation of the 2D spectrogram matrix and metadata."""
        stats = self.last_stats
        matrix = stats.get("matrix")
        if matrix is None or len(matrix) == 0:
            return ""

        with self._buffer_lock:
            timestamps = stats.get("timestamps", np.array(self._timestamps, dtype=np.float32))
            frame_indices = stats.get("frame_indices", np.array(self._frame_indices, dtype=np.int32))
            bin_centers_deg = stats.get("bin_centers_deg", np.array([]))
            mean_angles_deg = stats.get("mean_angles_deg", np.array(self._mean_angles, dtype=np.float32))
            directionality_r = stats.get("directionality_r", np.array(self._directionality, dtype=np.float32))
            mean_speeds = stats.get("mean_speeds", np.array(self._mean_speeds, dtype=np.float32))
            total_vectors_arr = stats.get("total_vectors_arr", np.array(self._total_vectors, dtype=np.int32))

        lines: List[str] = [
            "# Polar Spectrogram Vector Distribution Export",
            f"# Total Frames\t{len(matrix)}",
            f"# Number of Bins\t{self.num_bins}",
            f"# FPS\t{self.fps:.2f}",
            f"# Delta T (s)\t{self.delta_t:.5f}",
            f"# Normalization\t{self.normalization}",
            f"# Color Range Mode\t{self.color_scale_mode}",
            f"# Color Range Min\t{self.color_vmin:.2f}",
            f"# Color Range Max\t{self.color_vmax:.2f}",
            f"# Temporal Window (Avg Frames)\t{self.temporal_window}",
            f"# Sigma Time (Gaussian)\t{self.smooth_time:.2f}",
            f"# Sigma Angle (Gaussian)\t{self.smooth_angle:.2f}",
            f"# Interpolation\t{self.interpolation}",
            f"# Weight by Speed\t{self.weight_by_magnitude}",
            f"# Invert Y\t{self.invert_y}",
            f"# Orientation\t{self.orientation}",
            "",
        ]

        headers = [
            "Frame",
            "Time_s",
            "Total_Vectors",
            "Mean_Angle_deg",
            "Directionality_R",
            "Mean_Speed_um_s",
            *[f"Bin_{bc:.1f}deg" for bc in bin_centers_deg],
        ]
        lines.append("\t".join(headers))

        for i, row in enumerate(matrix):
            f_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            t_val = float(timestamps[i]) if i < len(timestamps) else (i * self.delta_t)
            v_cnt = int(total_vectors_arr[i]) if i < len(total_vectors_arr) else 0
            m_ang = float(mean_angles_deg[i]) if i < len(mean_angles_deg) else 0.0
            r_val = float(directionality_r[i]) if i < len(directionality_r) else 0.0
            spd_val = float(mean_speeds[i]) if (mean_speeds is not None and i < len(mean_speeds)) else 0.0

            row_items = [
                str(f_idx),
                f"{t_val:.4f}",
                str(v_cnt),
                f"{m_ang:.2f}",
                f"{r_val:.4f}",
                f"{spd_val:.2f}",
                *[f"{val:.3f}" for val in row],
            ]
            lines.append("\t".join(row_items))

        return "\n".join(lines)

    def is_ready(self) -> bool:
        """Check if module is ready to accept data."""
        return True

    def on_close(self) -> None:
        """Clean shutdown of background monitor thread, multiprocessing worker, and GPU textures."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        self._monitor_thread_running = False
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=0.3)
        self._monitor_thread = None
        if self.processor is not None:
            self.processor.stop()
            self.processor = None
        if dpg.does_item_exist(self.texture_registry_tag):
            try:
                dpg.delete_item(self.texture_registry_tag)
            except Exception:
                pass
        super().on_close()

    def close(self) -> None:
        """Programmatically close the Polar Spectrogram window."""
        self.on_close()
        if dpg.does_item_exist(self.winID):
            dpg.delete_item(self.winID)


EXPORTED_CLASS = PolarSpectrogramWin
EXPORTED_NAME = "Polar Spectrogram"
Polar_spectogram_win = PolarSpectrogramWin
