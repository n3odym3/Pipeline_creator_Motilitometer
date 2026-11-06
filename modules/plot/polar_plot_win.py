from __future__ import annotations

import collections
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
from modules.plot.polar_plot import PolarPlotProcessor

_TWO_PI: float = 2.0 * np.pi


class PolarPlotWin(WindowBase):
    """
    Polar Plot window module for visualizing global vector distributions and orientation fields
    using Matplotlib polar projections rendered in a background multiprocessing worker.
    """

    label: str = "Polar Plot"
    win_width: int = display_scaling.scale(620)
    win_height: int = display_scaling.scale(680)

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
        "accumulation",
        "replot_every_frame",
        "num_bins",
        "weight_by_magnitude",
        "show_mean_vector",
        "zero_location",
        "clockwise",
        "color_mode",
        "colormap",
        "theme_mode",
        "invert_y",
        "plot_title",
        "normalize",
        "canvas_width",
        "canvas_height",
        "dpi",
    ]

    accumulation: int = 30
    replot_every_frame: bool = True
    num_bins: int = 24
    weight_by_magnitude: bool = True
    show_mean_vector: bool = True
    zero_location: str = "E"
    clockwise: bool = False
    color_mode: str = "Angle (HSV Flow)"
    colormap: str = "turbo"
    theme_mode: str = "Auto"
    invert_y: bool = True
    plot_title: str = ""
    normalize: bool = False
    canvas_width: int = 1000
    canvas_height: int = 1000
    dpi: int = 100

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_thread_running: bool = False
        self.processor: Optional[PolarPlotProcessor] = None

        self.accumulation = max(1, int(self.accumulation if self.accumulation is not None else 30))
        if "replot" in kwargs:
            self.replot_every_frame = bool(kwargs["replot"])
        else:
            self.replot_every_frame = bool(self.replot_every_frame if self.replot_every_frame is not None else True)

        self.num_bins = max(8, min(360, int(self.num_bins if self.num_bins is not None else 24)))
        self.weight_by_magnitude = bool(self.weight_by_magnitude if self.weight_by_magnitude is not None else True)
        self.show_mean_vector = bool(self.show_mean_vector if self.show_mean_vector is not None else True)
        self.zero_location = str(self.zero_location or "E")
        self.clockwise = bool(self.clockwise if self.clockwise is not None else False)
        self.color_mode = str(self.color_mode or "Angle (HSV Flow)")
        self.colormap = str(self.colormap or "turbo")
        self.theme_mode = str(self.theme_mode or "Auto")
        self.invert_y = bool(self.invert_y if self.invert_y is not None else True)
        self.plot_title = str(self.plot_title or "")
        self.normalize = bool(self.normalize if self.normalize is not None else False)
        self.canvas_width = max(200, int(self.canvas_width if self.canvas_width is not None else 1000))
        self.canvas_height = max(200, int(self.canvas_height if self.canvas_height is not None else 1000))
        self.dpi = max(50, int(self.dpi if self.dpi is not None else 100))

        self.vector_buffer: Deque[Dict[str, np.ndarray]] = collections.deque(maxlen=self.accumulation)
        self._pending_frames: List[Dict[str, np.ndarray]] = []
        self._batch_loaded: bool = False
        self._last_frame_idx: Optional[int] = None

        self.texture_registry_tag: str = f"polar_registry_{self.UUID}"
        self.texture_tag: str = f"polar_texture_{self.UUID}"
        self.last_bgr_frame: Optional[np.ndarray] = None
        self.last_stats: Dict[str, Any] = {
            "mean_angle_deg": 0.0,
            "mean_angle_rad": 0.0,
            "directionality_r": 0.0,
            "mean_speed": 0.0,
            "total_vectors": 0,
            "accumulated_frames": 0,
        }

        self.accum_input_tag: str = f"polar_accum_{self.UUID}"
        self.replot_check_tag: str = f"polar_replot_{self.UUID}"
        self.bins_slider_tag: str = f"polar_bins_{self.UUID}"
        self.weight_check_tag: str = f"polar_weight_{self.UUID}"
        self.mean_arrow_check_tag: str = f"polar_mean_arr_{self.UUID}"
        self.color_mode_combo_tag: str = f"polar_colormode_{self.UUID}"
        self.theme_combo_tag: str = f"polar_theme_{self.UUID}"
        self.invert_y_check_tag: str = f"polar_inverty_{self.UUID}"
        self.zero_loc_combo_tag: str = f"polar_zero_loc_{self.UUID}"
        self.dir_combo_tag: str = f"polar_dir_{self.UUID}"
        self.cmap_combo_tag: str = f"polar_cmap_{self.UUID}"
        self.stats_text_tag: str = f"polar_stats_{self.UUID}"
        self.options_header_tag: str = f"polar_options_header_{self.UUID}"
        self.title_input_tag: str = f"polar_title_{self.UUID}"
        self.normalize_check_tag: str = f"polar_normalize_{self.UUID}"

        with dpg.texture_registry(tag=self.texture_registry_tag, show=False):
            dpg.add_raw_texture(
                width=self.canvas_width,
                height=self.canvas_height,
                default_value=np.zeros(self.canvas_width * self.canvas_height * 3, dtype=np.float32),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb,
            )

        self._build_ui()
        self.update_permission()

        self.processor = PolarPlotProcessor(
            params=self._get_processor_params(),
            buffer_size=5,
            drop_policy="drop_oldest",
            daemon=True,
        )
        self.processor.start()

        self._monitor_thread_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self.processor.submit([])

    def _is_light_theme(self) -> bool:
        """Determine whether the light theme is currently active."""
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
        return (30, 80, 150) if self._is_light_theme() else (180, 220, 255)

    def _get_processor_params(self) -> Dict[str, Any]:
        """Return parameter dictionary for PolarPlotProcessor."""
        return {
            "num_bins": self.num_bins,
            "weight_by_magnitude": self.weight_by_magnitude,
            "show_mean_vector": self.show_mean_vector,
            "zero_location": self.zero_location,
            "clockwise": self.clockwise,
            "color_mode": self.color_mode,
            "colormap": self.colormap,
            "is_light": self._is_light_theme(),
            "invert_y": self.invert_y,
            "plot_title": self.plot_title,
            "normalize": self.normalize,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "dpi": self.dpi,
        }

    def _update_processor_params(self) -> None:
        """Send updated parameters to the background processor."""
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
                        dpg.add_text("Copy current polar plot image to clipboard.")

                    btn_tsv = dpg.add_button(label="Export TSV", callback=self.export_tsv_to_clipboard)
                    with dpg.tooltip(btn_tsv):
                        dpg.add_text("Export circular histogram sector counts and stats as TSV to clipboard.")

                    btn_clear = dpg.add_button(label="Reset / Clear", callback=self.clear_buffer)
                    with dpg.tooltip(btn_clear):
                        dpg.add_text("Clear accumulated vector data and reset plot.")

                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="Accumulation",
                        default_value=self.accumulation,
                        min_value=1,
                        max_value=5000,
                        width=scale(120),
                        step=1,
                        callback=self._on_accumulation_changed,
                        tag=self.accum_input_tag,
                    )
                    with dpg.tooltip(self.accum_input_tag):
                        dpg.add_text(
                            "Number of frames to accumulate vectors from.\n"
                            "In batch mode (default): Replots every X frames.\n"
                            "In rolling mode: Rolling window over last X frames."
                        )

                    dpg.add_checkbox(
                        label="Rolling",
                        default_value=self.replot_every_frame,
                        callback=self._on_replot_toggled,
                        tag=self.replot_check_tag,
                    )
                    with dpg.tooltip(self.replot_check_tag):
                        dpg.add_text(
                            "Unchecked (default): Plots ONLY once every X frames (disjoint chunks of X frames).\n"
                            "Checked: Rolling window (replots on every single frame)."
                        )

                    dpg.add_slider_int(
                        label="Bins",
                        default_value=self.num_bins,
                        min_value=8,
                        max_value=72,
                        width=scale(95),
                        callback=self._on_bins_changed,
                        tag=self.bins_slider_tag,
                    )
                    with dpg.tooltip(self.bins_slider_tag):
                        dpg.add_text("Number of angular sectors.")

                    dpg.add_checkbox(
                        label="Weight Speed",
                        default_value=self.weight_by_magnitude,
                        callback=self._on_weight_toggled,
                        tag=self.weight_check_tag,
                    )
                    with dpg.tooltip(self.weight_check_tag):
                        dpg.add_text("Weight angular sectors by vector speed / magnitude.")

                    dpg.add_checkbox(
                        label="Mean Vector",
                        default_value=self.show_mean_vector,
                        callback=self._on_mean_vector_toggled,
                        tag=self.mean_arrow_check_tag,
                    )
                    with dpg.tooltip(self.mean_arrow_check_tag):
                        dpg.add_text("Overlay mean resultant circular direction arrow.")

                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        items=["Angle (HSV Flow)", "Angle (Colormap)", "Magnitude / Intensity"],
                        label="Color",
                        default_value=self.color_mode,
                        width=scale(140),
                        callback=self._on_color_mode_changed,
                        tag=self.color_mode_combo_tag,
                    )
                    with dpg.tooltip(self.color_mode_combo_tag):
                        dpg.add_text(
                            "Angle (HSV Flow): Exact same angle color wheel as Counter Tracker (Right=Red, Down=Green, Left=Cyan, Up=Blue)\n"
                            "Angle (Colormap): Uses chosen colormap (Turbo, Viridis, Twilight...)\n"
                            "Magnitude / Intensity: Color intensity based on vector count/speed"
                        )

                    dpg.add_combo(
                        items=[
                            "turbo",
                            "viridis",
                            "twilight",
                            "twilight_shifted",
                            "plasma",
                            "coolwarm",
                            "magma",
                            "ocean",
                            "hsv",
                        ],
                        label="Cmap",
                        default_value=self.colormap,
                        width=scale(95),
                        callback=self._on_colormap_changed,
                        tag=self.cmap_combo_tag,
                    )
                    with dpg.tooltip(self.cmap_combo_tag):
                        dpg.add_text("Colormap used when Color mode is Angle (Colormap) or Magnitude.")

                    dpg.add_combo(
                        items=["Auto", "Dark", "Light"],
                        label="Theme",
                        default_value=self.theme_mode,
                        width=scale(75),
                        callback=self._on_theme_changed,
                        tag=self.theme_combo_tag,
                    )

                    dpg.add_checkbox(
                        label="Invert Y",
                        default_value=self.invert_y,
                        callback=self._on_invert_y_toggled,
                        tag=self.invert_y_check_tag,
                    )
                    with dpg.tooltip(self.invert_y_check_tag):
                        dpg.add_text(
                            "Inverts Y axis to map image coordinates (where +Y is down) to screen coordinates.\n"
                            "When enabled, downward movement on video points DOWN on the polar plot."
                        )

                    dpg.add_checkbox(
                        label="Normalize (%)",
                        default_value=self.normalize,
                        callback=self._on_normalize_toggled,
                        tag=self.normalize_check_tag,
                    )
                    with dpg.tooltip(self.normalize_check_tag):
                        dpg.add_text("Normalize radial sector heights to percentages (%) of total.")

                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        items=["East (0° right)", "North (0° top)"],
                        label="Zero Ref",
                        default_value="East (0° right)" if self.zero_location == "E" else "North (0° top)",
                        width=scale(120),
                        callback=self._on_zero_loc_changed,
                        tag=self.zero_loc_combo_tag,
                    )
                    with dpg.tooltip(self.zero_loc_combo_tag):
                        dpg.add_text("Zero angle reference location (East or North).")

                    dpg.add_combo(
                        items=["Counter-Clockwise", "Clockwise"],
                        label="Direction",
                        default_value="Clockwise" if self.clockwise else "Counter-Clockwise",
                        width=scale(130),
                        callback=self._on_direction_changed,
                        tag=self.dir_combo_tag,
                    )
                    with dpg.tooltip(self.dir_combo_tag):
                        dpg.add_text("Theta increase direction (Clockwise or Counter-Clockwise).")

                    dpg.add_input_text(
                        label="Title",
                        default_value=self.plot_title,
                        hint="Custom plot title",
                        width=scale(150),
                        callback=self._on_title_changed,
                        tag=self.title_input_tag,
                    )
                    with dpg.tooltip(self.title_input_tag):
                        dpg.add_text("Custom plot title (leave blank for automatic summary).")

                with dpg.tree_node(label="Canvas Dimensions & DPI", default_open=False):
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(
                            label="Width",
                            default_value=self.canvas_width,
                            min_value=200,
                            min_clamped=True,
                            step=0,
                            step_fast=0,
                            width=scale(80),
                            callback=lambda s, a, *args, **kwargs: setattr(self, "canvas_width", max(200, int(a))),
                        )
                        dpg.add_input_int(
                            label="Height",
                            default_value=self.canvas_height,
                            min_value=200,
                            min_clamped=True,
                            step=0,
                            step_fast=0,
                            width=scale(80),
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
                            width=scale(65),
                            callback=lambda s, a, *args, **kwargs: setattr(self, "dpi", max(50, int(a))),
                        )
                        dpg.add_button(label="Apply Dimensions", callback=self._apply_options)

            dpg.add_text(
                "Frames: 0 | Vectors: 0 | Mean: 0.0° | R: 0.00 | Speed: 0.00 µm/s",
                tag=self.stats_text_tag,
                color=self._get_stats_color(),
            )

            dpg.add_separator()

            with dpg.group(tag=f"polar_plot_wrapper_{self.UUID}"):
                self._build_plot_widget()

    def _build_plot_widget(self) -> None:
        with dpg.plot(
            height=-1,
            width=-1,
            no_menus=True,
            tag=f"polar_plot_widget_{self.UUID}",
            parent=f"polar_plot_wrapper_{self.UUID}",
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
            if dpg.does_item_exist(self.texture_tag):
                dpg.delete_item(self.texture_tag)
            if dpg.does_item_exist(f"polar_plot_widget_{self.UUID}"):
                dpg.delete_item(f"polar_plot_widget_{self.UUID}")

            dpg.add_raw_texture(
                width=self.canvas_width,
                height=self.canvas_height,
                default_value=np.zeros(self.canvas_width * self.canvas_height * 3, dtype=np.float32),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb,
                parent=self.texture_registry_tag,
            )
            self._build_plot_widget()

        self._trigger_render()

    def _on_accumulation_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.accumulation = max(1, int(app_data))
        self.vector_buffer = collections.deque(self.vector_buffer, maxlen=self.accumulation)
        if len(self._pending_frames) >= self.accumulation:
            self.vector_buffer = collections.deque(self._pending_frames, maxlen=self.accumulation)
            self._pending_frames.clear()
            self._trigger_render()

    def _on_replot_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.replot_every_frame = bool(app_data)
        if self.replot_every_frame and self._pending_frames:
            self.vector_buffer.extend(self._pending_frames)
            self._pending_frames.clear()
            self._trigger_render()

    def _on_bins_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.num_bins = max(8, min(360, int(app_data)))
        self._update_processor_params()
        self._trigger_render()

    def _on_weight_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.weight_by_magnitude = bool(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_mean_vector_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.show_mean_vector = bool(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_color_mode_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.color_mode = str(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_theme_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.theme_mode = str(app_data)
        if dpg.does_item_exist(self.stats_text_tag):
            dpg.configure_item(self.stats_text_tag, color=self._get_stats_color())
        self._update_processor_params()
        self._trigger_render()

    def _on_invert_y_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.invert_y = bool(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_colormap_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.colormap = str(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_zero_loc_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.zero_location = "N" if "North" in str(app_data) else "E"
        self._update_processor_params()
        self._trigger_render()

    def _on_direction_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.clockwise = str(app_data) == "Clockwise"
        self._update_processor_params()
        self._trigger_render()

    def _on_title_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.plot_title = str(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_normalize_toggled(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        self.normalize = bool(app_data)
        self._update_processor_params()
        self._trigger_render()

    def clear_buffer(self, *args: Any, **kwargs: Any) -> None:
        """Clear all accumulated vectors and pending frames."""
        self.vector_buffer.clear()
        self._pending_frames.clear()
        self._batch_loaded = False
        self._last_frame_idx = None
        self.last_stats = {
            "mean_angle_deg": 0.0,
            "mean_angle_rad": 0.0,
            "directionality_r": 0.0,
            "mean_speed": 0.0,
            "total_vectors": 0,
            "accumulated_frames": 0,
        }
        if self.processor is not None:
            self.processor.clear()
            self.processor.submit([])
        if dpg.does_item_exist(self.stats_text_tag):
            dpg.set_value(
                self.stats_text_tag,
                "Frames: 0 | Vectors: 0 | Mean: 0.0° | R: 0.00 | Speed: 0.00 µm/s",
            )

    def _trigger_render(self) -> None:
        """Submit current vector buffer to background processor."""
        if self.processor is not None:
            self.processor.submit(list(self.vector_buffer))

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> bool:
        """Handle incoming vector orientations or commands from connected modules."""
        data_type = kwargs.get("data_type")
        payload = (
            data
            if data is not None
            else (kwargs.get("sample") or kwargs.get("cmd") or (args[0] if args else None))
        )

        if data_type == IOTypes.TRIGGER:
            if isinstance(payload, str) and payload.upper() in ("START", "RESET", "CLEAR"):
                self.clear_buffer()
            return True

        if data_type == IOTypes.CMD_DICT and isinstance(payload, dict):
            if str(payload.get("action", "")).lower() in ("clear", "reset"):
                self.clear_buffer()
                return True

        if isinstance(payload, dict) and payload.get("action") == "batch_vectors":
            self.vector_buffer.clear()
            self._pending_frames.clear()
            if self.processor is not None:
                self.processor.clear()
            angles_clean, mags_clean = PolarPlotProcessor.clean_vector_inputs(
                payload.get("angles"), payload.get("magnitudes")
            )
            if angles_clean.size > 0:
                self.vector_buffer.append({"angles": angles_clean, "magnitudes": mags_clean})
                self._batch_loaded = True
                self._trigger_render()
                logger.success(f"Polar Plot batch imported: {angles_clean.size} total vectors plotted.")
            return True

        angles: Any = None
        magnitudes: Any = None
        pkt_frame_idx: Optional[int] = None

        if isinstance(payload, dict):
            if "frame_idx" in payload and payload["frame_idx"] is not None:
                try:
                    pkt_frame_idx = int(payload["frame_idx"])
                except Exception:
                    pkt_frame_idx = None
            if "angles" in payload:
                angles = payload.get("angles")
                magnitudes = payload.get("magnitudes")
            elif "y" in payload:
                angles = payload.get("y")
                magnitudes = payload.get("x")
        elif isinstance(payload, (list, tuple, np.ndarray)):
            angles = payload

        if angles is not None and len(angles) > 0:
            if self._batch_loaded or (
                pkt_frame_idx is not None
                and self._last_frame_idx is not None
                and pkt_frame_idx < self._last_frame_idx
            ):
                self.vector_buffer.clear()
                self._pending_frames.clear()
                self._batch_loaded = False
                if self.processor is not None:
                    self.processor.clear()
            if pkt_frame_idx is not None:
                self._last_frame_idx = pkt_frame_idx
            angles_clean, mags_clean = PolarPlotProcessor.clean_vector_inputs(angles, magnitudes)
            if angles_clean.size == 0:
                return True

            frame_data = {
                "angles": angles_clean,
                "magnitudes": mags_clean,
            }

            if self.replot_every_frame:
                self.vector_buffer.append(frame_data)
                self._trigger_render()
            else:
                self._pending_frames.append(frame_data)
                if len(self._pending_frames) >= self.accumulation:
                    self.vector_buffer = collections.deque(self._pending_frames, maxlen=self.accumulation)
                    self._pending_frames.clear()
                    self._trigger_render()
                elif dpg.does_item_exist(self.stats_text_tag):
                    last_ang = self.last_stats["mean_angle_deg"]
                    last_r = self.last_stats["directionality_r"]
                    dpg.set_value(
                        self.stats_text_tag,
                        f"Accumulating: {len(self._pending_frames)}/{self.accumulation} frames | "
                        f"Last Mean: {last_ang:.1f}° | Last R: {last_r:.2f}",
                    )

        return True

    def _monitor_loop(self) -> None:
        """
        Background thread polling processed polar plot frames from PolarPlotProcessor
        and updating DPG textures and downstream output ports.
        """
        while self._monitor_thread_running:
            try:
                res = self.processor.get() if self.processor is not None else None
                if res is not None:
                    img_bgr, stats = res
                    self.last_bgr_frame = img_bgr
                    self.last_stats = stats

                    self._emit_frame(img_bgr)

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    flattened = img_rgb.ravel().astype(np.float32)
                    flattened *= 1.0 / 255.0
                    expected_size = self.canvas_width * self.canvas_height * 3
                    if dpg.does_item_exist(self.texture_tag) and flattened.size == expected_size:
                        dpg.set_value(self.texture_tag, flattened)

                    if dpg.does_item_exist(self.stats_text_tag):
                        dpg.set_value(
                            self.stats_text_tag,
                            f"Frames: {stats.get('accumulated_frames', 0)} | "
                            f"Vectors: {stats.get('total_vectors', 0)} | "
                            f"Mean: {stats.get('mean_angle_deg', 0.0):.1f}° | "
                            f"R: {stats.get('directionality_r', 0.0):.2f} | "
                            f"Speed: {stats.get('mean_speed', 0.0):.2f} µm/s",
                        )
                else:
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"Error in PolarPlotWin monitor loop: {e}")
                time.sleep(0.02)

    def _emit_frame(self, frame: np.ndarray) -> None:
        """Emit BGR frame to connected downstream modules."""
        for module in self.connections.get("Frame", []):
            if hasattr(module, "input_cb"):
                try:
                    module.input_cb(data=frame, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.debug(f"Polar Plot frame emit error: {e}")

    def copy_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """Copy the latest polar plot image to the clipboard."""
        if self.last_bgr_frame is not None:
            clipboardinjector.send_image(self.last_bgr_frame)
            logger.success("Polar plot image copied to clipboard.")
        else:
            logger.warning("No polar plot to copy.")

    def export_tsv_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """Export circular histogram sector counts and stats to clipboard in TSV format."""
        try:
            if not self.vector_buffer:
                logger.warning("No vector data to export.")
                return

            pooled_angles: List[np.ndarray] = []
            pooled_mags: List[np.ndarray] = []
            for item in self.vector_buffer:
                ang, mag = PolarPlotProcessor.clean_vector_inputs(
                    item.get("angles"), item.get("magnitudes")
                )
                if ang.size > 0:
                    pooled_angles.append(ang)
                    pooled_mags.append(mag)

            if not pooled_angles:
                logger.warning("No valid vector data to export.")
                return

            all_angles = np.concatenate(pooled_angles)
            all_mags = np.concatenate(pooled_mags)
            plot_angles = np.mod(-all_angles, _TWO_PI) if self.invert_y else all_angles

            num_bins = self.num_bins
            width = _TWO_PI / num_bins
            bin_centers = np.linspace(0.0, _TWO_PI, num_bins + 1, dtype=np.float32)[:-1] + width * 0.5
            bin_idx = np.clip((plot_angles * (num_bins / _TWO_PI)).astype(np.intp), 0, num_bins - 1)

            counts_raw = np.bincount(bin_idx, minlength=num_bins)
            counts_weighted = np.bincount(bin_idx, weights=all_mags, minlength=num_bins)

            lines: List[str] = [
                "# Polar Plot Vector Statistics",
                f"# Total Vectors\t{self.last_stats['total_vectors']}",
                f"# Accumulated Frames\t{self.last_stats['accumulated_frames']}",
                f"# Mean Angle (deg)\t{self.last_stats['mean_angle_deg']:.2f}",
                f"# Mean Angle (rad)\t{self.last_stats['mean_angle_rad']:.4f}",
                f"# Directionality (R)\t{self.last_stats['directionality_r']:.4f}",
                f"# Mean Speed (um/s)\t{self.last_stats['mean_speed']:.2f}",
                "",
                "Bin Center (deg)\tBin Center (rad)\tCount\tSpeed Weighted",
                *[
                    f"{ float(np.degrees(bc)):.1f}\t{float(bc):.4f}\t{int(cr)}\t{float(cw):.2f}"
                    for bc, cr, cw in zip(bin_centers, counts_raw, counts_weighted)
                ],
            ]

            clipboardinjector.send_text("\n".join(lines))
            logger.success(f"Polar vector distribution ({num_bins} bins) copied to clipboard as TSV.")

        except Exception as e:
            logger.error(f"Error exporting Polar TSV: {e}")

    def is_ready(self) -> bool:
        """Check if the module is ready to receive data."""
        return True

    def on_close(self) -> None:
        """Clean shutdown of background monitor thread and multiprocessing worker."""
        self._monitor_thread_running = False
        if self.processor is not None:
            self.processor.stop()
            self.processor = None


EXPORTED_CLASS = PolarPlotWin
EXPORTED_NAME = "Polar Plot"
