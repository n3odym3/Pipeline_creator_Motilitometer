from __future__ import annotations
from typing import Any
import collections
import threading
import time
import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from loguru import logger

from config.display_scaling import display_scaling
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.tools.clipboard_injector import clipboardinjector
from modules.plot.polar_plot import PolarPlotProcessor


class PolarPlotWin(WindowBase):
    """
    Polar Plot window module for visualizing global vector distributions and orientation fields
    using Matplotlib polar projections rendered in a background multiprocessing worker (ProcessingBase).

    Features:
    - Multiprocessing rendering via ProcessingBase for smooth, non-blocking UI.
    - Directional histogram / Rose diagram with configurable bins.
    - Angle-based HSV coloration matching Counter Tracker's optical flow visualization.
    - Automatic and manual Light / Dark theme support.
    - Y-axis inversion (Image -> Screen coordinate correction) so video movement matches plot direction.
    - Accumulation mode: Batch of X frames (default) or Sliding window (rolling every frame).
    - Magnitude/speed weighting option.
    - Circular statistics (mean circular angle, directionality index R, mean speed).
    - Mean resultant vector arrow overlay.
    - Emits rendered BGR frames to downstream modules and supports clipboard export.
    """

    # 1. Window Configuration
    label = "Polar Plot"
    win_width = display_scaling.scale(620)
    win_height = display_scaling.scale(680)

    # 2. Input/Output definitions
    accepted_input_types = [
        IOTypes.VECTORS,
        IOTypes.SAMPLE,
        IOTypes.DATALIST,
        IOTypes.CMD_DICT,
        IOTypes.TRIGGER,
    ]
    outputs = {
        "Frame": IOTypes.FRAME,
        "CMD": IOTypes.CMD_DICT,
    }

    # 3. Persistent fields
    persistent_fields = [
        "label", "accumulation", "replot_every_frame", "num_bins", "weight_by_magnitude",
        "show_mean_vector", "zero_location", "clockwise", "color_mode",
        "colormap", "theme_mode", "invert_y", "plot_title", "normalize",
        "canvas_width", "canvas_height", "dpi",
    ]

    # Persistent field defaults
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

        self._monitor_thread = None
        self._monitor_thread_running = False
        self.processor = None

        # Parameters & persistent fields
        self.accumulation = max(1, int(self.accumulation if self.accumulation is not None else 30))
        if "replot" in kwargs:
            self.replot_every_frame = bool(kwargs["replot"])
        else:
            self.replot_every_frame = bool(self.replot_every_frame if self.replot_every_frame is not None else True)

        self.num_bins = int(self.num_bins if self.num_bins is not None else 24)
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
        self.canvas_width = int(self.canvas_width if self.canvas_width is not None else 1000)
        self.canvas_height = int(self.canvas_height if self.canvas_height is not None else 1000)
        self.dpi = int(self.dpi if self.dpi is not None else 100)

        # Accumulation buffers
        self.vector_buffer = collections.deque(maxlen=self.accumulation)
        self._pending_frames = []

        # Texture state
        self.texture_registry_tag = f"polar_registry_{self.UUID}"
        self.texture_tag = f"polar_texture_{self.UUID}"
        self.last_bgr_frame = None
        self.last_stats = {
            "mean_angle_deg": 0.0,
            "mean_angle_rad": 0.0,
            "directionality_r": 0.0,
            "mean_speed": 0.0,
            "total_vectors": 0,
            "accumulated_frames": 0,
        }

        # UI Tags
        self.accum_input_tag = f"polar_accum_{self.UUID}"
        self.replot_check_tag = f"polar_replot_{self.UUID}"
        self.bins_slider_tag = f"polar_bins_{self.UUID}"
        self.weight_check_tag = f"polar_weight_{self.UUID}"
        self.mean_arrow_check_tag = f"polar_mean_arr_{self.UUID}"
        self.color_mode_combo_tag = f"polar_colormode_{self.UUID}"
        self.theme_combo_tag = f"polar_theme_{self.UUID}"
        self.invert_y_check_tag = f"polar_inverty_{self.UUID}"
        self.zero_loc_combo_tag = f"polar_zero_loc_{self.UUID}"
        self.dir_combo_tag = f"polar_dir_{self.UUID}"
        self.cmap_combo_tag = f"polar_cmap_{self.UUID}"
        self.stats_text_tag = f"polar_stats_{self.UUID}"
        self.options_header_tag = f"polar_options_header_{self.UUID}"
        self.title_input_tag = f"polar_title_{self.UUID}"
        self.normalize_check_tag = f"polar_normalize_{self.UUID}"

        # Initialize DPG raw texture
        with dpg.texture_registry(tag=self.texture_registry_tag, show=False):
            dpg.add_raw_texture(
                width=self.canvas_width,
                height=self.canvas_height,
                default_value=np.zeros(self.canvas_width * self.canvas_height * 3, dtype=np.float32),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb
            )

        # Build UI
        self._build_ui()
        self.update_permission()

        # Initialize ProcessingBase worker & background monitor thread
        self.processor = PolarPlotProcessor(
            params=self._get_processor_params(),
            buffer_size=5,
            drop_policy="drop_oldest",
            daemon=True
        )
        self.processor.start()

        self._monitor_thread_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        # Request initial empty plot
        self.processor.submit([])

    def _is_light_theme(self) -> bool:
        """Helper to determine whether the light theme is currently effective."""
        if self.theme_mode == "Light":
            return True
        elif self.theme_mode == "Dark":
            return False
        else:
            try:
                from config.theme_manager import theme_manager
                return bool(theme_manager.active_palette.get("is_light", False))
            except Exception:
                return False

    def _get_theme_colors(self):
        """Determines the color styling based on the active theme mode."""
        if self._is_light_theme():
            return {
                "fig_bg": "#f4f5f7",
                "ax_bg": "#ffffff",
                "text_color": "#1f2937",
                "tick_color": "#4b5563",
                "grid_color": "#d1d5db",
                "bar_edge": "#374151",
                "arrow_color": "#0284c7",
                "arrow_edge": "#0f172a",
                "stats_color": (30, 80, 150)
            }
        else:
            return {
                "fig_bg": "#1e1e1e",
                "ax_bg": "#141414",
                "text_color": "#ffffff",
                "tick_color": "#cccccc",
                "grid_color": "#444444",
                "bar_edge": "#ffffff",
                "arrow_color": "#00ffff",
                "arrow_edge": "#ffffff",
                "stats_color": (180, 220, 255)
            }

    def _get_processor_params(self) -> dict:
        """Returns the full dictionary of parameters for PolarPlotProcessor."""
        return {
            "accumulation": self.accumulation,
            "num_bins": self.num_bins,
            "weight_by_magnitude": self.weight_by_magnitude,
            "show_mean_vector": self.show_mean_vector,
            "zero_location": self.zero_location,
            "clockwise": self.clockwise,
            "color_mode": self.color_mode,
            "colormap": self.colormap,
            "theme_mode": self.theme_mode,
            "is_light": self._is_light_theme(),
            "invert_y": self.invert_y,
            "plot_title": self.plot_title,
            "normalize": self.normalize,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "dpi": self.dpi,
        }

    def _update_processor_params(self):
        """Sends updated parameters to the running background processor."""
        if hasattr(self, 'processor') and self.processor:
            self.processor.update_params(**self._get_processor_params())

    def update_permission(self):
        """Adjust module permissions and UI visibility based on app_state.mode."""
        from core.app_state import app_state
        mode = app_state.mode
        if dpg.does_item_exist(self.winID):
            is_user = (mode == "user")
            dpg.configure_item(self.winID, no_close=is_user)

    def _build_ui(self):
        scale = display_scaling.scale
        with self.window():
            # Collapsible Options & Export menu
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

                # Accumulation & analysis settings
                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="Accumulation",
                        default_value=self.accumulation,
                        min_value=1,
                        max_value=5000,
                        width=scale(120),
                        step=1,
                        callback=self._on_accumulation_changed,
                        tag=self.accum_input_tag
                    )
                    with dpg.tooltip(self.accum_input_tag):
                        dpg.add_text("Number of frames to accumulate vectors from.\n"
                                     "In batch mode (default): Replots every X frames.\n"
                                     "In rolling mode: Rolling window over last X frames.")

                    dpg.add_checkbox(
                        label="Rolling",
                        default_value=self.replot_every_frame,
                        callback=self._on_replot_toggled,
                        tag=self.replot_check_tag
                    )
                    with dpg.tooltip(self.replot_check_tag):
                        dpg.add_text("Unchecked (default): Plots ONLY once every X frames (disjoint chunks of X frames).\n"
                                     "Checked: Rolling window (replots on every single frame).")

                    dpg.add_slider_int(
                        label="Bins",
                        default_value=self.num_bins,
                        min_value=8,
                        max_value=72,
                        width=scale(95),
                        callback=self._on_bins_changed,
                        tag=self.bins_slider_tag
                    )
                    with dpg.tooltip(self.bins_slider_tag):
                        dpg.add_text("Number of angular sectors.")

                    dpg.add_checkbox(
                        label="Weight Speed",
                        default_value=self.weight_by_magnitude,
                        callback=self._on_weight_toggled,
                        tag=self.weight_check_tag
                    )
                    with dpg.tooltip(self.weight_check_tag):
                        dpg.add_text("Weight angular sectors by vector speed / magnitude.")

                    dpg.add_checkbox(
                        label="Mean Vector",
                        default_value=self.show_mean_vector,
                        callback=self._on_mean_vector_toggled,
                        tag=self.mean_arrow_check_tag
                    )
                    with dpg.tooltip(self.mean_arrow_check_tag):
                        dpg.add_text("Overlay mean resultant circular direction arrow.")

                # Display & color mapping settings
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        items=["Angle (HSV Flow)", "Angle (Colormap)", "Magnitude / Intensity"],
                        label="Color",
                        default_value=self.color_mode,
                        width=scale(140),
                        callback=self._on_color_mode_changed,
                        tag=self.color_mode_combo_tag
                    )
                    with dpg.tooltip(self.color_mode_combo_tag):
                        dpg.add_text("Angle (HSV Flow): Exact same angle color wheel as Counter Tracker (Right=Red, Down=Green, Left=Cyan, Up=Blue)\n"
                                     "Angle (Colormap): Uses chosen colormap (Turbo, Viridis, Twilight...)\n"
                                     "Magnitude / Intensity: Color intensity based on vector count/speed")

                    dpg.add_combo(
                        items=["turbo", "viridis", "twilight", "twilight_shifted", "plasma", "coolwarm", "magma", "ocean", "hsv"],
                        label="Cmap",
                        default_value=self.colormap,
                        width=scale(95),
                        callback=self._on_colormap_changed,
                        tag=self.cmap_combo_tag
                    )
                    with dpg.tooltip(self.cmap_combo_tag):
                        dpg.add_text("Colormap used when Color mode is Angle (Colormap) or Magnitude.")

                    dpg.add_combo(
                        items=["Auto", "Dark", "Light"],
                        label="Theme",
                        default_value=self.theme_mode,
                        width=scale(75),
                        callback=self._on_theme_changed,
                        tag=self.theme_combo_tag
                    )

                    dpg.add_checkbox(
                        label="Invert Y",
                        default_value=self.invert_y,
                        callback=self._on_invert_y_toggled,
                        tag=self.invert_y_check_tag
                    )
                    with dpg.tooltip(self.invert_y_check_tag):
                        dpg.add_text("Inverts Y axis to map image coordinates (where +Y is down) to screen coordinates.\n"
                                     "When enabled, downward movement on video points DOWN on the polar plot.")

                    dpg.add_checkbox(
                        label="Normalize (%)",
                        default_value=self.normalize,
                        callback=self._on_normalize_toggled,
                        tag=self.normalize_check_tag
                    )
                    with dpg.tooltip(self.normalize_check_tag):
                        dpg.add_text("Normalize radial sector heights to percentages (%) of total.")

                # Geometry & plot title settings
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        items=["East (0° right)", "North (0° top)"],
                        label="Zero Ref",
                        default_value="East (0° right)" if self.zero_location == "E" else "North (0° top)",
                        width=scale(120),
                        callback=self._on_zero_loc_changed,
                        tag=self.zero_loc_combo_tag
                    )
                    with dpg.tooltip(self.zero_loc_combo_tag):
                        dpg.add_text("Zero angle reference location (East or North).")

                    dpg.add_combo(
                        items=["Counter-Clockwise", "Clockwise"],
                        label="Direction",
                        default_value="Clockwise" if self.clockwise else "Counter-Clockwise",
                        width=scale(130),
                        callback=self._on_direction_changed,
                        tag=self.dir_combo_tag
                    )
                    with dpg.tooltip(self.dir_combo_tag):
                        dpg.add_text("Theta increase direction (Clockwise or Counter-Clockwise).")

                    dpg.add_input_text(
                        label="Title",
                        default_value=self.plot_title,
                        hint="Custom plot title",
                        width=scale(150),
                        callback=self._on_title_changed,
                        tag=self.title_input_tag
                    )
                    with dpg.tooltip(self.title_input_tag):
                        dpg.add_text("Custom plot title (leave blank for automatic summary).")

                # Canvas resolution & DPI
                with dpg.tree_node(label="Canvas Dimensions & DPI", default_open=False):
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(label="Width", default_value=self.canvas_width, width=scale(80),
                                          callback=lambda s, a: setattr(self, "canvas_width", a))
                        dpg.add_input_int(label="Height", default_value=self.canvas_height, width=scale(80),
                                          callback=lambda s, a: setattr(self, "canvas_height", a))
                        dpg.add_input_int(label="DPI", default_value=self.dpi, width=scale(65),
                                          callback=lambda s, a: setattr(self, "dpi", a))
                        dpg.add_button(label="Apply Dimensions", callback=self._apply_options)

            # Live circular statistics
            tc = self._get_theme_colors()
            dpg.add_text(
                "Frames: 0 | Vectors: 0 | Mean: 0.0° | R: 0.00 | Speed: 0.00 µm/s",
                tag=self.stats_text_tag,
                color=tc["stats_color"]
            )

            dpg.add_separator()

            # Plot display container
            with dpg.group(tag=f"polar_plot_wrapper_{self.UUID}"):
                self._build_plot_widget()

    def _build_plot_widget(self):
        with dpg.plot(
            height=-1,
            width=-1,
            no_menus=True,
            tag=f"polar_plot_widget_{self.UUID}",
            parent=f"polar_plot_wrapper_{self.UUID}"
        ):
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True)
            with dpg.plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True) as yaxis:
                dpg.add_image_series(self.texture_tag, [0, 0], [self.canvas_width, self.canvas_height])
                dpg.fit_axis_data(xaxis)
                dpg.fit_axis_data(yaxis)

    def _show_options_modal(self, *args, **kwargs):
        config_win = f"polar_config_{self.UUID}"
        if dpg.does_item_exist(config_win):
            dpg.show_item(config_win)
            dpg.focus_item(config_win)
            return

        with dpg.window(
            label=f"Options - {self.label}",
            tag=config_win,
            width=340,
            height=420,
            pos=(self.pos[0] + 40, self.pos[1] + 40)
        ):
            dpg.add_input_text(label="Title", default_value=self.plot_title,
                               callback=lambda s, a: setattr(self, "plot_title", a))
            dpg.add_checkbox(
                label="Rolling (Replot every frame)",
                default_value=self.replot_every_frame,
                callback=self._on_replot_toggled
            )
            dpg.add_combo(
                items=["Angle (HSV Flow)", "Angle (Colormap)", "Magnitude / Intensity"],
                label="Color Mode",
                default_value=self.color_mode,
                callback=self._on_color_mode_changed
            )
            dpg.add_combo(
                items=["Auto", "Dark", "Light"],
                label="Theme",
                default_value=self.theme_mode,
                callback=self._on_theme_changed
            )
            dpg.add_combo(
                items=["turbo", "viridis", "twilight", "twilight_shifted", "plasma", "coolwarm", "magma", "ocean", "hsv"],
                label="Colormap",
                default_value=self.colormap,
                callback=self._on_colormap_changed,
                tag=self.cmap_combo_tag
            )
            dpg.add_combo(
                items=["East (0° right)", "North (0° top)"],
                label="Zero Ref",
                default_value="East (0° right)" if self.zero_location == "E" else "North (0° top)",
                callback=self._on_zero_loc_changed,
                tag=self.zero_loc_combo_tag
            )
            dpg.add_combo(
                items=["Counter-Clockwise", "Clockwise"],
                label="Orientation",
                default_value="Clockwise" if self.clockwise else "Counter-Clockwise",
                callback=self._on_direction_changed,
                tag=self.dir_combo_tag
            )
            dpg.add_checkbox(
                label="Invert Y (Image -> Screen)",
                default_value=self.invert_y,
                callback=self._on_invert_y_toggled
            )
            dpg.add_checkbox(
                label="Normalize (%)",
                default_value=self.normalize,
                callback=lambda s, a: setattr(self, "normalize", a)
            )
            dpg.add_separator()
            dpg.add_input_int(label="Canvas Width", default_value=self.canvas_width,
                              callback=lambda s, a: setattr(self, "canvas_width", a))
            dpg.add_input_int(label="Canvas Height", default_value=self.canvas_height,
                              callback=lambda s, a: setattr(self, "canvas_height", a))
            dpg.add_input_int(label="DPI", default_value=self.dpi,
                              callback=lambda s, a: setattr(self, "dpi", a))
            dpg.add_separator()
            dpg.add_button(label="Apply and Refresh", width=-1, callback=self._apply_options)

    def _apply_options(self, *args, **kwargs):
        self._update_processor_params()
        if self.canvas_width != dpg.get_item_configuration(self.texture_tag).get("width") or \
           self.canvas_height != dpg.get_item_configuration(self.texture_tag).get("height"):
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
                parent=self.texture_registry_tag
            )
            self._build_plot_widget()

        self._trigger_render()

    def _on_accumulation_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.accumulation = max(1, int(app_data))
        current_items = list(self.vector_buffer)
        self.vector_buffer = collections.deque(current_items, maxlen=self.accumulation)
        self._update_processor_params()
        if len(self._pending_frames) >= self.accumulation:
            self.vector_buffer = collections.deque(self._pending_frames, maxlen=self.accumulation)
            self._pending_frames = []
            self._trigger_render()

    def _on_replot_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.replot_every_frame = bool(app_data)
        if dpg.does_item_exist(self.replot_check_tag):
            dpg.set_value(self.replot_check_tag, self.replot_every_frame)
        if self.replot_every_frame and self._pending_frames:
            for item in self._pending_frames:
                self.vector_buffer.append(item)
            self._pending_frames = []
            self._trigger_render()

    def _on_bins_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.num_bins = int(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_weight_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.weight_by_magnitude = bool(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_mean_vector_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.show_mean_vector = bool(app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_color_mode_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.color_mode = app_data
        if dpg.does_item_exist(self.color_mode_combo_tag):
            dpg.set_value(self.color_mode_combo_tag, app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_theme_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.theme_mode = app_data
        if dpg.does_item_exist(self.theme_combo_tag):
            dpg.set_value(self.theme_combo_tag, app_data)
        tc = self._get_theme_colors()
        if dpg.does_item_exist(self.stats_text_tag):
            dpg.configure_item(self.stats_text_tag, color=tc["stats_color"])
        self._update_processor_params()
        self._trigger_render()

    def _on_invert_y_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.invert_y = bool(app_data)
        if dpg.does_item_exist(self.invert_y_check_tag):
            dpg.set_value(self.invert_y_check_tag, self.invert_y)
        self._update_processor_params()
        self._trigger_render()

    def _on_colormap_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.colormap = app_data
        self._update_processor_params()
        self._trigger_render()

    def _on_zero_loc_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.zero_location = "N" if "North" in app_data else "E"
        self._update_processor_params()
        self._trigger_render()

    def _on_direction_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.clockwise = ("Clockwise" == app_data)
        if dpg.does_item_exist(self.dir_combo_tag):
            dpg.set_value(self.dir_combo_tag, app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_title_changed(self, sender, app_data, user_data=None, *args, **kwargs):
        self.plot_title = app_data
        if dpg.does_item_exist(self.title_input_tag):
            dpg.set_value(self.title_input_tag, app_data)
        self._update_processor_params()
        self._trigger_render()

    def _on_normalize_toggled(self, sender, app_data, user_data=None, *args, **kwargs):
        self.normalize = bool(app_data)
        if dpg.does_item_exist(self.normalize_check_tag):
            dpg.set_value(self.normalize_check_tag, self.normalize)
        self._update_processor_params()
        self._trigger_render()

    def clear_buffer(self, *args, **kwargs):
        """Clears all accumulated vectors and pending frames."""
        self.vector_buffer.clear()
        self._pending_frames.clear()
        self.last_stats = {
            "mean_angle_deg": 0.0,
            "mean_angle_rad": 0.0,
            "directionality_r": 0.0,
            "mean_speed": 0.0,
            "total_vectors": 0,
            "accumulated_frames": 0,
        }
        if hasattr(self, 'processor') and self.processor:
            self.processor.submit([])
        if dpg.does_item_exist(self.stats_text_tag):
            dpg.set_value(
                self.stats_text_tag,
                "Frames: 0 | Vectors: 0 | Mean: 0.0° | R: 0.00 | Speed: 0.00 µm/s"
            )

    def _trigger_render(self):
        """Submits current vector buffer to background processor."""
        if hasattr(self, 'processor') and self.processor:
            data = list(self.vector_buffer)
            self.processor.submit(data)

    def input_cb(self, data=None, *args, **kwargs):
        """
        Handles incoming vector orientations or commands from connected modules.
        """
        data_type = kwargs.get("data_type")
        payload = data if data is not None else (kwargs.get("sample") or kwargs.get("cmd") or (args[0] if args else None))

        if data_type == IOTypes.TRIGGER:
            if isinstance(payload, str) and payload.upper() == "START":
                self.clear_buffer()
            return True

        if data_type == IOTypes.CMD_DICT and isinstance(payload, dict):
            if payload.get("action") in ["clear", "reset"]:
                self.clear_buffer()
                return True

        # Process vector data
        angles = []
        magnitudes = []

        if isinstance(payload, dict):
            if "angles" in payload:
                angles = payload.get("angles", [])
                magnitudes = payload.get("magnitudes", [])
            elif "y" in payload:
                # SAMPLE format fallback
                angles = payload.get("y", [])
                magnitudes = payload.get("x", [])

        elif isinstance(payload, (list, np.ndarray)):
            angles = payload
            magnitudes = []

        if len(angles) > 0:
            angles_arr = np.asarray(angles, dtype=np.float32)
            if len(magnitudes) == len(angles):
                mags_arr = np.asarray(magnitudes, dtype=np.float32)
            else:
                mags_arr = np.ones_like(angles_arr, dtype=np.float32)

            frame_data = {
                "angles": angles_arr,
                "magnitudes": mags_arr
            }

            if self.replot_every_frame:
                # Rolling / Sliding window mode: replots every frame
                self.vector_buffer.append(frame_data)
                self._trigger_render()
            else:
                # Batch mode (default): accumulates X frames, then plots and resets for next X frames
                self._pending_frames.append(frame_data)
                if len(self._pending_frames) >= self.accumulation:
                    self.vector_buffer = collections.deque(self._pending_frames, maxlen=self.accumulation)
                    self._pending_frames = []
                    self._trigger_render()
                else:
                    # Update live accumulation progress text without heavy processing
                    if dpg.does_item_exist(self.stats_text_tag):
                        last_ang = self.last_stats['mean_angle_deg']
                        last_r = self.last_stats['directionality_r']
                        dpg.set_value(
                            self.stats_text_tag,
                            f"Accumulating: {len(self._pending_frames)}/{self.accumulation} frames | "
                            f"Last Mean: {last_ang:.1f}° | Last R: {last_r:.2f}"
                        )

    def _monitor_loop(self):
        """
        Background thread polling processed polar plot frames from PolarPlotProcessor
        and updating DPG textures and downstream output ports.
        """
        while self._monitor_thread_running:
            try:
                res = self.processor.get()
                if res is not None:
                    img_bgr, stats = res
                    self.last_bgr_frame = img_bgr
                    self.last_stats = stats

                    # Emit frame to downstream nodes
                    self._emit_frame(img_bgr)

                    # Update Dear PyGui Raw Texture
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    flattened = img_rgb.ravel().astype(np.float32) / 255.0
                    if dpg.does_item_exist(self.texture_tag) and flattened.size == (self.canvas_width * self.canvas_height * 3):
                        dpg.set_value(self.texture_tag, flattened)

                    # Update Stats Text Label
                    if dpg.does_item_exist(self.stats_text_tag):
                        dpg.set_value(
                            self.stats_text_tag,
                            f"Frames: {stats.get('accumulated_frames', 0)} | Vectors: {stats.get('total_vectors', 0)} | "
                            f"Mean: {stats.get('mean_angle_deg', 0.0):.1f}° | R: {stats.get('directionality_r', 0.0):.2f} | "
                            f"Speed: {stats.get('mean_speed', 0.0):.2f} µm/s"
                        )
                else:
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"Error in PolarPlotWin monitor loop: {e}")
                time.sleep(0.02)

    def _emit_frame(self, frame):
        for module in self.connections.get("Frame", []):
            if hasattr(module, 'input_cb'):
                try:
                    module.input_cb(data=frame, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.debug(f"Polar Plot frame emit error: {e}")

    def copy_to_clipboard(self, *args, **kwargs):
        """Copies the latest polar plot image to the clipboard."""
        if self.last_bgr_frame is not None:
            clipboardinjector.send_image(self.last_bgr_frame)
            logger.success("Polar plot image copied to clipboard.")
        else:
            logger.warning("No polar plot to copy.")

    def export_tsv_to_clipboard(self, *args, **kwargs):
        """Exports circular histogram sector counts and stats to clipboard in TSV format."""
        try:
            if not self.vector_buffer:
                logger.warning("No vector data to export.")
                return

            pooled_angles = []
            pooled_mags = []
            for item in self.vector_buffer:
                ang = item.get("angles")
                mag = item.get("magnitudes")
                if ang is not None and len(ang) > 0:
                    pooled_angles.append(ang)
                    if mag is not None and len(mag) == len(ang):
                        pooled_mags.append(mag)
                    else:
                        pooled_mags.append(np.ones_like(ang))

            if not pooled_angles:
                return

            all_angles = np.concatenate(pooled_angles)
            all_mags = np.concatenate(pooled_mags)
            all_angles = np.mod(all_angles, 2 * np.pi)

            if self.invert_y:
                plot_angles = np.mod(-all_angles, 2 * np.pi)
            else:
                plot_angles = all_angles

            theta_bins = np.linspace(0, 2 * np.pi, self.num_bins + 1)
            width = 2 * np.pi / self.num_bins
            bin_centers = theta_bins[:-1] + width / 2.0

            counts_raw, _ = np.histogram(plot_angles, bins=theta_bins)
            counts_weighted, _ = np.histogram(plot_angles, bins=theta_bins, weights=all_mags)

            lines = ["# Polar Plot Vector Statistics"]
            lines.append(f"# Total Vectors\t{self.last_stats['total_vectors']}")
            lines.append(f"# Accumulated Frames\t{self.last_stats['accumulated_frames']}")
            lines.append(f"# Mean Angle (deg)\t{self.last_stats['mean_angle_deg']:.2f}")
            lines.append(f"# Mean Angle (rad)\t{self.last_stats['mean_angle_rad']:.4f}")
            lines.append(f"# Directionality (R)\t{self.last_stats['directionality_r']:.4f}")
            lines.append(f"# Mean Speed (um/s)\t{self.last_stats['mean_speed']:.2f}")
            lines.append("")
            lines.append("Bin Center (deg)\tBin Center (rad)\tCount\tSpeed Weighted")

            for bc, cr, cw in zip(bin_centers, counts_raw, counts_weighted):
                deg = np.degrees(bc)
                lines.append(f"{deg:.1f}\t{bc:.4f}\t{cr}\t{cw:.2f}")

            tsv_content = "\n".join(lines)
            clipboardinjector.send_text(tsv_content)
            logger.success(f"Polar vector distribution ({self.num_bins} bins) copied to clipboard as TSV.")

        except Exception as e:
            logger.error(f"Error exporting Polar TSV: {e}")

    def is_ready(self) -> bool:
        """
        Check if the module is ready to receive data.

        Returns:
            bool: Always True.
        """
        return True

    def on_close(self):
        """Clean shutdown of background monitor thread and multiprocessing worker."""
        self._monitor_thread_running = False
        if getattr(self, 'processor', None) is not None:
            self.processor.stop()

    def close(self):
        self.on_close()
        super().close()


EXPORTED_CLASS = PolarPlotWin
EXPORTED_NAME = "Polar Plot"
