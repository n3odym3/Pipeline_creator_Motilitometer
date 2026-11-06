"""Motility Reprocessor: replays a pre-processed video's Counter Tracker data in sync with playback.

Connect ``Video Reader -> Playback`` (and optionally ``Video Reader -> Frame``) to this module.
When a video is loaded it looks for ``<video>.npz`` (written by the Counter Tracker with
"Auto Export Raw Data (npz)"). Then, for every displayed frame it re-emits, on the same
outputs as the Counter Tracker, the data up to that frame so that downstream modules
(plots, sample containers, polar plot...) cannot tell the difference.

Only distributions were stored, so Tracking / Contours / colour masks are not available.
Vectors and Raw Speeds are reconstructed from the stored histograms (angle bin of 5 deg).
"""
from __future__ import annotations

import os
import threading
import time
import uuid as _uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from core.app_state import app_state
from modules.Motilitometer.counter_tracker.angle_export import (
    PI_DEFAULT_LIGHT_POS,
    PI_DEFAULT_PRESET,
    PI_LIGHT_POSITIONS,
    PI_PRESETS,
    hist_phototaxis_series,
    in_angle_range,
    normalize_pi_light_pos,
    normalize_pi_preset,
    pi_light_arrows,
    pi_preset_ranges,
)
from modules.Motilitometer.counter_tracker.drift_processor import DriftProcessor
from modules.Motilitometer.counter_tracker.raw_export import RawData, raw_path_for

_METRIC_LABELS: Dict[str, str] = {
    "surface": "Surface %:",
    "speed": "Speed (µm/s):",
    "both": "Surface% | Speed:",
}
_GREY: Tuple[int, int, int] = (150, 150, 150)
_GREEN: Tuple[int, int, int] = (100, 200, 100)
_RED: Tuple[int, int, int] = (220, 100, 100)


class MotilityReprocessor_win(WindowBase):
    """Replays Counter Tracker raw data (.npz) synchronised with the Video Reader frame number."""

    label = "Motility Reprocessor"
    description = "Replays Counter Tracker raw data (.npz) and drift telemetry in sync with video playback."
    win_width = display_scaling.scale(390)
    win_height = display_scaling.scale(500)

    accepted_input_types = [IOTypes.PLAYBACK, IOTypes.FRAME, IOTypes.TRIGGER]
    outputs = {
        "Frame": IOTypes.FRAME,
        "CMD": IOTypes.CMD_DICT,
        "Result": IOTypes.SAMPLE,
        "Count": IOTypes.SAMPLE,
        "Motility": IOTypes.SAMPLE,
        "Raw Speeds": IOTypes.SAMPLE,
        "Vectors": IOTypes.VECTORS,
        "Phototaxis": IOTypes.SAMPLE,
    }
    persistent_fields = [
        "label",
        "active",
        "view_mode",
        "show_vector",
        "vector_scale",
        "show_drift_vector",
        "show_fake_drifters",
        "emit_drift_series",
        "x_axis_mode",
        "delta_t",
        "emit_phototaxis_series",
        "pi_light_pos",
        "pi_preset",
        "pi_up_min",
        "pi_up_max",
        "pi_down_min",
        "pi_down_max",
        "pi_weight_speed",
    ]

    active: bool = True
    view_mode: str = "Combined"
    show_vector: bool = True
    vector_scale: float = 1.0
    show_drift_vector: bool = True
    show_fake_drifters: bool = False
    emit_drift_series: bool = True
    x_axis_mode: str = "Fixed ΔT"
    delta_t: float = 1.0 / 30.0
    emit_phototaxis_series: bool = False
    pi_light_pos: str = PI_DEFAULT_LIGHT_POS
    pi_preset: str = PI_DEFAULT_PRESET
    pi_up_min: float = 180.0
    pi_up_max: float = 360.0
    pi_down_min: float = 0.0
    pi_down_max: float = 180.0
    pi_weight_speed: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        def _p(attr: str, default: Any) -> Any:
            val = getattr(self, attr, None)
            if val is None:
                val = kwargs.get(attr, default)
            return default if val is None else val

        # Defensive extraction of persistent fields
        self.active = bool(_p("active", True))
        self.view_mode = str(_p("view_mode", "Combined") or "Combined")
        self.show_vector = bool(_p("show_vector", True))
        try:
            self.vector_scale = float(_p("vector_scale", 1.0))
        except (ValueError, TypeError):
            self.vector_scale = 1.0
        self.show_drift_vector = bool(_p("show_drift_vector", True))
        self.show_fake_drifters = bool(_p("show_fake_drifters", False))
        self.emit_drift_series = bool(_p("emit_drift_series", True))
        self.x_axis_mode = str(_p("x_axis_mode", "Fixed ΔT") or "Fixed ΔT")
        try:
            self.delta_t = float(_p("delta_t", 1.0 / 30.0))
        except (ValueError, TypeError):
            self.delta_t = 1.0 / 30.0
        self.emit_phototaxis_series = bool(_p("emit_phototaxis_series", False))
        self.pi_light_pos = normalize_pi_light_pos(_p("pi_light_pos", PI_DEFAULT_LIGHT_POS))
        self.pi_preset = normalize_pi_preset(_p("pi_preset", PI_DEFAULT_PRESET))
        preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
        if preset_bounds is not None:
            self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
        else:
            try:
                self.pi_up_min = float(_p("pi_up_min", 180.0))
            except (ValueError, TypeError):
                self.pi_up_min = 180.0
            try:
                self.pi_up_max = float(_p("pi_up_max", 360.0))
            except (ValueError, TypeError):
                self.pi_up_max = 360.0
            try:
                self.pi_down_min = float(_p("pi_down_min", 0.0))
            except (ValueError, TypeError):
                self.pi_down_min = 0.0
            try:
                self.pi_down_max = float(_p("pi_down_max", 180.0))
            except (ValueError, TypeError):
                self.pi_down_max = 180.0
        self.pi_weight_speed = bool(_p("pi_weight_speed", True))

        u = self.UUID
        self.active_tag = f"motreproc_active_{u}"
        self.view_mode_tag = f"motreproc_view_mode_{u}"
        self.manip_group_tag = f"motreproc_manip_grp_{u}"
        self.show_vector_tag = f"motreproc_show_vec_{u}"
        self.vector_scale_tag = f"motreproc_vec_scale_{u}"
        self.drift_group_tag = f"motreproc_drift_grp_{u}"
        self.show_drift_vector_tag = f"motreproc_show_drift_vec_{u}"
        self.show_fake_drifters_tag = f"motreproc_show_fake_drf_{u}"
        self.emit_drift_series_tag = f"motreproc_emit_drift_ser_{u}"
        self.x_mode_tag = f"motreproc_xmode_{u}"
        self.delta_t_tag = f"motreproc_dt_{u}"
        self.status_tag = f"motreproc_status_{u}"
        self.frame_tag = f"motreproc_frame_{u}"
        self.metric_label_tag = f"motreproc_metric_lbl_{u}"
        self.metric_tag = f"motreproc_metric_{u}"
        self.count_tag = f"motreproc_count_{u}"
        self.motility_tag = f"motreproc_motility_{u}"
        self.speed_stats_tag = f"motreproc_spd_stats_{u}"
        self.vector_text_tag = f"motreproc_vec_text_{u}"
        self.drift_text_tag = f"motreproc_drift_text_{u}"
        self.emit_pi_tag = f"motreproc_emit_pi_{u}"
        self.pi_light_pos_tag = f"motreproc_pi_light_pos_{u}"
        self.pi_preset_tag = f"motreproc_pi_preset_{u}"
        self.pi_toward_lbl_tag = f"motreproc_pi_toward_lbl_{u}"
        self.pi_away_lbl_tag = f"motreproc_pi_away_lbl_{u}"
        self.pi_up_min_tag = f"motreproc_pi_up_min_{u}"
        self.pi_up_max_tag = f"motreproc_pi_up_max_{u}"
        self.pi_down_min_tag = f"motreproc_pi_down_min_{u}"
        self.pi_down_max_tag = f"motreproc_pi_down_max_{u}"
        self.pi_weight_speed_tag = f"motreproc_pi_weight_{u}"
        self.pi_text_tag = f"motreproc_pi_text_{u}"

        self._lock = threading.RLock()
        self._is_closing: bool = False
        self._data: Optional[RawData] = None
        self._drift_map: Optional[Dict[str, Any]] = None
        self._has_drift: bool = False
        self._video_fps: float = 30.0
        self._last_raw_frame: Optional[np.ndarray] = None
        self._path: Optional[str] = None
        self._source: str = ""
        self._idx: int = 0
        self._total: int = 0
        self._emitted: Dict[str, str] = {}
        self._is_full_emitted: bool = False
        self._suid_cache: Dict[Tuple[str, str], str] = {}
        self._drift_info_cache_key: Optional[Tuple[int, int, int]] = None
        self._drift_info_cache_val: Optional[Dict[str, Any]] = None
        self._cached_x_key: Optional[Tuple[int, str, float, float]] = None
        self._cached_x_full: Optional[np.ndarray] = None
        self._cached_pi_key: Optional[Tuple[int, float, float, float, float, bool]] = None
        self._cached_pi_full: Optional[np.ndarray] = None

        scale = display_scaling.scale
        t_arrow, a_arrow = pi_light_arrows(self.pi_light_pos)
        with self.window():
            dpg.add_text("No video", tag=self.status_tag, color=_GREY, wrap=scale(350))
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Active",
                    tag=self.active_tag,
                    default_value=bool(self.active),
                    callback=self._on_active_cb,
                )
                btn_send_all = dpg.add_button(label="Send All", callback=self.send_all)
                with dpg.tooltip(btn_send_all):
                    dpg.add_text(
                        "Emit complete datasets for the entire video at once:\n"
                        "- Full time series (Speed, Coverage, Count, Motility, Drift, Phototaxis) to downstream plots\n"
                        "- Full vector batch to Polar Spectrogram and Polar Plot"
                    )
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    ["Combined", "Manip", "Drift"],
                    tag=self.view_mode_tag,
                    default_value=self.view_mode,
                    width=scale(120),
                    callback=self._on_view_mode_cb,
                )
                with dpg.tooltip(self.view_mode_tag):
                    dpg.add_text("Switch between Combined (both), Manip only (.npz), or Drift only (_drift.csv / .npz).")
                dpg.add_slider_float(
                    tag=self.vector_scale_tag,
                    default_value=float(self.vector_scale),
                    min_value=0.1,
                    max_value=3.0,
                    width=-1,
                    format="x%.1f",
                    callback=self._on_param_cb,
                )
                with dpg.tooltip(self.vector_scale_tag):
                    dpg.add_text("Vector scale multiplier for overlay arrows.")
            with dpg.group(tag=self.manip_group_tag):
                dpg.add_checkbox(
                    label="Show Manip Vector",
                    tag=self.show_vector_tag,
                    default_value=bool(self.show_vector),
                    callback=self._on_param_cb,
                )
                with dpg.tooltip(self.show_vector_tag):
                    dpg.add_text("Draw the global population vector arrow from the .npz at image centre.")
            with dpg.group(tag=self.drift_group_tag, show=True):
                dpg.add_separator()
                dpg.add_text("Drift Visualization:", color=(220, 180, 255))
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Show Drift Vector",
                        tag=self.show_drift_vector_tag,
                        default_value=bool(self.show_drift_vector),
                        callback=self._on_param_cb,
                    )
                    dpg.add_checkbox(
                        label="Add Fake Drifters",
                        tag=self.show_fake_drifters_tag,
                        default_value=bool(self.show_fake_drifters),
                        callback=self._on_param_cb,
                    )
                dpg.add_checkbox(
                    label="Emit Drift Series (Speed / Angle)",
                    tag=self.emit_drift_series_tag,
                    default_value=bool(self.emit_drift_series),
                    callback=self._on_x_axis_cb,
                )
            with dpg.collapsing_header(label="Phototaxis Index", default_open=False):
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Emit Series",
                        tag=self.emit_pi_tag,
                        default_value=bool(self.emit_phototaxis_series),
                        callback=self._on_pi_setting_cb,
                    )
                    with dpg.tooltip(self.emit_pi_tag):
                        dpg.add_text(
                            "Emit Phototaxis Index (Toward - Away)/(Toward + Away) [-1, +1] time series\n"
                            "on Result and Phototaxis output ports."
                        )
                    dpg.add_checkbox(
                        label="Weight Speed",
                        tag=self.pi_weight_speed_tag,
                        default_value=bool(self.pi_weight_speed),
                        callback=self._on_pi_setting_cb,
                    )
                    with dpg.tooltip(self.pi_weight_speed_tag):
                        dpg.add_text("Weight each angle by vector speed magnitude (um/s) instead of raw vector counts.")
                with dpg.group(horizontal=True):
                    dpg.add_text("Light Pos:", color=_GREY)
                    dpg.add_combo(
                        list(PI_LIGHT_POSITIONS),
                        tag=self.pi_light_pos_tag,
                        default_value=self.pi_light_pos,
                        callback=self._on_pi_light_pos_cb,
                        width=scale(130),
                    )
                    with dpg.tooltip(self.pi_light_pos_tag):
                        dpg.add_text(
                            "Position of the light source in screen coordinates:\n"
                            "- Bottom (↓): 270° (default)\n"
                            "- Top (↑): 90°\n"
                            "- Right (→): 0°\n"
                            "- Left (←): 180°\n"
                            "Positive index (+1) = movement toward the light."
                        )
                with dpg.group(horizontal=True):
                    dpg.add_text("Preset:   ", color=_GREY)
                    dpg.add_combo(
                        list(PI_PRESETS),
                        tag=self.pi_preset_tag,
                        default_value=self.pi_preset,
                        callback=self._on_pi_preset_cb,
                        width=scale(210),
                    )
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Toward ({t_arrow}):", tag=self.pi_toward_lbl_tag, color=(100, 220, 140))
                    dpg.add_drag_float(
                        tag=self.pi_up_min_tag,
                        default_value=float(self.pi_up_min),
                        min_value=0.0,
                        max_value=360.0,
                        speed=1.0,
                        format="%.0f°",
                        width=scale(75),
                        callback=self._on_pi_custom_cb,
                    )
                    dpg.add_text("to", color=_GREY)
                    dpg.add_drag_float(
                        tag=self.pi_up_max_tag,
                        default_value=float(self.pi_up_max),
                        min_value=0.0,
                        max_value=360.0,
                        speed=1.0,
                        format="%.0f°",
                        width=scale(75),
                        callback=self._on_pi_custom_cb,
                    )
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Away ({a_arrow}):  ", tag=self.pi_away_lbl_tag, color=(230, 140, 110))
                    dpg.add_drag_float(
                        tag=self.pi_down_min_tag,
                        default_value=float(self.pi_down_min),
                        min_value=0.0,
                        max_value=360.0,
                        speed=1.0,
                        format="%.0f°",
                        width=scale(75),
                        callback=self._on_pi_custom_cb,
                    )
                    dpg.add_text("to", color=_GREY)
                    dpg.add_drag_float(
                        tag=self.pi_down_max_tag,
                        default_value=float(self.pi_down_max),
                        min_value=0.0,
                        max_value=360.0,
                        speed=1.0,
                        format="%.0f°",
                        width=scale(75),
                        callback=self._on_pi_custom_cb,
                    )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    ["Timestamp", "Index", "Fixed ΔT"],
                    tag=self.x_mode_tag,
                    width=scale(110),
                    default_value=self.x_axis_mode,
                    callback=self._on_x_axis_cb,
                )
                dpg.add_input_float(
                    tag=self.delta_t_tag,
                    default_value=float(self.delta_t),
                    width=scale(90),
                    step=0,
                    format="%.4f",
                    callback=self._on_x_axis_cb,
                )
                dpg.add_text("X axis / ΔT (s)", color=_GREY)
            dpg.add_separator()
            self._row("Frame:", self.frame_tag)
            with dpg.group(horizontal=True):
                dpg.add_text(_METRIC_LABELS["surface"], tag=self.metric_label_tag, color=_GREY)
                dpg.add_text("--", tag=self.metric_tag)
            self._row("Count:", self.count_tag)
            self._row("Motility:", self.motility_tag)
            self._row("Vectors speed:", self.speed_stats_tag)
            self._row("Global vector:", self.vector_text_tag)
            self._row("Phototaxis:", self.pi_text_tag)
            self._row("Drift:", self.drift_text_tag)

        self.update_permission()

    def update_permission(self) -> None:
        """Adjust widget permissions based on user mode."""
        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    @staticmethod
    def _row(label: str, tag: str) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text(label, color=_GREY)
            dpg.add_text("--", tag=tag)

    def _apply_preset_data(self, data: Dict[str, Any]) -> None:
        super()._apply_preset_data(data)
        self.pi_light_pos = normalize_pi_light_pos(self.pi_light_pos)
        self.pi_preset = normalize_pi_preset(self.pi_preset)
        preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
        if preset_bounds is not None:
            self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
        self._sync_pi_ui_widgets()

    # ── UI callbacks ──────────────────────────────────────────────────
    def _val(self, tag: str, default: Any = None) -> Any:
        try:
            return dpg.get_value(tag) if dpg.does_item_exist(tag) else default
        except Exception:
            return default

    def _sync_params(self) -> None:
        self.active = bool(self._val(self.active_tag, self.active))
        self.view_mode = str(self._val(self.view_mode_tag, self.view_mode) or self.view_mode)
        self.show_vector = bool(self._val(self.show_vector_tag, self.show_vector))
        self.vector_scale = float(self._val(self.vector_scale_tag, self.vector_scale))
        self.show_drift_vector = bool(self._val(self.show_drift_vector_tag, self.show_drift_vector))
        self.show_fake_drifters = bool(self._val(self.show_fake_drifters_tag, self.show_fake_drifters))
        self.emit_drift_series = bool(self._val(self.emit_drift_series_tag, self.emit_drift_series))
        self.x_axis_mode = str(self._val(self.x_mode_tag, self.x_axis_mode) or self.x_axis_mode)
        self.delta_t = float(self._val(self.delta_t_tag, self.delta_t) or self.delta_t)
        self.emit_phototaxis_series = bool(self._val(self.emit_pi_tag, self.emit_phototaxis_series))
        self.pi_light_pos = normalize_pi_light_pos(self._val(self.pi_light_pos_tag, self.pi_light_pos))
        self.pi_preset = normalize_pi_preset(self._val(self.pi_preset_tag, self.pi_preset))
        self.pi_up_min = float(self._val(self.pi_up_min_tag, self.pi_up_min))
        self.pi_up_max = float(self._val(self.pi_up_max_tag, self.pi_up_max))
        self.pi_down_min = float(self._val(self.pi_down_min_tag, self.pi_down_min))
        self.pi_down_max = float(self._val(self.pi_down_max_tag, self.pi_down_max))
        self.pi_weight_speed = bool(self._val(self.pi_weight_speed_tag, self.pi_weight_speed))

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

    def _on_pi_light_pos_cb(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        self.pi_light_pos = normalize_pi_light_pos(
            app_data if app_data is not None else self._val(self.pi_light_pos_tag, self.pi_light_pos)
        )
        preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
        if preset_bounds is not None:
            self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
        self._sync_pi_ui_widgets()
        self._on_pi_setting_cb()

    def _on_pi_preset_cb(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        self.pi_preset = normalize_pi_preset(
            app_data if app_data is not None else self._val(self.pi_preset_tag, self.pi_preset)
        )
        self.pi_light_pos = normalize_pi_light_pos(self._val(self.pi_light_pos_tag, self.pi_light_pos))
        preset_bounds = pi_preset_ranges(self.pi_preset, self.pi_light_pos)
        if preset_bounds is not None:
            self.pi_up_min, self.pi_up_max, self.pi_down_min, self.pi_down_max = preset_bounds
        self._sync_pi_ui_widgets()
        self._on_pi_setting_cb()

    def _on_pi_custom_cb(self, *args: Any, **kwargs: Any) -> None:
        self.pi_preset = "Custom Range"
        if dpg.does_item_exist(self.pi_preset_tag):
            dpg.set_value(self.pi_preset_tag, "Custom Range")
        self._on_pi_setting_cb()

    def _reemit_current(self) -> None:
        """Re-emits the current video's data (full dataset if 'Send All' was used, else up to current frame)."""
        if not self.active:
            return
        if self._is_full_emitted:
            self.send_all()
        else:
            with self._lock:
                if self._data is not None or self._drift_map is not None:
                    self._emit(self._idx)

    def _on_pi_setting_cb(self, *args: Any, **kwargs: Any) -> None:
        self._sync_params()
        was_full = self._is_full_emitted
        self._clear_downstream()
        self._is_full_emitted = was_full
        self._reemit_current()

    @staticmethod
    def _in_angle_range(angles_deg: np.ndarray, min_deg: float, max_deg: float) -> np.ndarray:
        """Check whether angles (in degrees) fall within ``[min_deg, max_deg]`` with circular wrap."""
        return in_angle_range(angles_deg, min_deg, max_deg)

    def _compute_phototaxis_series(self, d: Optional[RawData], k: Optional[int] = None) -> np.ndarray:
        """Vectorized computation of Phototaxis Index ``(Toward - Away) / (Toward + Away)`` in ``[-1, +1]``."""
        if d is None:
            return np.array([], dtype=np.float32)
        total_frames = len(d.frame_idx)
        n_frames = k if k is not None else total_frames
        if n_frames <= 0:
            return np.array([], dtype=np.float32)
        key = (
            id(d),
            float(self.pi_up_min),
            float(self.pi_up_max),
            float(self.pi_down_min),
            float(self.pi_down_max),
            bool(self.pi_weight_speed),
        )
        if self._cached_pi_key != key or self._cached_pi_full is None:
            counts = d.ang_counts if d.ang_counts is not None else None
            msums = d.ang_mag_sums if d.ang_mag_sums is not None else None
            if counts is None and msums is None:
                self._cached_pi_full = np.full(total_frames, np.nan, dtype=np.float32)
            else:
                self._cached_pi_full = hist_phototaxis_series(
                    counts if counts is not None else msums,
                    msums,
                    toward_min=self.pi_up_min,
                    toward_max=self.pi_up_max,
                    away_min=self.pi_down_min,
                    away_max=self.pi_down_max,
                    weight_by_mag=self.pi_weight_speed,
                ).astype(np.float32)
            self._cached_pi_key = key
        return self._cached_pi_full[:n_frames]

    def _compute_single_frame_pi(self, d: Optional[RawData], row: Optional[int]) -> float:
        """Compute Phototaxis Index for a single frame row."""
        if d is None or row is None:
            return float("nan")
        full_arr = self._compute_phototaxis_series(d)
        if 0 <= row < full_arr.size:
            return float(full_arr[row])
        return float("nan")

    def _compute_drift_pi_array(self, angles_deg: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        """Compute binary/ternary Phototaxis Index from drift angles and speeds."""
        in_toward = in_angle_range(angles_deg, self.pi_up_min, self.pi_up_max)
        in_away = in_angle_range(angles_deg, self.pi_down_min, self.pi_down_max)
        pi_drift = np.where(in_toward & (~in_away), 1.0, np.where(in_away & (~in_toward), -1.0, 0.0)).astype(np.float32)
        return np.where((speeds > 0.0) & np.isfinite(angles_deg), pi_drift, np.nan)

    def _should_emit_pi(self) -> bool:
        return bool(self.emit_phototaxis_series or self.connections.get("Phototaxis"))

    def _pi_conn_keys(self) -> Tuple[str, ...]:
        return ("Result", "Phototaxis") if self.emit_phototaxis_series else ("Phototaxis",)

    def _update_group_visibilities(self) -> None:
        has_manip = self._data is not None
        has_drift = self._has_drift
        mode = self.view_mode.lower()

        if not has_manip and not has_drift:
            show_manip_grp = mode in ("combined", "manip")
            show_drift_grp = mode in ("combined", "drift")
        else:
            show_manip_grp = has_manip and (mode in ("combined", "manip"))
            show_drift_grp = has_drift and (mode in ("combined", "drift"))

        if dpg.does_item_exist(self.manip_group_tag):
            dpg.configure_item(self.manip_group_tag, show=show_manip_grp)
        if dpg.does_item_exist(self.drift_group_tag):
            dpg.configure_item(self.drift_group_tag, show=show_drift_grp)

    def _dispatch_frame(self, out: Optional[np.ndarray]) -> None:
        if out is None:
            return
        for module in list(self.connections.get("Frame", [])):
            try:
                module.input_cb(data=out, data_type=IOTypes.FRAME)
            except Exception as e:
                logger.error(f"MotilityReprocessor frame dispatch error: {e}")

    def _on_view_mode_cb(self, *args: Any, **kwargs: Any) -> None:
        self._sync_params()
        self._update_group_visibilities()
        was_full = self._is_full_emitted
        self._clear_downstream()
        self._is_full_emitted = was_full
        self._reemit_current()
        with self._lock:
            out = (
                self._render_frame_overlays(self._last_raw_frame, self._idx)
                if isinstance(self._last_raw_frame, np.ndarray)
                else None
            )
        self._dispatch_frame(out)

    def _on_param_cb(self, *args: Any, **kwargs: Any) -> None:
        self._sync_params()
        with self._lock:
            out = (
                self._render_frame_overlays(self._last_raw_frame, self._idx)
                if isinstance(self._last_raw_frame, np.ndarray)
                else None
            )
        self._dispatch_frame(out)

    def _on_active_cb(self, *args: Any, **kwargs: Any) -> None:
        self._sync_params()
        if not self.active:
            self._clear_downstream()
        else:
            self._reemit_current()
        self._on_param_cb()

    def _on_x_axis_cb(self, *args: Any, **kwargs: Any) -> None:
        self._sync_params()
        was_full = self._is_full_emitted
        self._clear_downstream()
        self._is_full_emitted = was_full
        self._reemit_current()

    # ── Input ─────────────────────────────────────────────────────────
    def input_cb(self, *args: Any, **kwargs: Any) -> bool:
        data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
        data_type = kwargs.get("data_type")

        if data_type == IOTypes.PLAYBACK or (isinstance(data, dict) and "frame_idx" in data and "path" in data):
            return self._on_playback(data)
        if data_type == IOTypes.TRIGGER:
            if isinstance(data, str) and data.upper() == "START":
                self._is_full_emitted = False
                self._clear_downstream()
            elif isinstance(data, str) and data.upper() in ("SEND_ALL", "SENDALL"):
                self.send_all()
            return True
        if data_type == IOTypes.CMD_DICT and isinstance(data, dict):
            if data.get("action") in ("send_all", "sendall"):
                self.send_all()
                return True
        return self._on_frame(data)

    def _on_playback(self, pb: Dict[str, Any]) -> bool:
        path = pb.get("path")
        with self._lock:
            path_changed = path != self._path
            if path_changed:
                self._load(path, pb.get("name") or "")
            self._idx = int(pb.get("frame_idx", 0))
            self._total = int(pb.get("total_frames") or 0)
            if pb.get("fps"):
                self._video_fps = float(pb["fps"])
            if (self._data is None and self._drift_map is None) or not self.active:
                return True
            # When simply clicking a new video in Video Reader (frame 0), only refresh UI texts
            # so previous videos' emitted samples are preserved and no 1-point sample is emitted yet.
            if path_changed and self._idx == 0:
                d = self._data
                row = d.row(0) if d is not None else None
                self._update_texts(d, row, 0)
                return True
            self._is_full_emitted = False
            self._emit(self._idx)
        return True

    def _render_frame_overlays(self, frame: np.ndarray, idx: int) -> np.ndarray:
        """Draws the global population vector and/or drift vectors + fake drifters onto ``frame``."""
        d = self._data
        out = frame
        copied = False
        mode = self.view_mode.lower()
        draw_manip = (mode in ("combined", "manip")) and (d is not None) and self.active and self.show_vector
        draw_drift = (
            (mode in ("combined", "drift"))
            and self.active
            and self._has_drift
            and (self.show_drift_vector or self.show_fake_drifters)
        )

        if draw_manip and d is not None:
            row = d.row(idx)
            if row is not None:
                out = self._draw_vector(frame, d, row)
                copied = out is not frame

        if draw_drift:
            drift_info = self._get_drift_info(idx, out.shape[:2])
            if drift_info is not None and drift_info.get("enabled", False):
                if not copied:
                    out = out.copy()
                    if out.ndim == 2:
                        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
                m_px = float(drift_info.get("mean_mag_px", 0.0))
                m_um = float(drift_info.get("mean_mag_um_s", 0.0))
                fps_val = float((d.fps if d is not None else self._video_fps) or 30.0)
                calib_est = (
                    (m_um / (m_px * fps_val))
                    if (m_px > 1e-6 and m_um > 0 and fps_val > 0)
                    else (1.0 / max(fps_val, 1.0))
                )
                DriftProcessor.draw_drift_overlay(
                    out,
                    drift_info,
                    arrow_scale=max(1.0, float(self.vector_scale) * 3.0),
                    params={
                        "drift_show_vector": bool(self.show_drift_vector),
                        "drift_show_fake_drifters": bool(self.show_fake_drifters),
                        "calibration": calib_est,
                        "fps": fps_val,
                    },
                )
        return out

    def _on_frame(self, data: Any) -> bool:
        frame, idx = data, None
        if isinstance(data, (tuple, list)):
            frame = data[0] if len(data) else None
            if len(data) >= 3 and isinstance(data[2], (int, np.integer)):
                idx = int(data[2])
        out = frame
        with self._lock:
            if isinstance(frame, np.ndarray):
                self._last_raw_frame = frame
                target_idx = self._idx if idx is None else idx
                out = self._render_frame_overlays(frame, target_idx)
        self._dispatch_frame(out if isinstance(out, np.ndarray) else None)
        return True

    # ── Loading ───────────────────────────────────────────────────────
    def _set_status(self, text: str, color: Tuple[int, int, int] = _GREY) -> None:
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, text)
            dpg.configure_item(self.status_tag, color=color)

    def _load(self, path: Optional[str], name: str) -> None:
        # Detach tracking of previously emitted samples so earlier videos remain in downstream plots/containers
        self._emitted = {}
        self._is_full_emitted = False
        self._path, self._data, self._drift_map = path, None, None
        self._has_drift = False
        self._last_raw_frame = None
        self._drift_info_cache_key = None
        self._drift_info_cache_val = None
        self._cached_x_key = None
        self._cached_x_full = None
        self._cached_pi_key = None
        self._cached_pi_full = None
        DriftProcessor._fake_cell_offsets.clear()
        DriftProcessor._fake_last_frame_idx = None
        self._source = name or (os.path.splitext(os.path.basename(path))[0] if path else "")
        if not path:
            self._set_status("No video", _GREY)
            return

        npz = raw_path_for(path)
        drift_csv = DriftProcessor.drift_path_for(path)

        if os.path.isfile(drift_csv):
            try:
                self._drift_map = DriftProcessor.load_drift_csv(drift_csv)
            except Exception as e:
                logger.error(f"MotilityReprocessor failed to load drift CSV {drift_csv}: {e}")

        if os.path.isfile(npz):
            try:
                self._data = RawData(npz)
            except Exception as e:
                logger.error(f"MotilityReprocessor failed to load {npz}: {e}")

        d = self._data
        npz_has_drift = bool(
            d is not None
            and (
                (d.drift_speed is not None and np.any(np.isfinite(d.drift_speed)))
                or (d.drift_dx is not None and np.any(np.isfinite(d.drift_dx)))
            )
        )
        self._has_drift = (self._drift_map is not None) or npz_has_drift

        has_manip = d is not None
        has_drift = self._has_drift

        if has_manip and has_drift:
            modes = ["Combined", "Manip", "Drift"]
        elif has_manip:
            modes = ["Manip"]
        elif has_drift:
            modes = ["Drift"]
        else:
            modes = ["Combined", "Manip", "Drift"]

        current = self.view_mode if self.view_mode in modes else modes[0]
        self.view_mode = current
        if dpg.does_item_exist(self.view_mode_tag):
            dpg.configure_item(self.view_mode_tag, items=modes, default_value=current)
            dpg.set_value(self.view_mode_tag, current)
        self._update_group_visibilities()

        if d is None and self._drift_map is None:
            self._set_status(
                f"No preprocessed data found ({os.path.basename(npz)} or {os.path.basename(drift_csv)}).\n"
                "Run Counter Tracker with 'Auto Export Raw Data (npz)' or 'Only export drift'.",
                _RED,
            )
            return

        if d is not None:
            self._video_fps = float(d.fps or 30.0)
            drift_note = ""
            if self._drift_map is not None:
                gr, gc = self._drift_map.get("grid_shape", (1, 1))
                drift_note = f" + {os.path.basename(drift_csv)} [{gr}x{gc}]"
            elif npz_has_drift:
                drift_note = " + drift (in .npz)"
            self._set_status(f"{os.path.basename(npz)}: {len(d.frame_idx)} frames, mode '{d.mode}'{drift_note}", _GREEN)
            label = _METRIC_LABELS.get(d.metric_kind, "Value:")
            if dpg.does_item_exist(self.metric_label_tag):
                dpg.set_value(self.metric_label_tag, label)
            mode, dt = d.meta.get("x_axis_mode"), d.meta.get("delta_t")
            if mode in ("Timestamp", "Index", "Fixed ΔT") and dpg.does_item_exist(self.x_mode_tag):
                dpg.set_value(self.x_mode_tag, mode)
            if dt and dpg.does_item_exist(self.delta_t_tag):
                dpg.set_value(self.delta_t_tag, float(dt))
        elif self._drift_map is not None:
            gr, gc = self._drift_map.get("grid_shape", (1, 1))
            n_df = len(self._drift_map.get("frames", {}))
            self._set_status(
                f"Drift file detected: {os.path.basename(drift_csv)} ({n_df} frames, grid {gr}x{gc})",
                _GREEN,
            )
            if dpg.does_item_exist(self.metric_label_tag):
                dpg.set_value(self.metric_label_tag, "Drift Speed (µm/s):")
            if dpg.does_item_exist(self.show_drift_vector_tag):
                dpg.set_value(self.show_drift_vector_tag, True)
                self.show_drift_vector = True
        self._sync_params()

    def _get_drift_info(self, idx: int, frame_shape: Tuple[int, ...] = (1080, 1920)) -> Optional[Dict[str, Any]]:
        """Constructs a ``drift_info`` dict for ``idx`` from ``<video>_drift.csv`` or ``<video>.npz``."""
        h = max(1, int(frame_shape[0])) if frame_shape and len(frame_shape) >= 2 else 1080
        w = max(1, int(frame_shape[1])) if frame_shape and len(frame_shape) >= 2 else 1920

        cache_key = (id(self._data), id(self._drift_map), idx)
        if self._drift_info_cache_key == cache_key:
            cached = self._drift_info_cache_val
            if cached is not None and cached.get("frame_shape") != (h, w):
                cached["frame_shape"] = (h, w)
            return cached

        if self._drift_map is not None:
            entry = DriftProcessor._lookup_preprocessed_frame(self._drift_map, idx)
            if entry is not None:
                res = {
                    "enabled": True,
                    "frame_idx": idx,
                    "grid_shape": entry.get("grid_shape", (1, 1)),
                    "frame_shape": (h, w),
                    "locked": False,
                    "reused": True,
                    "cells": entry.get("cells", {}),
                    "mean_dx": float(entry.get("mean_dx", 0.0)),
                    "mean_dy": float(entry.get("mean_dy", 0.0)),
                    "mean_mag_px": float(entry.get("mean_mag_px", 0.0)),
                    "mean_mag_um_s": float(entry.get("mean_mag_um_s", 0.0)),
                    "mean_ang_deg": float(entry.get("mean_ang_deg", 0.0)),
                }
                self._drift_info_cache_key = cache_key
                self._drift_info_cache_val = res
                return res

        d = self._data
        if d is None or d.drift_speed is None:
            self._drift_info_cache_key = cache_key
            self._drift_info_cache_val = None
            return None
        row = d.row(idx)
        if row is None:
            self._drift_info_cache_key = cache_key
            self._drift_info_cache_val = None
            return None

        spd = float(d.drift_speed[row]) if np.isfinite(d.drift_speed[row]) else 0.0
        ang_deg = (
            float(d.drift_angle_deg[row])
            if (d.drift_angle_deg is not None and np.isfinite(d.drift_angle_deg[row]))
            else 0.0
        )
        dx = float(d.drift_dx[row]) if (d.drift_dx is not None and np.isfinite(d.drift_dx[row])) else 0.0
        dy = float(d.drift_dy[row]) if (d.drift_dy is not None and np.isfinite(d.drift_dy[row])) else 0.0
        mag_px = float(np.hypot(dx, dy))

        cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if d.drift_cell_speed is not None and d.drift_cell_speed.ndim == 3 and row < d.drift_cell_speed.shape[0]:
            g_rows, g_cols = int(d.drift_cell_speed.shape[1]), int(d.drift_cell_speed.shape[2])
            for r in range(g_rows):
                for c in range(g_cols):
                    c_spd = float(d.drift_cell_speed[row, r, c])
                    c_ang_deg = (
                        float(d.drift_cell_angle_deg[row, r, c])
                        if (d.drift_cell_angle_deg is not None and np.isfinite(d.drift_cell_angle_deg[row, r, c]))
                        else 0.0
                    )
                    c_dx = float(d.drift_cell_dx[row, r, c]) if d.drift_cell_dx is not None else dx
                    c_dy = float(d.drift_cell_dy[row, r, c]) if d.drift_cell_dy is not None else dy
                    c_mag_px = float(np.hypot(c_dx, c_dy))
                    c_ang_rad = (
                        float(np.arctan2(c_dy, c_dx) % (2.0 * np.pi))
                        if c_mag_px > 0
                        else float(np.deg2rad(c_ang_deg % 360.0))
                    )
                    cells[(r, c)] = {
                        "dx": c_dx,
                        "dy": c_dy,
                        "mag_px": c_mag_px,
                        "ang_rad": c_ang_rad,
                        "speed_um_s": c_spd,
                        "ang_deg": c_ang_deg,
                        "valid": c_mag_px > 0 or c_spd > 0,
                        "count": 0,
                    }
        else:
            g_rows, g_cols = 1, 1
            ang_rad = (
                float(np.arctan2(dy, dx) % (2.0 * np.pi))
                if mag_px > 0
                else float(np.deg2rad(ang_deg % 360.0))
            )
            cells[(0, 0)] = {
                "dx": dx,
                "dy": dy,
                "mag_px": mag_px,
                "ang_rad": ang_rad,
                "speed_um_s": spd,
                "ang_deg": ang_deg,
                "valid": mag_px > 0 or spd > 0,
                "count": 0,
            }

        res = {
            "enabled": True,
            "frame_idx": idx,
            "grid_shape": (g_rows, g_cols),
            "frame_shape": (h, w),
            "locked": False,
            "reused": True,
            "cells": cells,
            "mean_dx": dx,
            "mean_dy": dy,
            "mean_mag_px": mag_px,
            "mean_mag_um_s": spd,
            "mean_ang_deg": ang_deg,
        }
        self._drift_info_cache_key = cache_key
        self._drift_info_cache_val = res
        return res

    # ── Emission ──────────────────────────────────────────────────────
    def _suid(self, prefix: str, source: str) -> str:
        cache_key = (prefix, source)
        cached = self._suid_cache.get(cache_key)
        if cached is not None:
            return cached
        key = f"{prefix}_{self.UUID}_{source}" if prefix else f"{self.UUID}_{source}"
        val = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, key))
        self._suid_cache[cache_key] = val
        return val

    def _send_sample(self, sample: Dict[str, Any], conn_keys: Sequence[str]) -> None:
        sent = set()
        for key in conn_keys:
            for module in self.connections.get(key, []):
                if module in sent:
                    continue
                sent.add(module)
                try:
                    module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)
                except Exception as e:
                    logger.error(f"MotilityReprocessor sample dispatch error on {key}: {e}")

    def _clear_downstream(self) -> None:
        with self._lock:
            items, self._emitted = list(self._emitted.items()), {}
        for s_uuid, name in items:
            for action in ("select", "clear"):
                self._send_sample(
                    {"name": name, "uuid": s_uuid, "x": [], "y": [], "action": action},
                    ("Result", "Count", "Motility", "Phototaxis"),
                )
        for module in list(self.connections.get("Vectors", [])):
            try:
                module.input_cb(data={"action": "clear"}, data_type=IOTypes.CMD_DICT)
            except Exception as e:
                logger.error(f"MotilityReprocessor vectors clear dispatch error: {e}")

    def _build_x_from_indices(self, raw_indices: np.ndarray, fps: float) -> np.ndarray:
        raw = raw_indices.astype(np.float64)
        mode = self.x_axis_mode
        if mode == "Index":
            return raw
        if mode == "Timestamp":
            return raw / (fps or 1.0)
        return raw * float(self.delta_t or 1.0)

    def _build_x(self, d: RawData, k: int) -> np.ndarray:
        fps = float(d.fps or 1.0)
        key = (id(d), self.x_axis_mode, float(self.delta_t or 1.0), fps)
        if self._cached_x_key != key or self._cached_x_full is None:
            self._cached_x_full = self._build_x_from_indices(d.frame_idx, fps)
            self._cached_x_key = key
        return self._cached_x_full[:k]

    def _has_conn(self, keys: Sequence[str]) -> bool:
        return any(self.connections.get(ck) for ck in keys)

    def _emit_time_series(
        self,
        d: Optional[RawData],
        k: Optional[int],
        max_frame_idx: Optional[int],
        src: str,
        emit_manip: bool,
        emit_drift: bool,
    ) -> None:
        """Emit time series up to row ``k`` (or all rows if ``k`` is None)."""
        if not self._has_conn(("Result", "Count", "Motility", "Phototaxis")):
            return
        emit_pi = self._should_emit_pi()
        pi_keys = self._pi_conn_keys()

        if emit_manip and d is not None:
            n_rows = len(d.frame_idx) if k is None else k
            if n_rows > 0:
                x = self._build_x(d, n_rows)

                def series(prefix: str, name: str, arr: Optional[np.ndarray], keys: Sequence[str]) -> None:
                    if arr is None or not self._has_conn(keys):
                        return
                    y = arr[:n_rows]
                    ok = np.isfinite(y)
                    if not ok.any():
                        return
                    s_uuid = self._suid(prefix, src)
                    self._emitted[s_uuid] = name
                    self._send_sample(
                        {"uuid": s_uuid, "name": name, "x": x[ok].tolist(), "y": y[ok].tolist(), "action": "select"},
                        keys,
                    )

                if d.mode == "both":
                    series("Coverage", f"Coverage_{src}", d.surface, ("Result",))
                    series("Speed", f"Speed_{src}", d.result, ("Result",))
                else:
                    primary_prefix = "Speed" if d.metric_kind == "speed" else "Coverage"
                    series(primary_prefix, f"{primary_prefix}_{src}", d.result, ("Result",))
                series("Count", f"Count_{src}", d.count, ("Result", "Count"))
                series("Motility", f"Motility_{src}", d.motility, ("Result", "Motility"))

                if emit_pi and self._has_conn(pi_keys):
                    pi_arr = self._compute_phototaxis_series(d, n_rows)
                    series("Phototaxis", f"Phototaxis_{src}", pi_arr, pi_keys)

                if emit_drift and self._drift_map is None and d.drift_speed is not None:
                    series("DriftSpeed", f"DriftSpeed_{src}", d.drift_speed, ("Result",))
                    if d.drift_angle_deg is not None:
                        series("DriftAngle_deg", f"DriftAngle_deg_{src}", d.drift_angle_deg, ("Result",))

        if emit_drift and self._drift_map is not None:
            has_res = self._has_conn(("Result",))
            has_pi = emit_pi and (not emit_manip) and self._has_conn(pi_keys)
            if has_res or has_pi:
                sorted_keys = self._drift_map.get("sorted_keys") or []
                frames_dict = self._drift_map.get("frames") or {}
                upto_keys = (
                    sorted_keys
                    if max_frame_idx is None
                    else [fk for fk in sorted_keys if fk <= max_frame_idx]
                )
                if upto_keys:
                    fps_val = float((d.fps if d is not None else self._video_fps) or 30.0)
                    x_drift = self._build_x_from_indices(np.asarray(upto_keys, dtype=np.float64), fps_val).tolist()
                    y_spd = [float(frames_dict[fk].get("mean_mag_um_s", 0.0)) for fk in upto_keys]
                    y_ang = [float(frames_dict[fk].get("mean_ang_deg", 0.0)) for fk in upto_keys]
                    if has_res:
                        s_uuid_spd = self._suid("DriftSpeed", src)
                        self._emitted[s_uuid_spd] = f"DriftSpeed_{src}"
                        self._send_sample(
                            {"uuid": s_uuid_spd, "name": f"DriftSpeed_{src}", "x": x_drift, "y": y_spd, "action": "select"},
                            ("Result",),
                        )
                        s_uuid_ang = self._suid("DriftAngle_deg", src)
                        self._emitted[s_uuid_ang] = f"DriftAngle_deg_{src}"
                        self._send_sample(
                            {"uuid": s_uuid_ang, "name": f"DriftAngle_deg_{src}", "x": x_drift, "y": y_ang, "action": "select"},
                            ("Result",),
                        )
                    if has_pi:
                        pi_drift = self._compute_drift_pi_array(
                            np.asarray(y_ang, dtype=np.float64),
                            np.asarray(y_spd, dtype=np.float64),
                        )
                        ok = np.isfinite(pi_drift)
                        if ok.any():
                            x_arr = np.asarray(x_drift, dtype=np.float64)
                            s_uuid_pi = self._suid("Phototaxis", src)
                            self._emitted[s_uuid_pi] = f"Phototaxis_{src}"
                            self._send_sample(
                                {
                                    "uuid": s_uuid_pi,
                                    "name": f"Phototaxis_{src}",
                                    "x": x_arr[ok].tolist(),
                                    "y": pi_drift[ok].tolist(),
                                    "action": "select",
                                },
                                pi_keys,
                            )
        elif (not emit_manip) and emit_drift and d is not None and d.drift_speed is not None:
            n_rows = len(d.frame_idx) if k is None else k
            if n_rows > 0:
                x = self._build_x(d, n_rows)

                def drift_series_only(
                    prefix: str,
                    name: str,
                    arr: Optional[np.ndarray],
                    keys: Sequence[str] = ("Result",),
                ) -> None:
                    if arr is None or not self._has_conn(keys):
                        return
                    y = arr[:n_rows]
                    ok = np.isfinite(y)
                    if not ok.any():
                        return
                    s_uuid = self._suid(prefix, src)
                    self._emitted[s_uuid] = name
                    self._send_sample(
                        {"uuid": s_uuid, "name": name, "x": x[ok].tolist(), "y": y[ok].tolist(), "action": "select"},
                        keys,
                    )

                drift_series_only("DriftSpeed", f"DriftSpeed_{src}", d.drift_speed)
                if d.drift_angle_deg is not None:
                    drift_series_only("DriftAngle_deg", f"DriftAngle_deg_{src}", d.drift_angle_deg)
                if emit_pi and d.drift_angle_deg is not None and self._has_conn(pi_keys):
                    ang_slice = d.drift_angle_deg[:n_rows]
                    spd_slice = d.drift_speed[:n_rows] if d.drift_speed is not None else np.ones(n_rows)
                    pi_drift = self._compute_drift_pi_array(ang_slice, spd_slice)
                    drift_series_only("Phototaxis", f"Phototaxis_{src}", pi_drift, pi_keys)

    def _emit(self, idx: int) -> None:
        d = self._data
        row = d.row(idx) if d is not None else None
        self._update_texts(d, row, idx)

        src = self._source or (d.source if d is not None else "") or self.label
        mode = self.view_mode.lower()
        emit_manip = (mode in ("combined", "manip")) and (d is not None)
        emit_drift = (mode in ("combined", "drift")) and self._has_drift and self.emit_drift_series

        k = d.upto(idx) if d is not None else 0
        self._emit_time_series(d, k, idx, src, emit_manip, emit_drift)

        if emit_manip and d is not None and row is not None:
            if self.connections.get("Vectors"):
                ang, mag = d.vectors(row)
                if ang.size:
                    motility = d.motility[row] if d.motility is not None else np.nan
                    payload = {
                        "angles": ang,
                        "magnitudes": mag,
                        "mode": d.mode,
                        "count": int(d.count[row]) if np.isfinite(d.count[row]) else int(ang.size),
                        "motility_pct": float(motility) if np.isfinite(motility) else None,
                        "source": src,
                        "timestamp": time.time(),
                        "frame_idx": idx,
                    }
                    for module in list(self.connections["Vectors"]):
                        try:
                            module.input_cb(data=payload, data_type=IOTypes.VECTORS)
                        except Exception as e:
                            logger.error(f"MotilityReprocessor vectors dispatch error: {e}")
            if self.connections.get("Raw Speeds"):
                speeds = d.raw_speeds(row)
                if speeds.size:
                    sample = {
                        "uuid": self._suid("RawSpeed", src),
                        "name": f"RawSpeed_{src}",
                        "x": [],
                        "y": speeds.tolist(),
                        "action": "select",
                    }
                    for module in list(self.connections["Raw Speeds"]):
                        try:
                            module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)
                        except Exception as e:
                            logger.error(f"MotilityReprocessor raw speeds dispatch error: {e}")
        elif (not emit_manip) and emit_drift and self.connections.get("Vectors"):
            drift_info = self._get_drift_info(idx)
            if drift_info is not None and drift_info.get("enabled", False):
                cells = drift_info.get("cells", {})
                angles: List[float] = []
                mags: List[float] = []
                for cdata in cells.values():
                    if cdata.get("valid", False):
                        angles.append(float(cdata.get("ang_rad", 0.0)))
                        mags.append(float(cdata.get("speed_um_s", 0.0)))
                if not angles:
                    d_spd = float(drift_info.get("mean_mag_um_s", 0.0))
                    d_ang_deg = float(drift_info.get("mean_ang_deg", 0.0))
                    angles = [float(np.deg2rad(d_ang_deg % 360.0))]
                    mags = [d_spd]
                payload = {
                    "angles": angles,
                    "magnitudes": mags,
                    "mode": "drift",
                    "count": len(angles),
                    "motility_pct": 0.0,
                    "source": f"Drift_{src}",
                    "timestamp": time.time(),
                    "frame_idx": idx,
                }
                for module in list(self.connections["Vectors"]):
                    try:
                        module.input_cb(data=payload, data_type=IOTypes.VECTORS)
                    except Exception as e:
                        logger.error(f"MotilityReprocessor drift vectors dispatch error: {e}")

        cmd_modules = self.connections.get("CMD", [])
        if cmd_modules:
            mean_spd = 0.0
            std_spd = 0.0
            non_mot_pct = 0.0
            n_ind = 100

            if emit_manip and d is not None and row is not None:
                if d.vec_speed_mean is not None and np.isfinite(d.vec_speed_mean[row]):
                    mean_spd = float(d.vec_speed_mean[row])
                elif d.result is not None and np.isfinite(d.result[row]):
                    mean_spd = float(d.result[row])
                if d.vec_speed_std is not None and np.isfinite(d.vec_speed_std[row]):
                    std_spd = float(d.vec_speed_std[row])
                if d.motility is not None and np.isfinite(d.motility[row]):
                    mot_pct = float(d.motility[row])
                    non_mot_pct = max(0.0, min(100.0, 100.0 - mot_pct))
                if d.count is not None and np.isfinite(d.count[row]):
                    n_ind = max(1, int(d.count[row]))
            elif (not emit_manip) and emit_drift:
                non_mot_pct = 100.0

            d_on = False
            d_spd = 0.0
            sim_drift_angle = 270.0

            if emit_drift and self._has_drift:
                drift_info = self._get_drift_info(idx)
                if drift_info is not None and drift_info.get("enabled", False):
                    mag_um = float(drift_info.get("mean_mag_um_s", 0.0))
                    if mag_um > 0.0:
                        d_on = True
                        d_spd = mag_um
                        d_screen_ang = float(drift_info.get("mean_ang_deg", 90.0))
                        sim_drift_angle = (360.0 - d_screen_ang) % 360.0
                    if not emit_manip and drift_info.get("drift_count", 0) > 0:
                        n_ind = int(drift_info["drift_count"])

            fps_val = float((d.fps if d is not None else self._video_fps) or 30.0)
            calib_val = float((d.meta.get("calibration") if d is not None and isinstance(d.meta, dict) else None) or 1.0)
            if (calib_val <= 0.0 or calib_val == 1.0) and emit_drift and self._has_drift:
                drift_info_cal = self._get_drift_info(idx)
                if drift_info_cal:
                    m_px = float(drift_info_cal.get("mean_mag_px", 0.0))
                    m_um = float(drift_info_cal.get("mean_mag_um_s", 0.0))
                    if m_px > 1e-6 and m_um > 0 and fps_val > 0:
                        calib_val = m_um / (m_px * fps_val)

            cmd_payload = {
                "speed_mean": round(float(mean_spd), 2),
                "speed_std": round(float(std_spd), 2),
                "non_motile_pct": round(float(non_mot_pct), 1),
                "drift_on": bool(d_on),
                "drift_speed": round(float(d_spd), 2),
                "drift_angle": round(float(sim_drift_angle), 1),
                "calibration": round(float(calib_val), 4),
                "fps": round(float(fps_val), 2),
                "n_individuals": n_ind,
            }
            for module in cmd_modules:
                try:
                    module.input_cb(data=cmd_payload, data_type=IOTypes.CMD_DICT)
                except Exception as e:
                    logger.error(f"MotilityReprocessor CMD dispatch error: {e}")

    def send_all(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Emit complete datasets for the entire video at once."""
        self._sync_params()
        with self._lock:
            d = self._data
            drift_map = self._drift_map
            has_drift = self._has_drift
            if d is None and drift_map is None:
                logger.warning("MotilityReprocessor: No data loaded to send.")
                return
            self._is_full_emitted = True

            src = self._source or (d.source if d is not None else "") or self.label
            mode = self.view_mode.lower()
            emit_manip = (mode in ("combined", "manip")) and (d is not None)
            emit_drift = (mode in ("combined", "drift")) and has_drift and self.emit_drift_series

            total_frames = (
                len(d.frame_idx)
                if d is not None
                else (len(drift_map.get("sorted_keys", [])) if drift_map else 0)
            )

            # 1. Emit full time series samples
            self._emit_time_series(d, None, None, src, emit_manip, emit_drift)

            # 2. Emit full vectors batch
            if self.connections.get("Vectors"):
                if emit_manip and d is not None:
                    n = len(d.frame_idx)
                    fps_val = float(d.fps or 30.0)
                    dt_val = float(1.0 / max(0.1, fps_val))
                    batch_frames: List[Dict[str, Any]] = []
                    pooled_angles: List[np.ndarray] = []
                    pooled_mags: List[np.ndarray] = []
                    for i in range(n):
                        f_idx = int(d.frame_idx[i])
                        t_val = (
                            float(d.time_s[i])
                            if (d.time_s is not None and i < len(d.time_s))
                            else float(f_idx * dt_val)
                        )
                        ang, mag = d.vectors(i)
                        if ang.size > 0:
                            pooled_angles.append(ang)
                            pooled_mags.append(mag)
                        batch_frames.append({
                            "angles": ang,
                            "magnitudes": mag,
                            "frame_idx": f_idx,
                            "timestamp": t_val,
                        })

                    all_ang = np.concatenate(pooled_angles) if pooled_angles else np.array([], dtype=np.float32)
                    all_mag = np.concatenate(pooled_mags) if pooled_mags else np.array([], dtype=np.float32)
                    mean_mot = (
                        float(np.nanmean(d.motility))
                        if (d.motility is not None and np.any(np.isfinite(d.motility)))
                        else None
                    )
                    mean_cnt = (
                        int(np.nanmean(d.count))
                        if (d.count is not None and np.any(np.isfinite(d.count)))
                        else len(all_ang)
                    )

                    payload = {
                        "action": "batch_vectors",
                        "frames": batch_frames,
                        "angles": all_ang,
                        "magnitudes": all_mag,
                        "mode": d.mode,
                        "count": mean_cnt,
                        "motility_pct": mean_mot,
                        "source": src,
                        "fps": fps_val,
                        "delta_t": dt_val,
                        "total_frames": n,
                    }
                    for module in list(self.connections["Vectors"]):
                        try:
                            module.input_cb(data=payload, data_type=IOTypes.VECTORS)
                        except Exception as e:
                            logger.error(f"MotilityReprocessor batch vectors dispatch error: {e}")

                elif (not emit_manip) and emit_drift:
                    batch_frames = []
                    pooled_angles = []
                    pooled_mags = []
                    fps_val = float((d.fps if d is not None else self._video_fps) or 30.0)
                    dt_val = float(1.0 / max(0.1, fps_val))

                    if drift_map is not None:
                        sorted_keys = drift_map.get("sorted_keys") or []
                        frames_dict = drift_map.get("frames") or {}
                        for fk in sorted_keys:
                            drift_entry = frames_dict.get(fk, {})
                            d_spd = float(drift_entry.get("mean_mag_um_s", 0.0))
                            d_ang_deg = float(drift_entry.get("mean_ang_deg", 0.0))
                            d_ang_rad = float(np.deg2rad(d_ang_deg % 360.0))
                            ang_arr = np.array([d_ang_rad], dtype=np.float32)
                            mag_arr = np.array([d_spd], dtype=np.float32)
                            pooled_angles.append(ang_arr)
                            pooled_mags.append(mag_arr)
                            batch_frames.append({
                                "angles": ang_arr,
                                "magnitudes": mag_arr,
                                "frame_idx": int(fk),
                                "timestamp": float(fk * dt_val),
                            })
                    elif d is not None and d.drift_speed is not None:
                        for i in range(len(d.frame_idx)):
                            f_idx = int(d.frame_idx[i])
                            t_val = (
                                float(d.time_s[i])
                                if (d.time_s is not None and i < len(d.time_s))
                                else float(f_idx * dt_val)
                            )
                            spd = float(d.drift_speed[i]) if np.isfinite(d.drift_speed[i]) else 0.0
                            ang_deg = (
                                float(d.drift_angle_deg[i])
                                if (d.drift_angle_deg is not None and np.isfinite(d.drift_angle_deg[i]))
                                else 0.0
                            )
                            ang_rad = float(np.deg2rad(ang_deg % 360.0))
                            ang_arr = np.array([ang_rad], dtype=np.float32)
                            mag_arr = np.array([spd], dtype=np.float32)
                            pooled_angles.append(ang_arr)
                            pooled_mags.append(mag_arr)
                            batch_frames.append({
                                "angles": ang_arr,
                                "magnitudes": mag_arr,
                                "frame_idx": f_idx,
                                "timestamp": t_val,
                            })

                    if batch_frames:
                        all_ang = np.concatenate(pooled_angles) if pooled_angles else np.array([], dtype=np.float32)
                        all_mag = np.concatenate(pooled_mags) if pooled_mags else np.array([], dtype=np.float32)
                        payload = {
                            "action": "batch_vectors",
                            "frames": batch_frames,
                            "angles": all_ang,
                            "magnitudes": all_mag,
                            "mode": "drift",
                            "count": len(all_ang),
                            "motility_pct": 0.0,
                            "source": f"Drift_{src}",
                            "fps": fps_val,
                            "delta_t": dt_val,
                            "total_frames": len(batch_frames),
                        }
                        for module in list(self.connections["Vectors"]):
                            try:
                                module.input_cb(data=payload, data_type=IOTypes.VECTORS)
                            except Exception as e:
                                logger.error(f"MotilityReprocessor batch drift vectors dispatch error: {e}")

            # 3. Emit pooled raw speeds
            if self.connections.get("Raw Speeds") and emit_manip and d is not None and d.speed_counts is not None:
                total_counts = np.sum(d.speed_counts, axis=0).astype(np.int64)
                centers = (d.speed_edges[:-1] + d.speed_edges[1:]) / 2.0
                pooled_speeds = np.repeat(centers, total_counts)
                if pooled_speeds.size > 20000:
                    pooled_speeds = pooled_speeds[:: pooled_speeds.size // 20000]
                sample = {
                    "uuid": self._suid("RawSpeed", src),
                    "name": f"RawSpeed_{src}",
                    "x": [],
                    "y": pooled_speeds.tolist(),
                    "action": "select",
                }
                for module in list(self.connections["Raw Speeds"]):
                    try:
                        module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)
                    except Exception as e:
                        logger.error(f"MotilityReprocessor raw speeds dispatch error: {e}")

            last_idx = int(d.frame_idx[-1]) if (d is not None and len(d.frame_idx) > 0) else (self._total or 0)
            last_row = (len(d.frame_idx) - 1) if (d is not None and len(d.frame_idx) > 0) else None
            self._update_texts(d, last_row, last_idx)
            logger.success(f"MotilityReprocessor: Sent all video data ({total_frames} frames) to downstream modules.")

    def _format_pi_value(self, pi_val: float) -> str:
        if not np.isfinite(pi_val):
            return "--"
        t_arrow, a_arrow = pi_light_arrows(self.pi_light_pos)
        if pi_val > 0.05:
            return f"{pi_val:+.2f} (Toward {t_arrow})"
        if pi_val < -0.05:
            return f"{pi_val:+.2f} (Away {a_arrow})"
        return f"{pi_val:+.2f} (Neutral)"

    def _update_texts(self, d: Optional[RawData], row: Optional[int], idx: int) -> None:
        def put(tag: str, text: str) -> None:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, text)

        put(self.frame_tag, f"{idx} / {self._total}" if self._total else str(idx))

        mode = self.view_mode.lower()
        show_manip = mode in ("combined", "manip")
        show_drift = mode in ("combined", "drift")

        drift_info = self._get_drift_info(idx) if (self._has_drift and show_drift) else None
        if drift_info is not None and drift_info.get("enabled", False):
            d_spd = float(drift_info.get("mean_mag_um_s", 0.0))
            d_ang = float(drift_info.get("mean_ang_deg", 0.0))
            gr, gc = drift_info.get("grid_shape", (1, 1))
            grid_txt = f" [{gr}x{gc}]" if (gr > 1 or gc > 1) else ""
            put(self.drift_text_tag, f"{d_spd:.2f} µm/s @ {d_ang:.1f}°{grid_txt}")
        else:
            put(self.drift_text_tag, "--")

        if not show_manip or d is None or row is None:
            if show_drift and drift_info is not None:
                if dpg.does_item_exist(self.metric_label_tag):
                    dpg.set_value(self.metric_label_tag, "Drift Speed (µm/s):")
                d_spd = float(drift_info.get("mean_mag_um_s", 0.0))
                d_ang = float(drift_info.get("mean_ang_deg", 0.0))
                put(self.metric_tag, f"{d_spd:.3f}")
                pi_arr = self._compute_drift_pi_array(np.array([d_ang]), np.array([d_spd]))
                put(self.pi_text_tag, self._format_pi_value(float(pi_arr[0])))
            else:
                put(self.metric_tag, "--")
                put(self.pi_text_tag, "--")
            for tag in (self.count_tag, self.motility_tag, self.speed_stats_tag, self.vector_text_tag):
                put(tag, "--")
            return

        if dpg.does_item_exist(self.metric_label_tag):
            dpg.set_value(self.metric_label_tag, _METRIC_LABELS.get(d.metric_kind, "Value:"))

        def f(arr: Optional[np.ndarray]) -> float:
            return float(arr[row]) if arr is not None and np.isfinite(arr[row]) else float("nan")

        res, surf = f(d.result), f(d.surface)
        if d.metric_kind == "both":
            put(self.metric_tag, f"{surf:.1f}% | {res:.3f}")
        elif d.metric_kind == "speed":
            put(self.metric_tag, f"{res:.3f}")
        else:
            put(self.metric_tag, f"{res:.2f} %")
        cnt, mot = f(d.count), f(d.motility)
        put(self.count_tag, f"{int(cnt):,}" if np.isfinite(cnt) else "--")
        put(self.motility_tag, f"{mot:.1f}%" if np.isfinite(mot) else "--")
        m, s = f(d.vec_speed_mean), f(d.vec_speed_std)
        put(self.speed_stats_tag, f"{m:.2f} ± {s:.2f} µm/s (n={int(d.n_vec[row])})" if np.isfinite(m) else "--")
        vx, vy = f(d.vx_mean), f(d.vy_mean)
        if np.isfinite(vx) and np.isfinite(vy):
            put(self.vector_text_tag, f"{np.hypot(vx, vy):.2f} µm/s @ {np.degrees(np.arctan2(-vy, vx)) % 360:.0f}°")
        else:
            put(self.vector_text_tag, "--")

        pi_val = self._compute_single_frame_pi(d, row)
        put(self.pi_text_tag, self._format_pi_value(pi_val))

    def _draw_vector(self, frame: np.ndarray, d: RawData, row: int) -> np.ndarray:
        """Draw the population's mean motion vector at the image centre."""
        vx, vy = float(d.vx_mean[row]), float(d.vy_mean[row])
        mag = float(np.hypot(vx, vy))
        if not np.isfinite(mag) or mag <= 0:
            return frame
        out = frame.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        h, w = out.shape[:2]
        length = min(1.5, mag / (d.vector_ref or mag)) * 0.35 * min(h, w) * float(self.vector_scale)
        cx, cy = w // 2, h // 2
        end = (int(cx + vx / mag * length), int(cy + vy / mag * length))
        cv2.arrowedLine(out, (cx, cy), end, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        return out

    def is_ready(self) -> bool:
        """Check if module is ready to process or re-emit data."""
        return True

    def on_close(self) -> None:
        """Cleanup on window close."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        with self._lock:
            self._data = None
            self._last_raw_frame = None
            self._drift_info_cache_val = None
            self._cached_x_full = None
            self._cached_pi_full = None
        super().on_close()

    def close(self) -> None:
        """Explicit close handler."""
        self.on_close()
        super().close()

    def destroy(self) -> None:
        """Complete resource deallocation."""
        self.on_close()


MotilityReprocessorWin = MotilityReprocessor_win
EXPORTED_CLASS = MotilityReprocessor_win
EXPORTED_NAME = "Motility Reprocessor"
