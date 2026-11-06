from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid as _uuid
from bisect import bisect_left
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.file_explorer import file_explorer
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from core.app_state import app_state

from .simulator_engine import SimulationEngine, save_ground_truth, save_params
from .simulator_params import (
    BG_MODES, EXPORT_FORMATS, PARAMS, RESTART_KEYS, SECTIONS, SPRITE_MODES, Param,
)

_IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
ANGLE_KEYS = {"intention_angle", "photo_angle", "drift_angle"}
OVERLAY_VIEW_MODES = ["Blend (Overlay)", "Input Frame", "Simulation"]
_PARAM_BY_KEY: Dict[str, Param] = {p.key: p for p in PARAMS}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


class MotilitySimulator_win(WindowBase):
    """Synthetic motility video generator with perfectly known ground truth.

    Streams frames live to downstream modules (e.g. the Motilitometer Tracker) and/or exports a
    video together with ``*_gt.csv`` (per-frame ground truth) and ``*_params.json``.
    Supports presets saving/loading identical to Counter Tracker, with joystick-style hold & drag angle buttons.
    """

    label = "Motility Simulator"
    description = "Synthetic motility video generator with ground truth for Motilitometer Tracker validation."
    win_width = display_scaling.scale(430)
    win_height = display_scaling.scale(780)

    accepted_input_types = [IOTypes.CMD_DICT, IOTypes.FRAME, IOTypes.TRIGGER]
    outputs = {
        "Frame": IOTypes.FRAME,
        "CMD": IOTypes.CMD_DICT,
        "Tracking": IOTypes.TRACKING,
        "Contours": IOTypes.CONTOURS,
        "Result": IOTypes.SAMPLE,
        "Count": IOTypes.SAMPLE,
        "Motility": IOTypes.SAMPLE,
        "Raw Speeds": IOTypes.SAMPLE,
        "Vectors": IOTypes.VECTORS,
        "Trigger": IOTypes.TRIGGER,
        "File Path": IOTypes.FILE_PATH,
    }
    persistent_fields = [
        "label", "export_dir", "export_name", "export_format",
        "update_with_input", "overlay_view_mode", "x_axis_mode", "delta_t",
    ] + [p.key for p in PARAMS]

    export_dir: str = os.path.join(os.path.expanduser("~"), "MotilitySimulator")
    export_name: str = "synthetic"
    export_format: str = EXPORT_FORMATS[0]
    update_with_input: bool = True
    overlay_view_mode: str = OVERLAY_VIEW_MODES[0]
    x_axis_mode: str = "Fixed ΔT"
    delta_t: float = 1.0 / 30.0

    # parameters whose widget visibility depends on another parameter
    _VISIBLE_WHEN: Dict[str, Callable[["MotilitySimulator_win"], bool]] = {
        "bg_color": lambda s: s.bg_mode == BG_MODES[0],
        "bg_image": lambda s: s.bg_mode == BG_MODES[1],
        "color": lambda s: s.sprite_mode in (SPRITE_MODES[0], SPRITE_MODES[2]),
        "tex_noise": lambda s: s.sprite_mode == SPRITE_MODES[0],
        "tex_grain": lambda s: s.sprite_mode == SPRITE_MODES[0],
        "sprite_png": lambda s: s.sprite_mode == SPRITE_MODES[1],
        "intention_angle": lambda s: str(s.intention_mode).startswith("Common"),
        "intention_spread": lambda s: str(s.intention_mode).startswith("Common"),
        "adhesion_prob": lambda s: bool(s.adhesion),
        "helix_period": lambda s: bool(s.helical),
        "helix_amp": lambda s: bool(s.helical),
        "chirality": lambda s: bool(s.helical),
        "drift_angle": lambda s: bool(s.drift_on),
        "drift_speed": lambda s: bool(s.drift_on),
        "drift_motile_pct": lambda s: bool(s.drift_on),
        "oof_blur": lambda s: float(s.oof_pct or 0) > 0,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        def _p(attr: str, default: Any) -> Any:
            val = getattr(self, attr, None)
            if val is None:
                val = kwargs.get(attr, default)
            return default if val is None else val

        # Defensive extraction of persistent fields
        self.export_dir = str(_p("export_dir", "") or os.path.join(os.path.expanduser("~"), "MotilitySimulator"))
        self.export_name = str(_p("export_name", "synthetic") or "synthetic")
        self.export_format = str(_p("export_format", EXPORT_FORMATS[0]) or EXPORT_FORMATS[0])
        self.update_with_input = bool(_p("update_with_input", True))
        self.overlay_view_mode = str(_p("overlay_view_mode", OVERLAY_VIEW_MODES[0]) or OVERLAY_VIEW_MODES[0])
        self.x_axis_mode = str(_p("x_axis_mode", "Fixed ΔT") or "Fixed ΔT")
        try:
            self.delta_t = float(_p("delta_t", 1.0 / 30.0))
        except (ValueError, TypeError):
            self.delta_t = 1.0 / 30.0

        for p in PARAMS:
            val = getattr(self, p.key, kwargs.get(p.key, p.default))
            if val is None:
                val = p.default
            try:
                if p.kind == "int":
                    val = int(val)
                elif p.kind == "float":
                    val = float(val)
                elif p.kind == "bool":
                    val = bool(val)
                elif p.kind in ("str", "combo", "path"):
                    val = str(val)
            except (ValueError, TypeError):
                val = p.default
            setattr(self, p.key, val)
        U = self.UUID
        self.winID = f"motility_simulator_win_{U}"
        self._tag = lambda key: f"ms_{key}_{U}"
        self._gtag = lambda key: f"msg_{key}_{U}"
        self._joy_btn_tag = lambda key: f"ms_joy_btn_{key}_{U}"
        self._joy_ihr_tag = lambda key: f"ms_joy_ihr_{key}_{U}"
        self._joy_drag: Dict[str, Dict[str, Any]] = {}

        self.status_tag = f"ms_status_{U}"
        self.metrics_tag = f"ms_metrics_{U}"
        self.progress_tag = f"ms_progress_{U}"
        self.pause_btn_tag = f"ms_pause_{U}"
        self.export_btn_tag = f"ms_export_{U}"
        self.export_dir_tag = f"ms_export_dir_{U}"
        self.export_name_tag = f"ms_export_name_{U}"
        self.export_fmt_tag = f"ms_export_fmt_{U}"
        self.preset_menu_tag = f"ms_preset_menu_{U}"
        self.preset_menu_list_tag = f"ms_preset_list_{U}"
        self.update_with_input_tag = f"ms_update_with_input_{U}"
        self.overlay_view_mode_tag = f"ms_overlay_view_mode_{U}"

        self._last_input_frame: Optional[np.ndarray] = None
        self._last_input_idx: Optional[int] = None
        self._source_name: str = ""
        self.samples_buffers: Dict[str, Dict[str, Any]] = {}
        self._suid_cache: Dict[Tuple[str, str], str] = {}
        self._buffers_lock = threading.Lock()
        self._tracking_history: Dict[int, List[Tuple[float, float]]] = {}

        self._engine: Optional[SimulationEngine] = None
        self._live_thread: Optional[threading.Thread] = None
        self._export_thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._export_cancel = False
        self._pending_restart = False
        self._gen = 0
        self._is_closing = False
        self._structural_dirty = False
        self._cmd_lock = threading.Lock()

        # Presets discovery
        self._presets: Dict[str, str] = self._scan_presets()

        self._build_ui()
        self._refresh_visibility()
        self.update_permission()

    # ── Presets Management ────────────────────────────────────────────────
    def _scan_presets(self) -> Dict[str, str]:
        """Scans module directory and presets/ subfolder for .json configuration files."""
        module_dir = os.path.dirname(os.path.abspath(__file__))
        presets: Dict[str, str] = {}
        for folder, prefix in ((module_dir, ""), (os.path.join(module_dir, "presets"), "presets/")):
            try:
                if os.path.exists(folder):
                    for entry in sorted(os.scandir(folder), key=lambda e: e.name.lower()):
                        if entry.is_file() and entry.name.lower().endswith(".json"):
                            presets[f"{prefix}{entry.name}"] = os.path.abspath(entry.path)
            except Exception:
                pass
        return presets

    def _populate_preset_menu(self, parent: Any = None) -> None:
        """Populates the Presets submenu with detected preset items."""
        for name, path in self._presets.items():
            kw: Dict[str, Any] = {"parent": parent} if parent else {}
            dpg.add_menu_item(label=name, user_data=path, callback=self._on_preset_menu_item_clicked, **kw)

    def _refresh_presets_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, select_path: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        """Refreshes the presets list by rescanning the module folder."""
        self._presets = self._scan_presets()
        if select_path and os.path.exists(select_path):
            norm = lambda p: os.path.normcase(os.path.normpath(p))
            if not any(norm(p) == norm(select_path) for p in self._presets.values()):
                self._presets[os.path.basename(select_path)] = select_path

        if dpg.does_item_exist(self.preset_menu_list_tag):
            dpg.delete_item(self.preset_menu_list_tag, children_only=True)
            self._populate_preset_menu(parent=self.preset_menu_list_tag)

    def _on_preset_menu_item_clicked(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Called when a preset is selected from the Presets menu item list."""
        if user_data:
            self._load_config_file(str(user_data), "Applied preset from")

    def _run_file_dialog(self, dialog: str, on_path: Callable[[str], None], **opts: Any) -> None:
        """Runs a tkinter file dialog (`asksaveasfilename`/`askopenfilename`) in a thread, then calls on_path(path)."""
        def run_dialog() -> None:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                module_dir = os.path.dirname(os.path.abspath(__file__))
                presets_dir = os.path.join(module_dir, "presets")
                os.makedirs(presets_dir, exist_ok=True)
                opts.setdefault("initialdir", presets_dir)
                if dialog == "asksaveasfilename":
                    path = filedialog.asksaveasfilename(**opts)
                else:
                    path = filedialog.askopenfilename(**opts)
                root.destroy()
                if path:
                    on_path(path)
            except Exception as e:
                logger.error(f"MotilitySimulator: File dialog error: {e}")

        threading.Thread(target=run_dialog, daemon=True).start()

    def _export_settings_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Opens a file dialog to export current module settings to a JSON file."""
        def save(file_path: str) -> None:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self._get_current_params(), f, indent=4)
                logger.info(f"MotilitySimulator: Settings successfully exported to {file_path}")
                self._status(f"Exported config to {os.path.basename(file_path)}")
                self._refresh_presets_cb(select_path=file_path)
            except Exception as e:
                logger.error(f"MotilitySimulator: Export config error: {e}")
                self._status(f"Export error: {e}", (255, 90, 90))

        self._run_file_dialog(
            "asksaveasfilename",
            save,
            title="Export Motility Simulator Configuration",
            initialfile="config_motility_simulator.json",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )

    def _import_settings_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Opens a file dialog to import module settings from any JSON file."""
        self._run_file_dialog(
            "askopenfilename",
            lambda p: self._load_config_file(p, "Settings successfully imported from"),
            title="Import Motility Simulator Configuration",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )

    def _get_current_params(self) -> Dict[str, Any]:
        """Collects the current configuration parameters."""
        return self._cfg()

    def _load_config_file(self, file_path: str, success_msg: str = "Applied preset from") -> None:
        """Loads a JSON config file, applies it to the widgets and refreshes the preset menu."""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._apply_imported_params(data)
                self._refresh_presets_cb(select_path=file_path)
                logger.info(f"MotilitySimulator: {success_msg} {file_path}")
                self._status(f"{success_msg} {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"MotilitySimulator: Error applying config from {file_path}: {e}")
                self._status(f"Error loading config: {e}", (255, 90, 90))

    def _apply_imported_params(self, p: Dict[str, Any]) -> None:
        """Applies loaded dictionary parameters to UI widgets and updates the simulation."""
        if not isinstance(p, dict):
            return

        flat: Dict[str, Any] = {}
        for k, v in p.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[sub_k] = sub_v
            else:
                flat[k] = v

        for k, v in flat.items():
            param = _PARAM_BY_KEY.get(k)
            if param is not None:
                if param.kind == "color" and isinstance(v, (list, tuple)):
                    v = [int(round(float(c))) for c in v[:3]]
                elif param.kind == "int":
                    v = int(v)
                elif param.kind == "float":
                    v = float(v)
                elif param.kind == "bool":
                    v = bool(v)
                setattr(self, k, v)
                self._write_widget(param, v)

        if "update_with_input" in flat:
            self.update_with_input = bool(flat["update_with_input"])
            if dpg.does_item_exist(self.update_with_input_tag):
                dpg.set_value(self.update_with_input_tag, self.update_with_input)
        if "overlay_view_mode" in flat and flat["overlay_view_mode"] in OVERLAY_VIEW_MODES:
            self.overlay_view_mode = flat["overlay_view_mode"]
            if dpg.does_item_exist(self.overlay_view_mode_tag):
                dpg.set_value(self.overlay_view_mode_tag, self.overlay_view_mode)

        self._refresh_visibility()

        if self._running:
            with self._cmd_lock:
                self._pending_restart = True
            self._status("Preset applied live to simulation.", (100, 200, 100))
        else:
            self._status("Config applied. Ready to start.")

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        sc = display_scaling.scale
        with self.window(menubar=True):
            with dpg.menu_bar():
                with dpg.menu(label="Preset", tag=self.preset_menu_tag):
                    dpg.add_menu_item(label="Save Config...", callback=self._export_settings_cb)
                    dpg.add_menu_item(label="Load Config from File...", callback=self._import_settings_cb)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Scan / Refresh Presets", callback=self._refresh_presets_cb)
                    with dpg.menu(label="Presets", tag=self.preset_menu_list_tag):
                        self._populate_preset_menu()

            # Input stream controls
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Update with input",
                    default_value=self.update_with_input,
                    tag=self.update_with_input_tag,
                    callback=self._on_update_with_input_cb,
                )
                dpg.add_combo(
                    items=OVERLAY_VIEW_MODES,
                    default_value=self.overlay_view_mode,
                    label="View",
                    width=sc(140),
                    tag=self.overlay_view_mode_tag,
                    callback=self._on_overlay_view_mode_cb,
                )
            with dpg.tooltip(self.update_with_input_tag):
                dpg.add_text("When checked, incoming CMD parameters update the simulation.\nWhen unchecked, current parameters are held constant.")
            with dpg.tooltip(self.overlay_view_mode_tag):
                dpg.add_text("Switch between showing the blended overlay, the raw input frame, or pure simulation.")
            dpg.add_separator()

            # Transport controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", width=sc(70), callback=self.start_live)
                dpg.add_button(label="Pause", width=sc(70), tag=self.pause_btn_tag, callback=self.toggle_pause)
                dpg.add_button(label="Stop", width=sc(70), callback=self.stop_live)
                dpg.add_button(label="Preview", width=sc(70), callback=self.preview_frame)
                dpg.add_button(label="Reset", width=sc(70), callback=self.reset_metrics)

            dpg.add_text("Ready", tag=self.status_tag, color=(100, 200, 100), wrap=sc(400))
            dpg.add_text("Count: 0  |  Motility: --  |  Mean Speed: 0.0 µm/s", tag=self.metrics_tag, color=(180, 220, 255), wrap=sc(400))
            dpg.add_separator()

            for section in SECTIONS:
                with dpg.collapsing_header(label=section, default_open=section in (SECTIONS[0], SECTIONS[2])):
                    if section == "Photoshock":
                        dpg.add_button(
                            label="PHOTOSHOCK",
                            width=-1,
                            height=sc(32),
                            callback=self.photoshock,
                        )
                        dpg.add_spacer(height=sc(4))
                    for p in [q for q in PARAMS if q.section == section]:
                        with dpg.group(tag=self._gtag(p.key)):
                            self._add_param_widget(p)

            with dpg.collapsing_header(label="Export (video + ground truth)", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        tag=self.export_dir_tag,
                        default_value=self.export_dir,
                        width=sc(260),
                        callback=lambda s, v, *a: setattr(self, "export_dir", v),
                    )
                    dpg.add_button(label="Browse", callback=lambda *a: self._browse_export_dir())
                dpg.add_input_text(
                    label="File name",
                    tag=self.export_name_tag,
                    default_value=self.export_name,
                    width=sc(200),
                    callback=lambda s, v, *a: setattr(self, "export_name", v),
                )
                dpg.add_combo(
                    items=EXPORT_FORMATS,
                    default_value=self.export_format,
                    label="Format",
                    tag=self.export_fmt_tag,
                    width=sc(200),
                    callback=lambda s, v, *a: setattr(self, "export_format", v),
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Export video + GT",
                        tag=self.export_btn_tag,
                        width=sc(180),
                        callback=self.toggle_export,
                    )
                dpg.add_progress_bar(tag=self.progress_tag, default_value=0.0, width=-1)

        # Setup joystick event handlers outside window container to avoid stack issues
        self._setup_joy_handlers()

    def _setup_joy_handlers(self) -> None:
        """Create and bind item handler registries for joystick buttons outside the window container."""
        try:
            for key in ANGLE_KEYS:
                btn_tag = self._joy_btn_tag(key)
                ihr_tag = self._joy_ihr_tag(key)
                if not dpg.does_item_exist(btn_tag):
                    continue
                if dpg.does_item_exist(ihr_tag):
                    dpg.delete_item(ihr_tag)
                with dpg.item_handler_registry(tag=ihr_tag):
                    dpg.add_item_clicked_handler(
                        button=dpg.mvMouseButton_Left,
                        callback=self._on_joy_click,
                        user_data=key,
                    )
                    dpg.add_item_active_handler(
                        callback=self._on_joy_active,
                        user_data=key,
                    )
                    dpg.add_item_deactivated_handler(
                        callback=self._on_joy_release,
                        user_data=key,
                    )
                dpg.bind_item_handler_registry(btn_tag, ihr_tag)
        except Exception as e:
            logger.warning(f"MotilitySimulator: Could not setup joystick event handlers: {e}")

    def _add_param_widget(self, p: Param) -> None:
        sc = display_scaling.scale
        tag = self._tag(p.key)
        label = p.label + (" *" if p.key in RESTART_KEYS else "")
        val = getattr(self, p.key)
        kw = dict(tag=tag, callback=self._on_param, user_data=p.key)

        # Elements requiring an angle direction get a joystick button right before the input box
        if p.key in ANGLE_KEYS:
            btn_tag = self._joy_btn_tag(p.key)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    tag=btn_tag,
                    width=sc(65),
                )
                with dpg.tooltip(btn_tag):
                    dpg.add_text("Joystick")
                dpg.add_input_float(
                    label=label,
                    default_value=float(val),
                    min_value=float(p.vmin),
                    max_value=float(p.vmax),
                    min_clamped=True,
                    max_clamped=True,
                    step=0,
                    format=p.fmt,
                    width=sc(90),
                    **kw,
                )
        elif p.kind == "int":
            dpg.add_input_int(
                label=label,
                default_value=int(val),
                min_value=int(p.vmin),
                max_value=int(p.vmax),
                min_clamped=True,
                max_clamped=True,
                step=0,
                width=sc(150),
                **kw,
            )
        elif p.kind == "float":
            dpg.add_input_float(
                label=label,
                default_value=float(val),
                min_value=float(p.vmin),
                max_value=float(p.vmax),
                min_clamped=True,
                max_clamped=True,
                step=0,
                format=p.fmt,
                width=sc(150),
                **kw,
            )
        elif p.kind == "bool":
            dpg.add_checkbox(label=label, default_value=bool(val), **kw)
        elif p.kind == "combo":
            dpg.add_combo(items=list(p.items), default_value=val, label=label, width=sc(220), **kw)
        elif p.kind == "color":
            rgb = [int(v) for v in list(val)[:3]]
            dpg.add_color_edit(
                default_value=rgb + [255],
                label=label,
                no_alpha=True,
                no_inputs=False,
                width=sc(220),
                **kw,
            )
        elif p.kind == "path":
            with dpg.group(horizontal=True):
                dpg.add_input_text(default_value=str(val or ""), label="", width=sc(190), **kw)
                dpg.add_button(label="...", user_data=p.key, callback=lambda s, a, u, *args: self._browse_file(u))
                dpg.add_text(label)
        else:  # str
            dpg.add_input_text(label=label, default_value=str(val or ""), width=sc(150), **kw)
        if p.tip:
            with dpg.tooltip(tag):
                dpg.add_text(p.tip, wrap=sc(320))

    def _on_joy_click(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Start tracking mouse position when joystick button is clicked."""
        key = user_data
        if not key:
            return
        self._joy_drag[key] = {
            "active": True,
            "start_pos": dpg.get_mouse_pos(),
        }

    def _on_joy_active(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Update angle continuously as user drags mouse away from click origin (joystick)."""
        key = user_data
        drag = self._joy_drag.get(key)
        if not drag or not drag.get("active"):
            return
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            drag["active"] = False
            return

        start_pos = drag["start_pos"]
        cur_pos = dpg.get_mouse_pos()
        dx = cur_pos[0] - start_pos[0]
        dy = cur_pos[1] - start_pos[1]
        if math.hypot(dx, dy) < 8.0:
            return

        # Convention: 0=Right (dx>0), 90=Up (dy<0 -> -dy>0), 180=Left (dx<0), 270=Down (dy>0 -> -dy<0)
        rad = math.atan2(-dy, dx)
        deg = math.degrees(rad) % 360.0
        if deg < 0.0:
            deg += 360.0
        val = float(int(round(deg)) % 360)

        # Update input widget value
        tag = self._tag(key)
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)

        # Apply to parameter
        p = _PARAM_BY_KEY.get(key)
        if p is not None:
            self._set_param(p, val)

    def _on_joy_release(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Stop tracking mouse position when joystick button is released."""
        key = user_data
        if key in self._joy_drag:
            self._joy_drag[key]["active"] = False

    def _on_param(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        key = user_data
        p = _PARAM_BY_KEY[key]
        if p.kind == "color":
            vals = [float(v) for v in list(app_data)[:3]]
            if max(vals) <= 1.0 and any(v != int(v) for v in vals):
                vals = [v * 255.0 for v in vals]
            val: Any = [int(round(v)) for v in vals]
        elif p.kind == "int":
            val = int(app_data)
        elif p.kind == "float":
            val = float(app_data)
        elif p.kind == "bool":
            val = bool(app_data)
        else:
            val = app_data
        self._set_param(p, val)

    def _set_param(self, p: Param, val: Any) -> None:
        setattr(self, p.key, val)
        if p.key == "fps" and float(val) > 0:
            self.delta_t = 1.0 / float(val)
        eng = self._engine
        if p.key in RESTART_KEYS:
            if self._running:
                self._structural_dirty = True
                self._status("This parameter needs a restart: press Start to apply.", (255, 200, 90))
        elif eng is not None:
            eng.update_params({p.key: val})
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        for key, rule in self._VISIBLE_WHEN.items():
            g = self._gtag(key)
            if dpg.does_item_exist(g):
                try:
                    dpg.configure_item(g, show=bool(rule(self)))
                except Exception:
                    pass

    def _write_widget(self, p: Param, val: Any) -> None:
        tag = self._tag(p.key)
        if dpg.does_item_exist(tag):
            if p.kind == "color":
                dpg.set_value(tag, [int(v) for v in list(val)[:3]] + [255])
            else:
                dpg.set_value(tag, val)

    def _status(self, text: str, color: Any = (100, 200, 100)) -> None:
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, text)
            dpg.configure_item(self.status_tag, color=color)

    def _update_metrics(self, count: int, pct_mot: float, mean_speed: float) -> None:
        if dpg.does_item_exist(self.metrics_tag):
            dpg.set_value(
                self.metrics_tag,
                f"Count: {count}  |  Motility: {pct_mot:.1f}%  |  Mean Speed: {mean_speed:.1f} µm/s",
            )

    def _browse_file(self, key: str) -> None:
        cur = str(getattr(self, key) or "")

        def _cb(path: str, *_: Any) -> None:
            if path:
                p = _PARAM_BY_KEY.get(key)
                if p is not None:
                    self._write_widget(p, path)
                    self._set_param(p, path)

        file_explorer.select_file(default_path=cur, extensions=_IMAGE_TYPES, callback=_cb)

    def _browse_export_dir(self) -> None:
        def _cb(path: str, *_: Any) -> None:
            if path:
                self.export_dir = path
                dpg.set_value(self.export_dir_tag, path)

        file_explorer.select_folder(default_path=self.export_dir, callback=_cb)

    def update_permission(self) -> None:
        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def is_ready(self) -> bool:
        return not self._is_closing

    # ── configuration snapshot ────────────────────────────────────────────
    def _cfg(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in PARAMS:
            v = getattr(self, p.key)
            out[p.key] = list(v) if isinstance(v, (list, tuple)) else v
        return out

    # ── emission ──────────────────────────────────────────────────────────
    def _emit(self, output: str, data: Any, data_type: Any) -> None:
        for module in list(self.connections.get(output, [])):
            if hasattr(module, "input_cb") and callable(module.input_cb):
                try:
                    module.input_cb(data=data, data_type=data_type)
                except Exception as e:
                    logger.error(f"MotilitySimulator dispatch error ({output}) to {getattr(module, 'winID', module)}: {e}")

    def _emit_frame(self, frame: np.ndarray, idx: int, name: str = "synthetic") -> None:
        self._emit("Frame", (frame, name, idx), IOTypes.FRAME)

    def _suid(self, prefix: str, source: str) -> str:
        cache_key = (prefix, source)
        cached = self._suid_cache.get(cache_key)
        if cached is not None:
            return cached
        key = f"{prefix}_{self.UUID}_{source}" if prefix else f"{self.UUID}_{source}"
        val = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, key))
        self._suid_cache[cache_key] = val
        return val

    def _send_sample(self, sample: dict, conn_keys: List[str] | Tuple[str, ...], swallow: bool = True) -> None:
        sent = set()
        for key in conn_keys:
            for module in self.connections.get(key, []):
                if module in sent:
                    continue
                sent.add(module)
                try:
                    module.input_cb(sample=sample, data=sample, data_type=IOTypes.SAMPLE)
                except Exception as e:
                    if not swallow:
                        logger.error(f"MotilitySimulator sample dispatch error on {key}: {e}")

    def _build_x_axis(self, raw_x: List[float], has_frame_idx: bool = False) -> List[float]:
        mode = getattr(self, "x_axis_mode", "Fixed ΔT")
        n = len(raw_x)
        fps_val = float(getattr(self, "fps", 30.0) or 30.0)
        dt = float(getattr(self, "delta_t", 1.0 / fps_val) or (1.0 / fps_val))
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

    def _accumulate_point(self, s_uuid: str, name: str, val: float, timestamp: float, frame_idx: Optional[int] = None) -> None:
        with self._buffers_lock:
            buff = self.samples_buffers.setdefault(
                s_uuid, {"x": [], "y": [], "name": name, "t0": None, "has_frame_idx": False}
            )
            if buff["t0"] is None:
                buff["t0"] = timestamp

            if len(buff["x"]) != len(buff["y"]):
                min_len = min(len(buff["x"]), len(buff["y"]))
                buff["x"] = list(buff["x"][:min_len])
                buff["y"] = list(buff["y"][:min_len])

            if _is_num(frame_idx):
                buff["has_frame_idx"] = True
                f_val = int(frame_idx)
                bx = buff["x"]
                if not bx or f_val > bx[-1]:
                    bx.append(f_val)
                    buff["y"].append(float(val))
                else:
                    i = bisect_left(bx, f_val)
                    if i < len(bx) and bx[i] == f_val:
                        buff["y"][i] = float(val)
                    else:
                        bx.insert(i, f_val)
                        buff["y"].insert(i, float(val))
            else:
                buff["x"].append(float(timestamp - buff["t0"]))
                buff["y"].append(float(val))

    def _emit_full_history(self, s_uuid: str, conn_keys: Optional[List[str]] = None) -> None:
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

    def _push_history(self, s_uuid: str, name: str, val: float, timestamp: float, frame_idx: Optional[int], conn_keys: List[str]) -> None:
        self._accumulate_point(s_uuid, name, val, timestamp, frame_idx)
        self._emit_full_history(s_uuid, conn_keys=conn_keys)

    def _clear_downstream_plots(self) -> None:
        target_keys = ("Result", "Count", "Motility")
        with self._buffers_lock:
            items = [(s_uuid, buff["name"]) for s_uuid, buff in self.samples_buffers.items()]
            self.samples_buffers.clear()
        for s_uuid, name in items:
            for action in ("select", "clear"):
                self._send_sample({"name": name, "uuid": s_uuid, "x": [], "y": [], "action": action},
                                  target_keys)
        self._send_sample({"name": self.label, "uuid": str(self.UUID), "x": [], "y": [], "action": "clear"},
                          target_keys)

    def reset_metrics(self, *args: Any, **kwargs: Any) -> None:
        """Full reset of ground truth time-series curves, downstream plots, and tracking history."""
        self._clear_downstream_plots()
        self._tracking_history.clear()
        self._update_metrics(0, 0.0, 0.0)
        self._status("Ground truth plots and metrics reset.")

    def _emit_ground_truth(self, engine: SimulationEngine, frame_idx: int, force_metrics: bool = False) -> Optional[Tuple[int, float, float]]:
        """Extracts and dispatches full ground truth metrics matching CounterTracker output signatures."""
        conn_result = bool(self.connections.get("Result"))
        conn_count = bool(self.connections.get("Count"))
        conn_motility = bool(self.connections.get("Motility"))
        conn_raw_speeds = bool(self.connections.get("Raw Speeds"))
        conn_vectors = bool(self.connections.get("Vectors"))
        conn_tracking = bool(self.connections.get("Tracking"))
        conn_contours = bool(self.connections.get("Contours"))
        conn_cmd = bool(self.connections.get("CMD"))
        any_conn = (
            conn_result or conn_count or conn_motility or conn_raw_speeds
            or conn_vectors or conn_tracking or conn_contours or conn_cmd
        )
        if not any_conn and not force_metrics:
            return None

        timestamp = time.time()
        src = f"{self._source_name} (GT)" if self._source_name else f"{self.label} (GT)"
        need_speeds_list = conn_raw_speeds or conn_vectors or conn_tracking
        need_pos = conn_tracking or conn_contours

        with engine._lock:
            r = engine.radius
            x, y = engine.pos[:, 0], engine.pos[:, 1]
            vis = (x + r > 0) & (x - r < engine.W) & (y + r > 0) & (y - r < engine.H)
            idx = np.nonzero(vis)[0]
            n_vis = int(idx.size)

            if n_vis > 0:
                vx = engine.vel[idx, 0]
                vy = engine.vel[idx, 1]
                sp_um = np.hypot(vx, vy) / engine.conv
                is_mot = engine.motile[idx] & (~engine.stuck[idx])
                n_mot = int(np.count_nonzero(is_mot))
                pct_mot = float(n_mot / n_vis * 100.0)
                mean_spd = float(np.mean(sp_um))

                raw_speeds: List[float] = sp_um.tolist() if need_speeds_list else []
                ang_rad: List[float] = (np.arctan2(vy, vx) % (2.0 * np.pi)).tolist() if conn_vectors else []
                mag_um: List[float] = raw_speeds

                vis_ids = engine.ids[idx] if conn_tracking else np.empty(0, dtype=np.int64)
                pos_x = x[idx] if need_pos else np.empty(0)
                pos_y = y[idx] if need_pos else np.empty(0)
                vis_r = r[idx] if conn_contours else np.empty(0)
            else:
                n_mot = 0
                pct_mot = 0.0
                mean_spd = 0.0
                raw_speeds = []
                ang_rad = []
                mag_um = []
                vis_ids = np.empty(0, dtype=np.int64)
                pos_x = np.empty(0)
                pos_y = np.empty(0)
                vis_r = np.empty(0)
                is_mot = np.empty(0, dtype=bool)

        if not any_conn:
            return (n_vis, pct_mot, mean_spd)

        # ── 1. Result, Count, Motility (Time Series Samples) ───────────
        if conn_result or conn_count or conn_motility:
            if conn_result:
                self._push_history(self._suid("Speed", src), f"Speed_{src}", mean_spd, timestamp, frame_idx, ["Result"])
            if conn_result or conn_count:
                self._push_history(self._suid("Count", src), f"Count_{src}", float(n_vis), timestamp, frame_idx, ["Result", "Count"])
            if conn_result or conn_motility:
                self._push_history(self._suid("Motility", src), f"Motility_{src}", pct_mot, timestamp, frame_idx, ["Result", "Motility"])

        # ── 2. Raw Speeds ─────────────────────────────────────────────
        if conn_raw_speeds and raw_speeds:
            spd_sample = {
                "uuid": self._suid("RawSpeed", src),
                "name": f"RawSpeed_{src}",
                "x": [],
                "y": raw_speeds,
                "action": "select",
            }
            for module in self.connections.get("Raw Speeds", []):
                try:
                    module.input_cb(sample=spd_sample, data=spd_sample, data_type=IOTypes.SAMPLE)
                except Exception as e:
                    logger.debug(f"MotilitySimulator raw speeds dispatch error: {e}")

        # ── 3. Vectors ────────────────────────────────────────────────
        if conn_vectors:
            drift_info = None
            if getattr(engine, "drift_on", False):
                d_spd = float(getattr(self, "drift_speed", 0.0))
                d_ang_ccw = float(getattr(self, "drift_angle", 0.0))
                d_screen_deg = (360.0 - d_ang_ccw) % 360.0
                drift_info = {
                    "enabled": True,
                    "mean_mag_um_s": d_spd,
                    "mean_ang_deg": d_screen_deg,
                    "drift_count": (n_vis - n_mot) if n_vis > 0 else 0,
                }
            payload = {
                "angles": ang_rad,
                "magnitudes": mag_um,
                "mode": "simulation",
                "count": n_vis,
                "motility_pct": pct_mot,
                "mobile_tracks": n_mot,
                "total_tracks": n_vis,
                "drift_info": drift_info,
                "source": src,
                "timestamp": timestamp,
                "frame_idx": frame_idx,
            }
            for module in self.connections.get("Vectors", []):
                try:
                    module.input_cb(data=payload, data_type=IOTypes.VECTORS)
                except Exception as e:
                    logger.debug(f"MotilitySimulator vectors dispatch error: {e}")

        # ── 4. Tracking ───────────────────────────────────────────────
        if conn_tracking:
            tracking_dict = {}
            current_ids = set()
            ids_list = vis_ids.tolist()
            px_list = pos_x.tolist()
            py_list = pos_y.tolist()
            mot_list = is_mot.tolist()
            dt_val = engine.dt
            for i in range(n_vis):
                ind_id = ids_list[i]
                current_ids.add(ind_id)
                hist = self._tracking_history.setdefault(ind_id, [])
                hist.append((px_list[i], py_list[i]))
                if len(hist) > 30:
                    hist.pop(0)
                spd_i = raw_speeds[i] if raw_speeds else 0.0
                tracking_dict[ind_id] = {
                    "points": list(hist),
                    "age": len(hist),
                    "distance": float(len(hist) * spd_i * dt_val),
                    "rolling_speed": spd_i,
                    "is_drift": not mot_list[i],
                    "color": (50, 220, 100),
                }
            stale = [tid for tid in self._tracking_history if tid not in current_ids]
            for tid in stale:
                del self._tracking_history[tid]

            for module in self.connections.get("Tracking", []):
                try:
                    module.input_cb(data=tracking_dict, data_type=IOTypes.TRACKING)
                except Exception as e:
                    logger.debug(f"MotilitySimulator tracking dispatch error: {e}")

        # ── 5. Contours ───────────────────────────────────────────────
        if conn_contours:
            px_list = pos_x.tolist()
            py_list = pos_y.tolist()
            diam_list = (2.0 * vis_r).tolist()
            boxes = [((px_list[i], py_list[i]), (diam_list[i], diam_list[i]), 0.0) for i in range(n_vis)]
            for module in self.connections.get("Contours", []):
                try:
                    module.input_cb(data=boxes, data_type=IOTypes.CONTOURS)
                except Exception as e:
                    logger.debug(f"MotilitySimulator contours dispatch error: {e}")

        # ── 6. CMD ────────────────────────────────────────────────────
        if conn_cmd:
            cmd_payload = {
                "speed_mean": round(mean_spd, 2),
                "speed_std": round(float(getattr(self, "speed_std", 0.0)), 2),
                "non_motile_pct": round(100.0 - pct_mot, 1),
                "drift_on": bool(getattr(self, "drift_on", False)),
                "drift_speed": round(float(getattr(self, "drift_speed", 0.0)), 2),
                "drift_angle": round(float(getattr(self, "drift_angle", 0.0)), 1),
                "calibration": float(getattr(self, "calibration", 0.7575)),
                "fps": float(getattr(self, "fps", 30.0)),
                "n_individuals": n_vis,
            }
            for module in self.connections.get("CMD", []):
                try:
                    module.input_cb(data=cmd_payload, data_type=IOTypes.CMD_DICT)
                except Exception as e:
                    logger.debug(f"MotilitySimulator CMD dispatch error: {e}")

        return (n_vis, pct_mot, mean_spd)

    # ── live generation ───────────────────────────────────────────────────
    def start_live(self, *args: Any, **kwargs: Any) -> None:
        if self._export_thread is not None and self._export_thread.is_alive():
            self._status("Export in progress.", (255, 200, 90))
            return
        self._stop_thread()
        self._clear_downstream_plots()
        self._tracking_history.clear()
        self._gen += 1
        self._running, self._paused = True, False
        self._pending_restart = False
        self._structural_dirty = False
        if dpg.does_item_exist(self.pause_btn_tag):
            dpg.set_item_label(self.pause_btn_tag, "Pause")
        self._live_thread = threading.Thread(target=self._live_loop, args=(self._gen, self._cfg()), daemon=True)
        self._live_thread.start()

    def _stop_thread(self) -> None:
        self._running = False
        t = self._live_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3.0)
        self._live_thread = None

    def stop_live(self, *args: Any, **kwargs: Any) -> None:
        self._stop_thread()
        self._status("Stopped.")
        if self._engine is not None:
            try:
                m = self._engine.live_metrics()
                self._update_metrics(m["count"], m["pct_motile"], m["mean_speed_um"])
            except Exception:
                pass

    def toggle_pause(self, *args: Any, **kwargs: Any) -> None:
        if not self._running:
            return
        self._paused = not self._paused
        dpg.set_item_label(self.pause_btn_tag, "Resume" if self._paused else "Pause")

    def _live_loop(self, gen: int, cfg: Dict[str, Any]) -> None:
        try:
            self._status("Building population...", (255, 200, 90))
            engine = SimulationEngine(cfg)
        except Exception as e:
            logger.exception("MotilitySimulator build error")
            self._status(f"Error: {e}", (255, 90, 90))
            self._running = False
            return
        self._engine = engine
        fps = engine.fps
        total = int(round(float(cfg["duration_s"]) * fps))
        realtime = bool(cfg["realtime"])
        interval = 1.0 / fps
        idx = 0
        self._emit("Trigger", "START", IOTypes.TRIGGER)
        try:
            m = engine.live_metrics()
            self._update_metrics(m["count"], m["pct_motile"], m["mean_speed_um"])
        except Exception:
            pass
        nxt = t_start = time.perf_counter()
        while self._running and gen == self._gen and not self._is_closing:
            # Hot restart handling without thread stopping
            if self._pending_restart:
                with self._cmd_lock:
                    try:
                        fresh_cfg = self._cfg()
                        engine = SimulationEngine(fresh_cfg)
                        self._engine = engine
                        fps = engine.fps
                        total = int(round(float(fresh_cfg["duration_s"]) * fps))
                        realtime = bool(fresh_cfg["realtime"])
                        interval = 1.0 / fps
                        idx = 0
                        nxt = t_start = time.perf_counter()
                        self._pending_restart = False
                        self._structural_dirty = False
                        self._status("New configuration applied live...")
                        m = engine.live_metrics()
                        self._update_metrics(m["count"], m["pct_motile"], m["mean_speed_um"])
                    except Exception as e:
                        logger.error(f"MotilitySimulator live update error: {e}")
                        self._status(f"Live update error: {e}", (255, 90, 90))
                        self._pending_restart = False
                continue

            if self._paused:
                if engine is not None:
                    engine.apply_pending()
                time.sleep(0.02)
                nxt = time.perf_counter()
                continue
            if not self.is_outputs_ready():
                time.sleep(0.001)
                continue
            if realtime:
                now = time.perf_counter()
                if now < nxt:
                    time.sleep(min(0.005, nxt - now))
                    continue
            frame = engine.render()
            self._emit_frame(frame, idx)
            need_ui_metrics = ((idx + 1) % 10 == 0)
            gt_metrics = self._emit_ground_truth(engine, idx, force_metrics=need_ui_metrics)
            engine.step()
            idx += 1
            nxt += interval
            now = time.perf_counter()
            if nxt < now - 0.5:
                nxt = now
            if need_ui_metrics:
                rate = idx / max(now - t_start, 1e-6)
                self._status(f"Running: frame {idx}" + (f"/{total}" if total else "") + f"  ({rate:.1f} fps)")
                try:
                    if gt_metrics is not None:
                        self._update_metrics(*gt_metrics)
                    else:
                        m = engine.live_metrics()
                        self._update_metrics(m["count"], m["pct_motile"], m["mean_speed_um"])
                except Exception:
                    pass
            if total and idx >= total:
                break
        if gen == self._gen:
            self._running = False
            self._status(f"Finished ({idx} frames)." if not self._is_closing else "")
        self._emit("Trigger", "STOP", IOTypes.TRIGGER)

    def photoshock(self, *args: Any, **kwargs: Any) -> None:
        eng = self._engine
        if eng is None or not self._running:
            self._status("Start the simulation before triggering a photoshock.", (255, 200, 90))
            return
        eng.trigger_photoshock(self.shock_intensity, self.shock_duration)
        self._status(
            f"Photoshock! intensity {self.shock_intensity:.0f}%, duration {self.shock_duration:.2f}s",
            (255, 160, 60),
        )

    def preview_frame(self, *args: Any, **kwargs: Any) -> None:
        """Build a fresh population and emit its first frame (to tune the look of the organisms)."""
        cfg = self._cfg()

        def _task() -> None:
            try:
                eng = SimulationEngine(cfg)
                self._emit_frame(eng.render(), 0, "synthetic_preview")
                gt_metrics = self._emit_ground_truth(eng, 0, force_metrics=True)
                if gt_metrics is not None:
                    self._update_metrics(*gt_metrics)
                self._status("Preview frame sent.")
            except Exception as e:
                logger.exception("MotilitySimulator preview error")
                self._status(f"Error: {e}", (255, 90, 90))

        threading.Thread(target=_task, daemon=True).start()

    # ── export ────────────────────────────────────────────────────────────
    def toggle_export(self, *args: Any, **kwargs: Any) -> None:
        if self._export_thread is not None and self._export_thread.is_alive():
            self._export_cancel = True
            return
        self._stop_thread()
        self._export_cancel = False
        dpg.set_item_label(self.export_btn_tag, "Cancel export")
        self._export_thread = threading.Thread(target=self._export_loop, args=(self._cfg(),), daemon=True)
        self._export_thread.start()

    @staticmethod
    def _open_writer(path_base: str, fmt: str, fps: float, size: tuple):
        candidates = {
            EXPORT_FORMATS[0]: [(".avi", "FFV1"), (".avi", "MJPG")],
            EXPORT_FORMATS[1]: [(".avi", "MJPG")],
            EXPORT_FORMATS[2]: [(".mp4", "mp4v"), (".avi", "MJPG")],
        }.get(fmt, [(".avi", "MJPG")])
        for ext, cc in candidates:
            path = path_base + ext
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, size, True)
            if w.isOpened():
                return w, path, cc
            w.release()
        return None, None, None

    def _export_loop(self, cfg: Dict[str, Any]) -> None:
        writer = None
        try:
            os.makedirs(self.export_dir, exist_ok=True)
            name = "".join(ch for ch in (self.export_name or "synthetic") if ch not in '<>:"/\\|?*').strip() or "synthetic"
            base = os.path.join(self.export_dir, name)
            if any(os.path.exists(base + e) for e in (".avi", ".mp4")):
                base += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            self._status("Building population...", (255, 200, 90))
            engine = SimulationEngine(cfg)
            dur = float(cfg["duration_s"]) or 10.0
            total = max(1, int(round(dur * engine.fps)))
            writer, path, codec = self._open_writer(base, self.export_format, engine.fps, (engine.W, engine.H))
            if writer is None:
                raise RuntimeError("No usable video codec found.")
            rows: List[np.ndarray] = []
            t0 = time.perf_counter()
            for i in range(total):
                if self._export_cancel or self._is_closing:
                    break
                writer.write(engine.render())
                rows.append(engine.ground_truth())
                engine.step()
                if i % 10 == 0 or i == total - 1:
                    dpg.set_value(self.progress_tag, (i + 1) / total)
                    rate = (i + 1) / max(time.perf_counter() - t0, 1e-6)
                    self._status(f"Exporting {i + 1}/{total} ({rate:.1f} fps)", (255, 200, 90))
            writer.release()
            writer = None
            if self._export_cancel or self._is_closing:
                self._status("Export cancelled.", (255, 200, 90))
                return
            stem = os.path.splitext(path)[0]
            save_ground_truth(stem + "_gt.csv", rows)
            params = engine.params_dict()
            params["video"] = {"path": path, "codec": codec, "frames": total}
            save_params(stem + "_params.json", params)
            self._status(f"Exported {os.path.basename(path)} (+ _gt.csv, _params.json)")
            self._emit("File Path", (path, os.path.basename(path)), IOTypes.FILE_PATH)
        except Exception as e:
            logger.exception("MotilitySimulator export error")
            self._status(f"Export error: {e}", (255, 90, 90))
        finally:
            if writer is not None:
                writer.release()
            if dpg.does_item_exist(self.export_btn_tag):
                dpg.set_item_label(self.export_btn_tag, "Export video + GT")

    # ── frame-driven overlay & inputs ──────────────────────────────────────
    def _on_update_with_input_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Called when 'Update with input' checkbox is toggled."""
        self.update_with_input = bool(app_data)
        if self.update_with_input:
            self._status("Update with input: ON (applying incoming CMD)", (100, 220, 255))
        else:
            self._status("Update with input: OFF (holding current params)", (255, 200, 90))

    def _on_overlay_view_mode_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Called when overlay view mode dropdown changes."""
        if app_data in OVERLAY_VIEW_MODES:
            self.overlay_view_mode = app_data
        elif isinstance(app_data, str):
            self.overlay_view_mode = app_data

        if self._last_input_frame is not None and self._engine is not None and not self._running:
            with self._cmd_lock:
                try:
                    out_frame = self._render_overlay_view(self._last_input_frame, self._engine)
                    idx = self._last_input_idx if self._last_input_idx is not None else self._gen
                    self._emit_frame(out_frame, idx, "overlay_synthetic")
                    self._status(f"Overlay [{self.overlay_view_mode}]: frame {idx}", (100, 220, 255))
                except Exception as e:
                    logger.debug(f"Error re-rendering overlay view: {e}")

    def _render_overlay_view(self, frame: np.ndarray, engine: SimulationEngine) -> np.ndarray:
        """Renders the appropriate frame according to overlay_view_mode."""
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        mode = self.overlay_view_mode
        if mode == "Input Frame":
            if frame.ndim == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame.copy()
        elif mode == "Simulation":
            if self.bg_mode == BG_MODES[2]:
                synth_bg = np.empty((engine.H, engine.W, 3), dtype=np.uint8)
                bg_col = getattr(self, "bg_color", [200, 200, 200])
                synth_bg[:] = (int(bg_col[2]), int(bg_col[1]), int(bg_col[0]))
                return engine.render(bg_frame=synth_bg)
            else:
                return engine.render()
        else:  # "Blend (Overlay)"
            return engine.render(bg_frame=frame)

    def _handle_input_frame(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> None:
        """Handles an incoming video frame (from Counter Tracker or Video Reader) in overlay mode."""
        if self._running:
            self._stop_thread()
            if dpg.does_item_exist(self.pause_btn_tag):
                dpg.set_item_label(self.pause_btn_tag, "Pause")

        # Auto-switch bg_mode to Live frame overlay if still on default solid color
        if self.bg_mode == BG_MODES[0]:
            self.bg_mode = BG_MODES[2]
            p_bg = _PARAM_BY_KEY.get("bg_mode")
            if p_bg:
                self._write_widget(p_bg, self.bg_mode)
            self._refresh_visibility()

        with self._cmd_lock:
            if self._engine is None or self._pending_restart:
                cfg = self._cfg()
                cfg["width"] = int(frame.shape[1])
                cfg["height"] = int(frame.shape[0])
                try:
                    self._engine = SimulationEngine(cfg)
                    self._pending_restart = False
                except Exception as e:
                    logger.error(f"MotilitySimulator overlay build error: {e}")
                    return

            engine = self._engine
            if engine.W != frame.shape[1] or engine.H != frame.shape[0]:
                engine.W = int(frame.shape[1])
                engine.H = int(frame.shape[0])
                engine._canvas = np.empty((engine.H, engine.W, 3), dtype=np.float32)
                engine._clamp_inside()

            # Advance physics one step
            engine.step()

            # Render overlay directly or according to overlay_view_mode
            out_frame = self._render_overlay_view(frame, engine)

            eff_idx = frame_idx if frame_idx is not None else self._gen
            self._gen += 1
            self._last_input_frame = frame
            self._last_input_idx = eff_idx
            self._emit_frame(out_frame, eff_idx, "overlay_synthetic")
            gt_metrics = self._emit_ground_truth(engine, eff_idx, force_metrics=True)

            try:
                if gt_metrics is not None:
                    self._update_metrics(*gt_metrics)
                    if eff_idx % 15 == 0:
                        self._status(f"Overlay [{self.overlay_view_mode}]: frame {eff_idx}  |  {gt_metrics[0]} simulated algae", (100, 220, 255))
            except Exception:
                pass

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """Handles incoming commands (CMD_DICT) or live video frames (FRAME)."""
        data = kwargs.get("data") if kwargs.get("data") is not None else (args[0] if args else None)
        data_type = kwargs.get("data_type") if kwargs.get("data_type") is not None else kwargs.get("datatype")

        # ── 0. TRIGGER input ──────────────────────────────────────────────
        if data_type == IOTypes.TRIGGER or (isinstance(data, str) and data.strip().upper() in ("START", "STOP", "RESET")):
            cmd = str(data).strip().upper()
            if cmd == "START":
                self.start_live()
                return
            elif cmd == "STOP":
                self.stop_live()
                return
            elif cmd == "RESET":
                self.reset_metrics()
                return

        # ── 1. FRAME input ────────────────────────────────────────────────
        is_frame = (data_type == IOTypes.FRAME) or isinstance(data, np.ndarray) or (
            isinstance(data, (tuple, list)) and len(data) > 0 and isinstance(data[0], np.ndarray)
        )
        if is_frame:
            frame, frame_idx = None, None
            if isinstance(data, np.ndarray):
                frame = data
            elif isinstance(data, (tuple, list)):
                frame = data[0]
                if len(data) >= 2 and isinstance(data[1], str):
                    self._source_name = data[1]
                if len(data) >= 3 and isinstance(data[2], (int, np.integer)) and not isinstance(data[2], bool):
                    frame_idx = int(data[2])
                elif len(data) >= 2 and isinstance(data[1], (int, np.integer)) and not isinstance(data[1], bool):
                    frame_idx = int(data[1])

            if frame is not None:
                self._handle_input_frame(frame, frame_idx)
            return

        # ── 2. CMD_DICT input ─────────────────────────────────────────────
        if not isinstance(data, dict):
            return

        if "source" in data and isinstance(data["source"], str):
            self._source_name = data["source"]

        # Check if dict contains a frame payload
        if "frame" in data and isinstance(data["frame"], np.ndarray):
            f = data["frame"]
            f_idx = data.get("frame_idx", data.get("idx"))
            self._handle_input_frame(f, f_idx)

        # Apply parameters only if update_with_input is enabled
        if self.update_with_input:
            if "preset" in data or "config" in data:
                preset_val = data.get("preset", data.get("config"))
                if preset_val in self._presets:
                    self._load_config_file(self._presets[preset_val])
                elif os.path.exists(str(preset_val)):
                    self._load_config_file(str(preset_val))

            cmd_changes = {}
            for k, v in data.items():
                if k == "n_individuals" and not getattr(self, "sync_count", False):
                    continue
                param = _PARAM_BY_KEY.get(k)
                if param is not None:
                    try:
                        if param.kind == "int":
                            val = int(v)
                        elif param.kind == "float":
                            val = float(v)
                        elif param.kind == "bool":
                            val = bool(v)
                        else:
                            val = v
                        setattr(self, k, val)
                        self._write_widget(param, val)
                        cmd_changes[k] = val
                    except (ValueError, TypeError):
                        pass

            if cmd_changes and self._engine is not None:
                self._engine.update_params(cmd_changes)

            self._refresh_visibility()

        if data.get("stop"):
            self.stop_live()
        if data.get("start"):
            self.start_live()
        if data.get("pause"):
            self.toggle_pause()
        if data.get("photoshock"):
            self.photoshock()
        if data.get("export"):
            self.toggle_export()
        if data.get("reset"):
            self.reset_metrics()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def on_close(self) -> None:
        if self._is_closing:
            return
        self._is_closing = True
        self._export_cancel = True
        self._clear_downstream_plots()
        self._stop_thread()
        t = self._export_thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        super().on_close()

    def close(self) -> None:
        self.on_close()
        super().close()

    def destroy(self) -> None:
        self.close()
        super().destroy()


# Default values as class attributes so WindowBase restores/serialises every parameter.
for _p in PARAMS:
    setattr(MotilitySimulator_win, _p.key, list(_p.default) if isinstance(_p.default, list) else _p.default)

MotilitySimulatorWin = MotilitySimulator_win
EXPORTED_CLASS = MotilitySimulator_win
EXPORTED_NAME = "Motility Simulator"
