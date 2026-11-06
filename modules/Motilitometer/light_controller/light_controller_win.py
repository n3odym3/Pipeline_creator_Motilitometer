from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
from loguru import logger

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from .light_controller_driver import DEFAULT_CALIBRATION, LightControllerDriver

_CHANNEL_CMD_RE = re.compile(r"(?i)(ir|r|g|b|w)\s*(\d+)")


class LightControllerWin(WindowBase):
    """
    LightControllerWin:
    Motilitometer module for controlling RGBW and IR illumination channels via Serial.
    Features:
    - Calibrated µE to hardware PWM conversion
    - Interactive direct sliders (R, G, B, W, IR)
    - Dynamic visual feedback pastille mixing RGB + White light in real-time
    - Pulse trigger with configurable duration
    - Full integration with SequenceProcessor commands (e.g. R100G50, R100, etc.)
    - Live serial console monitor and editable calibration settings
    """

    # 1. Window Configuration
    label = "Light Controller"
    win_width = display_scaling.scale(540)
    win_height = display_scaling.scale(580)

    # 2. Input/Output definitions
    accepted_input_types = [IOTypes.CMD_DICT, IOTypes.TEXT, IOTypes.TRIGGER]
    outputs = {
        "Status": IOTypes.STATUS_DICT,
        "received_data": IOTypes.TEXT,
    }

    # 3. Persistent fields saved into pipeline JSON
    persistent_fields = [
        "label",
        "selected_port",
        "selected_baudrate",
        "auto_send",
        "r_val",
        "g_val",
        "b_val",
        "w_val",
        "ir_val",
        "pulse_dur",
        "pulse_r",
        "pulse_g",
        "pulse_b",
        "pulse_w",
        "pulse_ir",
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Driver initialization
        calib_file = os.path.join(os.path.dirname(__file__), "light_calibration.json")
        self.driver = LightControllerDriver(calib_file_path=calib_file)
        self.driver.on_data_received = self._on_serial_data_received

        # State variables
        self.selected_port = str(kwargs.get("selected_port", getattr(self, "selected_port", "") or ""))
        self.selected_baudrate = int(kwargs.get("selected_baudrate", getattr(self, "selected_baudrate", 115200) or 115200))
        raw_auto = kwargs.get("auto_send", getattr(self, "auto_send", None))
        self.auto_send = bool(raw_auto) if raw_auto is not None else True

        self.r_val = int(kwargs.get("r_val", getattr(self, "r_val", 0) or 0))
        self.g_val = int(kwargs.get("g_val", getattr(self, "g_val", 0) or 0))
        self.b_val = int(kwargs.get("b_val", getattr(self, "b_val", 0) or 0))
        self.w_val = int(kwargs.get("w_val", getattr(self, "w_val", 0) or 0))
        self.ir_val = int(kwargs.get("ir_val", getattr(self, "ir_val", 0) or 0))

        self.pulse_dur = int(kwargs.get("pulse_dur", getattr(self, "pulse_dur", 100) or 100))
        self.pulse_r = int(kwargs.get("pulse_r", getattr(self, "pulse_r", 0) or 0))
        self.pulse_g = int(kwargs.get("pulse_g", getattr(self, "pulse_g", 0) or 0))
        self.pulse_b = int(kwargs.get("pulse_b", getattr(self, "pulse_b", 0) or 0))
        self.pulse_w = int(kwargs.get("pulse_w", getattr(self, "pulse_w", 0) or 0))
        self.pulse_ir = int(kwargs.get("pulse_ir", getattr(self, "pulse_ir", 0) or 0))

        # Unique UI Tags
        self.combo_port_tag = f"lc_port_{self.UUID}"
        self.combo_baud_tag = f"lc_baud_{self.UUID}"
        self.btn_connect_tag = f"lc_connect_{self.UUID}"
        self.status_dot_tag = f"lc_status_dot_{self.UUID}"
        self.status_text_tag = f"lc_status_text_{self.UUID}"

        self.pastille_drawlist_tag = f"lc_pastille_dl_{self.UUID}"
        self.pastille_circle_tag = f"lc_pastille_circ_{self.UUID}"
        self.pastille_hex_tag = f"lc_pastille_hex_{self.UUID}"

        self.r_slider_tag = f"lc_r_slider_{self.UUID}"
        self.g_slider_tag = f"lc_g_slider_{self.UUID}"
        self.b_slider_tag = f"lc_b_slider_{self.UUID}"
        self.w_slider_tag = f"lc_w_slider_{self.UUID}"
        self.ir_slider_tag = f"lc_ir_slider_{self.UUID}"

        self.r_pwm_tag = f"lc_r_pwm_{self.UUID}"
        self.g_pwm_tag = f"lc_g_pwm_{self.UUID}"
        self.b_pwm_tag = f"lc_b_pwm_{self.UUID}"
        self.w_pwm_tag = f"lc_w_pwm_{self.UUID}"
        self.ir_pwm_tag = f"lc_ir_pwm_{self.UUID}"

        self.auto_send_tag = f"lc_auto_send_{self.UUID}"

        self.pulse_dur_tag = f"lc_pulse_dur_{self.UUID}"
        self.pulse_r_tag = f"lc_pulse_r_{self.UUID}"
        self.pulse_g_tag = f"lc_pulse_g_{self.UUID}"
        self.pulse_b_tag = f"lc_pulse_b_{self.UUID}"
        self.pulse_w_tag = f"lc_pulse_w_{self.UUID}"
        self.pulse_ir_tag = f"lc_pulse_ir_{self.UUID}"

        self.console_tag = f"lc_console_{self.UUID}"

        # Calibration UI tags
        self.cal_max_tag = f"lc_cal_max_{self.UUID}"
        self.cal_lowcorr_tag = f"lc_cal_lowcorr_{self.UUID}"
        self.cal_lowm_tag = f"lc_cal_lowm_{self.UUID}"
        self.cal_lowp_tag = f"lc_cal_lowp_{self.UUID}"
        self.cal_m_tag = f"lc_cal_m_{self.UUID}"
        self.cal_p_tag = f"lc_cal_p_{self.UUID}"

        # Build UI
        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        scale = display_scaling.scale
        max_light = self.driver.calibration.get("light", {}).get("max", 500)
        available_ports = self.driver.list_com_ports()

        with self.window(on_close=self.on_close):
            # 1. Connection Header
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Refresh", callback=self.refresh_ports_cb)
                    dpg.add_combo(
                        items=available_ports,
                        default_value=self.selected_port if self.selected_port in available_ports else (available_ports[0] if available_ports else ""),
                        tag=self.combo_port_tag,
                        width=scale(140),
                        callback=self._on_port_changed,
                    )
                    dpg.add_combo(
                        items=["9600", "19200", "38400", "57600", "115200", "230400"],
                        default_value=str(self.selected_baudrate),
                        tag=self.combo_baud_tag,
                        width=scale(90),
                        callback=self._on_baud_changed,
                    )
                    dpg.add_button(label="Connect", tag=self.btn_connect_tag, callback=self.toggle_connect_cb, width=scale(90))

                with dpg.group(horizontal=True):
                    dpg.add_text("Status: ")
                    dpg.add_text("Disconnected", tag=self.status_text_tag, color=[255, 80, 80, 255])

            dpg.add_separator()

            # 2. Real-Time Visual Color Feedback Header & Quick Actions
            with dpg.group(horizontal=True):
                # Color Pastille Drawing
                with dpg.drawlist(width=scale(50), height=scale(50), tag=self.pastille_drawlist_tag):
                    # Background circle outline
                    dpg.draw_circle(center=[scale(25), scale(25)], radius=scale(21), color=[180, 180, 180, 255], thickness=2)
                    # Dynamic color circle
                    dpg.draw_circle(
                        center=[scale(25), scale(25)],
                        radius=scale(19),
                        color=[0, 0, 0, 0],
                        fill=[20, 20, 20, 255],
                        tag=self.pastille_circle_tag,
                    )
                with dpg.group():
                    dpg.add_text("RGB Illumination Preview")
                    dpg.add_text("#000000 | 0 µE", tag=self.pastille_hex_tag, color=[200, 200, 200, 255])

                dpg.add_spacer(width=scale(20))
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label="Auto-send", tag=self.auto_send_tag, default_value=self.auto_send, callback=self._on_auto_send_changed)
                        dpg.add_button(label="Send Now", callback=lambda: self.send_current_channels())
                        dpg.add_button(label="All OFF", callback=lambda: self.blackout_cb())

            dpg.add_separator()

            # 3. Continuous Light Controls
            dpg.add_text("Continuous Light (µE / Intensity)")

            # Red Slider
            with dpg.group(horizontal=True):
                dpg.add_text("R", color=[255, 100, 100, 255])
                dpg.add_slider_int(
                    tag=self.r_slider_tag,
                    min_value=0,
                    max_value=max_light,
                    default_value=self.r_val,
                    width=scale(330),
                    callback=self._on_channel_slider_changed,
                )
                dpg.add_text("PWM: 0", tag=self.r_pwm_tag)

            # Green Slider
            with dpg.group(horizontal=True):
                dpg.add_text("G", color=[100, 255, 100, 255])
                dpg.add_slider_int(
                    tag=self.g_slider_tag,
                    min_value=0,
                    max_value=max_light,
                    default_value=self.g_val,
                    width=scale(330),
                    callback=self._on_channel_slider_changed,
                )
                dpg.add_text("PWM: 0", tag=self.g_pwm_tag)

            # Blue Slider
            with dpg.group(horizontal=True):
                dpg.add_text("B", color=[100, 160, 255, 255])
                dpg.add_slider_int(
                    tag=self.b_slider_tag,
                    min_value=0,
                    max_value=max_light,
                    default_value=self.b_val,
                    width=scale(330),
                    callback=self._on_channel_slider_changed,
                )
                dpg.add_text("PWM: 0", tag=self.b_pwm_tag)

            # White Slider
            with dpg.group(horizontal=True):
                dpg.add_text("W", color=[240, 240, 240, 255])
                dpg.add_slider_int(
                    tag=self.w_slider_tag,
                    min_value=0,
                    max_value=max_light,
                    default_value=self.w_val,
                    width=scale(330),
                    callback=self._on_channel_slider_changed,
                )
                dpg.add_text("PWM: 0", tag=self.w_pwm_tag)

            # IR Slider
            with dpg.group(horizontal=True):
                dpg.add_text("IR", color=[200, 100, 255, 255])
                dpg.add_slider_int(
                    tag=self.ir_slider_tag,
                    min_value=0,
                    max_value=1000,
                    default_value=self.ir_val,
                    width=scale(330),
                    callback=self._on_channel_slider_changed,
                )
                dpg.add_text("Val: 0", tag=self.ir_pwm_tag)

            dpg.add_separator()

            # 4. Pulse / Flash Mode
            with dpg.collapsing_header(label="Pulse / Flash Mode", default_open=False):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Trigger Pulse", callback=self.trigger_pulse_cb, width=scale(130))
                    dpg.add_text("Duration (ms):")
                    dpg.add_drag_int(tag=self.pulse_dur_tag, min_value=1, max_value=5000, default_value=self.pulse_dur, width=scale(80))

                with dpg.group(horizontal=True):
                    dpg.add_text("R:")
                    dpg.add_drag_int(tag=self.pulse_r_tag, min_value=0, max_value=max_light, default_value=self.pulse_r, width=scale(50))
                    dpg.add_text("G:")
                    dpg.add_drag_int(tag=self.pulse_g_tag, min_value=0, max_value=max_light, default_value=self.pulse_g, width=scale(50))
                    dpg.add_text("B:")
                    dpg.add_drag_int(tag=self.pulse_b_tag, min_value=0, max_value=max_light, default_value=self.pulse_b, width=scale(50))
                    dpg.add_text("W:")
                    dpg.add_drag_int(tag=self.pulse_w_tag, min_value=0, max_value=max_light, default_value=self.pulse_w, width=scale(50))
                    dpg.add_text("IR:")
                    dpg.add_drag_int(tag=self.pulse_ir_tag, min_value=0, max_value=1000, default_value=self.pulse_ir, width=scale(50))

            # 5. Calibration Settings
            with dpg.collapsing_header(label="Calibration Settings", default_open=False):
                cal = self.driver.calibration.get("light", DEFAULT_CALIBRATION["light"])
                with dpg.group(horizontal=True):
                    dpg.add_text("Max µE:")
                    dpg.add_input_int(tag=self.cal_max_tag, default_value=cal.get("max", 500), width=scale(90))
                    dpg.add_text("Low Corr Threshold:")
                    dpg.add_input_float(tag=self.cal_lowcorr_tag, default_value=cal.get("low_corr", 15.0), width=scale(90))

                with dpg.group(horizontal=True):
                    dpg.add_text("Low M:")
                    dpg.add_input_float(tag=self.cal_lowm_tag, default_value=cal.get("low_m", 0.2), format="%.4f", width=scale(90))
                    dpg.add_text("Low P:")
                    dpg.add_input_float(tag=self.cal_lowp_tag, default_value=cal.get("low_p", -5.0), format="%.4f", width=scale(90))

                with dpg.group(horizontal=True):
                    dpg.add_text("High M:")
                    dpg.add_input_float(tag=self.cal_m_tag, default_value=cal.get("m", 0.4975), format="%.4f", width=scale(90))
                    dpg.add_text("High P:")
                    dpg.add_input_float(tag=self.cal_p_tag, default_value=cal.get("p", -48.897), format="%.4f", width=scale(90))

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Apply & Save Calibration", callback=self._apply_calibration_cb)
                    dpg.add_button(label="Reset Defaults", callback=self._reset_calibration_cb)

            # 6. Serial Console / Feedback
            with dpg.collapsing_header(label="Serial Console & Feedback", default_open=False):
                dpg.add_input_text(
                    tag=self.console_tag,
                    multiline=True,
                    default_value="",
                    height=scale(100),
                    width=-1,
                    readonly=True,
                )

        self._update_pastille()
        self._update_pwm_displays()

    def update_permission(self) -> None:
        """Lock critical settings when in user mode."""
        is_user = (app_state.mode == "user")
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def refresh_ports_cb(self, *args, **kwargs) -> None:
        """Refreshes available COM ports in dropdown."""
        ports = self.driver.list_com_ports()
        dpg.configure_item(self.combo_port_tag, items=ports)
        if ports:
            current = dpg.get_value(self.combo_port_tag)
            if current not in ports:
                dpg.set_value(self.combo_port_tag, ports[0])
                self.selected_port = ports[0]
        self._log_console(f"Found COM ports: {ports}")

    def _on_port_changed(self, sender: str, app_data: str, *args, **kwargs) -> None:
        self.selected_port = app_data

    def _on_baud_changed(self, sender: str, app_data: str, *args, **kwargs) -> None:
        try:
            self.selected_baudrate = int(app_data)
        except ValueError:
            pass

    def _on_auto_send_changed(self, sender: str, app_data: bool, *args, **kwargs) -> None:
        self.auto_send = bool(app_data)

    def toggle_connect_cb(self, *args, **kwargs) -> None:
        """Toggles serial connection."""
        if self.driver.is_connected:
            self.driver.disconnect()
            self._update_connection_ui(False)
        else:
            port = dpg.get_value(self.combo_port_tag)
            if not port:
                self._log_console("No port selected!")
                return
            baud = int(dpg.get_value(self.combo_baud_tag))
            success = self.driver.connect(port=port, baudrate=baud)
            self._update_connection_ui(success)

    def _update_connection_ui(self, connected: bool) -> None:
        if connected:
            dpg.configure_item(self.btn_connect_tag, label="Disconnect")
            dpg.set_value(self.status_text_tag, f"Connected ({self.driver.port} @ {self.driver.baudrate})")
            dpg.configure_item(self.status_text_tag, color=[80, 255, 80, 255])
            self._log_console(f"Connected to {self.driver.port}")
        else:
            dpg.configure_item(self.btn_connect_tag, label="Connect")
            dpg.set_value(self.status_text_tag, "Disconnected")
            dpg.configure_item(self.status_text_tag, color=[255, 80, 80, 255])
            self._log_console("Disconnected")

    def _on_channel_slider_changed(self, sender: Optional[str] = None, app_data: Any = None, *args, **kwargs) -> None:
        """Invoked when any continuous slider is dragged/changed."""
        if dpg.does_item_exist(self.r_slider_tag):
            self.r_val = int(dpg.get_value(self.r_slider_tag))
            self.g_val = int(dpg.get_value(self.g_slider_tag))
            self.b_val = int(dpg.get_value(self.b_slider_tag))
            self.w_val = int(dpg.get_value(self.w_slider_tag))
            self.ir_val = int(dpg.get_value(self.ir_slider_tag))

        self._update_pastille()

        if self.auto_send:
            self.send_current_channels()
        else:
            self._update_pwm_displays()

    def _update_pastille(self) -> None:
        """Calculates combined visual RGB color for the feedback pastille."""
        max_light = max(1.0, float(self.driver.calibration.get("light", {}).get("max", 500)))

        # Normalize 0.0 - 1.0
        rn = min(1.0, max(0.0, self.r_val / max_light))
        gn = min(1.0, max(0.0, self.g_val / max_light))
        bn = min(1.0, max(0.0, self.b_val / max_light))
        wn = min(1.0, max(0.0, self.w_val / max_light))

        # Add white contribution to RGB channels
        r_disp = int(min(255, (rn + wn * 0.6) * 255))
        g_disp = int(min(255, (gn + wn * 0.6) * 255))
        b_disp = int(min(255, (bn + wn * 0.6) * 255))

        # Fallback dark background if all 0
        if r_disp == 0 and g_disp == 0 and b_disp == 0:
            fill_color = [20, 20, 20, 255]
        else:
            fill_color = [r_disp, g_disp, b_disp, 255]

        hex_str = f"#{r_disp:02X}{g_disp:02X}{b_disp:02X}"
        total_ue = self.r_val + self.g_val + self.b_val + self.w_val

        if dpg.does_item_exist(self.pastille_circle_tag):
            dpg.configure_item(self.pastille_circle_tag, fill=fill_color)
        if dpg.does_item_exist(self.pastille_hex_tag):
            dpg.set_value(self.pastille_hex_tag, f"{hex_str} | Total: {total_ue} µE")

    def _update_pwm_displays(self, pwm_dict: Optional[Dict[str, int]] = None) -> None:
        """Updates live calculated PWM labels."""
        if dpg.does_item_exist(self.r_pwm_tag):
            if pwm_dict is not None:
                pwm_r = pwm_dict["r"]
                pwm_g = pwm_dict["g"]
                pwm_b = pwm_dict["b"]
                pwm_w = pwm_dict["w"]
            else:
                pwm_r = self.driver.get_pwm_val(self.r_val)
                pwm_g = self.driver.get_pwm_val(self.g_val)
                pwm_b = self.driver.get_pwm_val(self.b_val)
                pwm_w = self.driver.get_pwm_val(self.w_val)
            dpg.set_value(self.r_pwm_tag, f"PWM: {pwm_r}")
            dpg.set_value(self.g_pwm_tag, f"PWM: {pwm_g}")
            dpg.set_value(self.b_pwm_tag, f"PWM: {pwm_b}")
            dpg.set_value(self.w_pwm_tag, f"PWM: {pwm_w}")
            dpg.set_value(self.ir_pwm_tag, f"Val: {self.ir_val}")

    def send_current_channels(self) -> None:
        """Sends current RGBWIR values to hardware and emits downstream status."""
        success, pwm_dict = self.driver.send_light_command(
            r_ue=self.r_val,
            g_ue=self.g_val,
            b_ue=self.b_val,
            w_ue=self.w_val,
            ir_val=self.ir_val,
        )
        self._update_pwm_displays(pwm_dict)
        msg = f"Sent Light: R={self.r_val}({pwm_dict['r']}) G={self.g_val}({pwm_dict['g']}) B={self.b_val}({pwm_dict['b']}) W={self.w_val}({pwm_dict['w']}) IR={self.ir_val}"
        self._log_console(msg)
        if self.connections.get("Status"):
            self._emit_status({"type": "continuous", "channels": {"r": self.r_val, "g": self.g_val, "b": self.b_val, "w": self.w_val, "ir": self.ir_val}, "pwm": pwm_dict})

    def blackout_cb(self) -> None:
        """Turns all channels to 0."""
        self.r_val = 0
        self.g_val = 0
        self.b_val = 0
        self.w_val = 0
        self.ir_val = 0

        for tag in [self.r_slider_tag, self.g_slider_tag, self.b_slider_tag, self.w_slider_tag, self.ir_slider_tag]:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, 0)

        self._update_pastille()
        self._update_pwm_displays({"r": 0, "g": 0, "b": 0, "w": 0})
        self.driver.blackout()
        self._log_console("Blackout: all channels 0")
        if self.connections.get("Status"):
            self._emit_status({"type": "blackout", "channels": {"r": 0, "g": 0, "b": 0, "w": 0, "ir": 0}})

    def trigger_pulse_cb(self, *args, **kwargs) -> None:
        """Triggers a calibrated pulse command."""
        dur = int(dpg.get_value(self.pulse_dur_tag)) if dpg.does_item_exist(self.pulse_dur_tag) else self.pulse_dur
        pr = int(dpg.get_value(self.pulse_r_tag)) if dpg.does_item_exist(self.pulse_r_tag) else self.pulse_r
        pg = int(dpg.get_value(self.pulse_g_tag)) if dpg.does_item_exist(self.pulse_g_tag) else self.pulse_g
        pb = int(dpg.get_value(self.pulse_b_tag)) if dpg.does_item_exist(self.pulse_b_tag) else self.pulse_b
        pw = int(dpg.get_value(self.pulse_w_tag)) if dpg.does_item_exist(self.pulse_w_tag) else self.pulse_w
        pir = int(dpg.get_value(self.pulse_ir_tag)) if dpg.does_item_exist(self.pulse_ir_tag) else self.pulse_ir

        success, pulse_dict = self.driver.send_pulse_command(
            duration_ms=dur,
            r_ue=pr,
            g_ue=pg,
            b_ue=pb,
            w_ue=pw,
            ir_val=pir,
        )
        self._log_console(f"Triggered Pulse: {dur}ms | R:{pr} G:{pg} B:{pb} W:{pw} IR:{pir}")
        if self.connections.get("Status"):
            self._emit_status({"type": "pulse", "pulse": pulse_dict})

    def _apply_calibration_cb(self, *args, **kwargs) -> None:
        """Applies calibration changes from UI inputs."""
        try:
            max_val = int(dpg.get_value(self.cal_max_tag))
            low_corr = float(dpg.get_value(self.cal_lowcorr_tag))
            low_m = float(dpg.get_value(self.cal_lowm_tag))
            low_p = float(dpg.get_value(self.cal_lowp_tag))
            m_val = float(dpg.get_value(self.cal_m_tag))
            p_val = float(dpg.get_value(self.cal_p_tag))

            self.driver.calibration["light"] = {
                "max": max_val,
                "low_corr": low_corr,
                "low_m": low_m,
                "low_p": low_p,
                "m": m_val,
                "p": p_val,
            }
            self.driver.save_calibration()

            # Update slider limits
            for tag in [self.r_slider_tag, self.g_slider_tag, self.b_slider_tag, self.w_slider_tag]:
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, max_value=max_val)

            self._update_pwm_displays()
            self._update_pastille()
            self._log_console("Calibration applied and saved.")
        except Exception as e:
            self._log_console(f"Calibration error: {e}")

    def _reset_calibration_cb(self, *args, **kwargs) -> None:
        """Resets calibration to defaults."""
        self.driver.calibration = {"light": dict(DEFAULT_CALIBRATION["light"])}
        cal = DEFAULT_CALIBRATION["light"]
        if dpg.does_item_exist(self.cal_max_tag):
            dpg.set_value(self.cal_max_tag, cal["max"])
            dpg.set_value(self.cal_lowcorr_tag, cal["low_corr"])
            dpg.set_value(self.cal_lowm_tag, cal["low_m"])
            dpg.set_value(self.cal_lowp_tag, cal["low_p"])
            dpg.set_value(self.cal_m_tag, cal["m"])
            dpg.set_value(self.cal_p_tag, cal["p"])
        self._apply_calibration_cb()
        self._log_console("Calibration reset to factory defaults.")

    def _on_serial_data_received(self, text: str) -> None:
        """Handles incoming serial data stream."""
        self._log_console(f"<< {text}")
        for mod in self.connections.get("received_data", []):
            try:
                mod.input_cb(data=text, data_type=IOTypes.TEXT)
            except Exception as e:
                logger.error(f"[LightControllerWin] Error emitting received data: {e}")

    def _log_console(self, text: str) -> None:
        """Appends message to internal console log."""
        if dpg.does_item_exist(self.console_tag):
            cur = dpg.get_value(self.console_tag)
            ts = datetime.now().strftime("[%H:%M:%S] ")
            new_text = f"{cur}\n{ts}{text}" if cur else f"{ts}{text}"
            if len(new_text) > 4000:
                new_text = new_text[-4000:]
            dpg.set_value(self.console_tag, new_text)

    def _emit_status(self, status: Dict[str, Any]) -> None:
        for mod in self.connections.get("Status", []):
            try:
                mod.input_cb(data=status, data_type=IOTypes.STATUS_DICT)
            except Exception as e:
                logger.error(f"[LightControllerWin] Status emit error: {e}")

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Receives incoming commands from SequenceProcessor or other modules.
        Supports CMD_DICT, JSON strings, compact syntax R[VAL]G[VAL]B[VAL]W[VAL]IR[VAL], etc.
        """
        data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
        data_type = kwargs.get("data_type")

        # 1. Handle Trigger
        if data_type == IOTypes.TRIGGER or (isinstance(data, str) and data.lower() == "trigger"):
            self.trigger_pulse_cb()
            return

        # 2. Handle Text / String Commands (e.g. 'R100G50', 'r 100', JSON string)
        if data_type is IOTypes.TEXT or isinstance(data, str):
            cmd_str = str(data).strip()

            # Check if JSON formatted string
            if cmd_str.startswith("{") and cmd_str.endswith("}"):
                try:
                    data = json.loads(cmd_str)
                except Exception:
                    pass
            else:
                # Parse compact string format: R[VAL]G[VAL]B[VAL]W[VAL]IR[VAL]
                # Default all channels to 0 if a new channel command is received
                matches = list(_CHANNEL_CMD_RE.finditer(cmd_str))

                if matches:
                    channels = {"r": 0, "g": 0, "b": 0, "w": 0, "ir": 0}
                    for m in matches:
                        ch = m.group(1).lower()
                        val = int(m.group(2))
                        channels[ch] = val
                    self._apply_channel_dict(channels)
                    return

        # 3. Handle Command Dictionaries
        if data_type == IOTypes.CMD_DICT or isinstance(data, dict):
            cmd_payload = kwargs.get("cmd") if kwargs.get("cmd") is not None else data

            if not isinstance(cmd_payload, dict):
                return

            if "connect" in cmd_payload:
                port = cmd_payload.get("port", self.selected_port)
                baud = cmd_payload.get("baudrate", self.selected_baudrate)
                if port:
                    self.driver.connect(port=port, baudrate=baud)
                    self._update_connection_ui(self.driver.is_connected)
                return

            if "disconnect" in cmd_payload:
                self.driver.disconnect()
                self._update_connection_ui(False)
                return

            if "blackout" in cmd_payload or cmd_payload.get("off", False):
                self.blackout_cb()
                return

            if "pulse" in cmd_payload:
                if isinstance(cmd_payload["pulse"], dict):
                    p_info = cmd_payload["pulse"]
                    dur = p_info.get("pulse", p_info.get("duration", 100))
                    self.driver.send_pulse_command(
                        duration_ms=dur,
                        r_ue=p_info.get("r", 0),
                        g_ue=p_info.get("g", 0),
                        b_ue=p_info.get("b", 0),
                        w_ue=p_info.get("w", 0),
                        ir_val=p_info.get("ir", 0),
                    )
                else:
                    self.trigger_pulse_cb()
                return

            if "light" in cmd_payload and isinstance(cmd_payload["light"], dict):
                self._apply_channel_dict(cmd_payload["light"])
                return

            # Direct dictionary with channel keys
            ch_keys = {"r", "g", "b", "w", "ir"}
            if any(k in cmd_payload for k in ch_keys):
                channels = {
                    "r": cmd_payload.get("r", 0),
                    "g": cmd_payload.get("g", 0),
                    "b": cmd_payload.get("b", 0),
                    "w": cmd_payload.get("w", 0),
                    "ir": cmd_payload.get("ir", 0),
                }
                self._apply_channel_dict(channels)
                return

            if "send" in cmd_payload:
                # Raw string or JSON send
                raw_cmd = str(cmd_payload["send"]).strip()
                if raw_cmd.startswith("{") and raw_cmd.endswith("}"):
                    try:
                        parsed = json.loads(raw_cmd)
                        if any(k in parsed for k in ch_keys):
                            self._apply_channel_dict(parsed)
                            return
                    except Exception:
                        pass
                self.driver.send_raw_command(raw_cmd)
                self._log_console(f">> {raw_cmd}")

    def _apply_channel_dict(self, channels: Dict[str, Any]) -> None:
        """Applies channel dict, updates UI sliders and pastille, and sends to hardware."""
        self.r_val = int(channels.get("r", 0))
        self.g_val = int(channels.get("g", 0))
        self.b_val = int(channels.get("b", 0))
        self.w_val = int(channels.get("w", 0))
        self.ir_val = int(channels.get("ir", 0))

        if dpg.does_item_exist(self.r_slider_tag):
            dpg.set_value(self.r_slider_tag, self.r_val)
            dpg.set_value(self.g_slider_tag, self.g_val)
            dpg.set_value(self.b_slider_tag, self.b_val)
            dpg.set_value(self.w_slider_tag, self.w_val)
            dpg.set_value(self.ir_slider_tag, self.ir_val)

        self._update_pastille()
        self.send_current_channels()

    def serialize(self) -> Dict[str, Any]:
        """Custom serialization to preserve state."""
        data = super().serialize()
        data.update({
            "selected_port": self.selected_port,
            "selected_baudrate": self.selected_baudrate,
            "auto_send": self.auto_send,
            "r_val": self.r_val,
            "g_val": self.g_val,
            "b_val": self.b_val,
            "w_val": self.w_val,
            "ir_val": self.ir_val,
            "pulse_dur": self.pulse_dur,
            "pulse_r": self.pulse_r,
            "pulse_g": self.pulse_g,
            "pulse_b": self.pulse_b,
            "pulse_w": self.pulse_w,
            "pulse_ir": self.pulse_ir,
        })
        return data

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore state from serialized dictionary."""
        super().deserialize(data)
        self.selected_port = data.get("selected_port", "")
        self.selected_baudrate = data.get("selected_baudrate", 115200)
        self.auto_send = data.get("auto_send", True)

        self.r_val = data.get("r_val", 0)
        self.g_val = data.get("g_val", 0)
        self.b_val = data.get("b_val", 0)
        self.w_val = data.get("w_val", 0)
        self.ir_val = data.get("ir_val", 0)

        self.pulse_dur = data.get("pulse_dur", 100)
        self.pulse_r = data.get("pulse_r", 0)
        self.pulse_g = data.get("pulse_g", 0)
        self.pulse_b = data.get("pulse_b", 0)
        self.pulse_w = data.get("pulse_w", 0)
        self.pulse_ir = data.get("pulse_ir", 0)

        # Update UI items
        if dpg.does_item_exist(self.combo_port_tag) and self.selected_port:
            dpg.set_value(self.combo_port_tag, self.selected_port)
        if dpg.does_item_exist(self.combo_baud_tag):
            dpg.set_value(self.combo_baud_tag, str(self.selected_baudrate))
        if dpg.does_item_exist(self.auto_send_tag):
            dpg.set_value(self.auto_send_tag, self.auto_send)

        for tag, val in [
            (self.r_slider_tag, self.r_val),
            (self.g_slider_tag, self.g_val),
            (self.b_slider_tag, self.b_val),
            (self.w_slider_tag, self.w_val),
            (self.ir_slider_tag, self.ir_val),
            (self.pulse_dur_tag, self.pulse_dur),
            (self.pulse_r_tag, self.pulse_r),
            (self.pulse_g_tag, self.pulse_g),
            (self.pulse_b_tag, self.pulse_b),
            (self.pulse_w_tag, self.pulse_w),
            (self.pulse_ir_tag, self.pulse_ir),
        ]:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)

        self._update_pastille()
        self._update_pwm_displays()

    def on_close(self) -> None:
        """Closes serial connection upon window close."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        self.driver.disconnect()
        super().on_close()


EXPORTED_CLASS = LightControllerWin
EXPORTED_NAME = "Light Controller"
