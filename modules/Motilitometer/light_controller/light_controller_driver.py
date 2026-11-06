from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from loguru import logger
import serial
import serial.tools.list_ports


DEFAULT_CALIBRATION: Dict[str, Any] = {
    "light": {
        "max": 500,
        "low_corr": 15,
        "low_m": 0.2,
        "low_p": -5,
        "m": 0.4975,
        "p": -48.897
    }
}

_BLACKOUT_CMD = '{"r": 0, "g": 0, "b": 0, "w": 0, "ir": 0}\n'


class LightControllerDriver:
    """
    Serial driver for the Motilitometer Light Controller.
    Manages serial connection, calibration conversions (µE -> PWM),
    and JSON formatted command exchange.
    """

    def __init__(self, calib_file_path: Optional[str] = None) -> None:
        self.ser: Optional[serial.Serial] = None
        self.port: Optional[str] = None
        self.baudrate: int = 115200
        self.lock = threading.Lock()
        self.is_connected: bool = False
        
        self.calibration: Dict[str, Any] = {"light": dict(DEFAULT_CALIBRATION["light"])}
        self._cal_low_corr: float = 15.0
        self._cal_low_m: float = 0.2
        self._cal_low_p: float = -5.0
        self._cal_m: float = 0.4975
        self._cal_p: float = -48.897
        self._sync_calibration_cache()

        self.calib_file_path = calib_file_path
        if calib_file_path and os.path.exists(calib_file_path):
            self.load_calibration(calib_file_path)

        self._read_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self.on_data_received: Optional[Callable[[str], None]] = None
        self.last_received: str = ""

    @staticmethod
    def list_com_ports() -> List[str]:
        """Returns list of available serial COM port device strings."""
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception as e:
            logger.error(f"[LightControllerDriver] Error listing COM ports: {e}")
            return []

    def _sync_calibration_cache(self) -> None:
        """Update cached float calibration parameters from self.calibration."""
        calib_cfg = self.calibration.get("light", DEFAULT_CALIBRATION["light"])
        self._cal_low_corr = float(calib_cfg.get("low_corr", 15))
        self._cal_low_m = float(calib_cfg.get("low_m", 0.2))
        self._cal_low_p = float(calib_cfg.get("low_p", -5))
        self._cal_m = float(calib_cfg.get("m", 0.4975))
        self._cal_p = float(calib_cfg.get("p", -48.897))

    def load_calibration(self, filepath: str) -> bool:
        """Loads calibration settings from a JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "light" in data:
                self.calibration = data
                self.calib_file_path = filepath
                self._sync_calibration_cache()
                logger.info(f"[LightControllerDriver] Calibration loaded successfully from {filepath}")
                return True
            else:
                logger.warning(f"[LightControllerDriver] Missing 'light' key in {filepath}")
                return False
        except Exception as e:
            logger.error(f"[LightControllerDriver] Error loading calibration from {filepath}: {e}")
            return False

    def save_calibration(self, filepath: Optional[str] = None) -> bool:
        """Saves current calibration settings to a JSON file."""
        self._sync_calibration_cache()
        path = filepath or self.calib_file_path
        if not path:
            return False
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.calibration, f, indent=4)
            logger.info(f"[LightControllerDriver] Calibration saved to {path}")
            return True
        except Exception as e:
            logger.error(f"[LightControllerDriver] Error saving calibration to {path}: {e}")
            return False

    def get_pwm_val(self, micro_e: float) -> int:
        """
        Converts microEinstein (µE) intensity to hardware PWM integer based on calibration curves.
        """
        if micro_e <= 0:
            return 0

        if micro_e <= self._cal_low_corr:
            m = self._cal_low_m
            p = self._cal_low_p
        else:
            m = self._cal_m
            p = self._cal_p

        if m == 0:
            return 0

        cmd = (micro_e - p) / m
        return max(0, int(round(cmd)))

    def connect(self, port: str, baudrate: int = 115200, timeout: float = 0.5) -> bool:
        """Connects to the specified serial port."""
        with self.lock:
            if self.is_connected:
                self._disconnect_unsafe()

            try:
                self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
                self.port = port
                self.baudrate = baudrate
                self.is_connected = True
                logger.success(f"[LightControllerDriver] Connected to {port} @ {baudrate} baud")

                self._running = True
                self._read_thread = threading.Thread(target=self._listen_serial, daemon=True)
                self._read_thread.start()
                return True
            except Exception as e:
                logger.error(f"[LightControllerDriver] Failed to connect to {port}: {e}")
                self.is_connected = False
                self.ser = None
                return False

    def disconnect(self) -> None:
        """Disconnects the serial device."""
        with self.lock:
            self._disconnect_unsafe()

    def _disconnect_unsafe(self) -> None:
        self._running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception as e:
                logger.error(f"[LightControllerDriver] Error closing port: {e}")
        self.ser = None
        self.is_connected = False
        logger.info("[LightControllerDriver] Serial disconnected")

    def _listen_serial(self) -> None:
        """Background thread reading lines / data chunks from serial."""
        while self._running:
            try:
                if self.ser and self.ser.is_open:
                    if self.ser.in_waiting > 0:
                        raw_data = self.ser.readline()
                        if raw_data:
                            text = raw_data.decode("utf-8", errors="replace").strip()
                            if text:
                                self.last_received = text
                                if self.on_data_received:
                                    try:
                                        self.on_data_received(text)
                                    except Exception as cb_err:
                                        logger.error(f"[LightControllerDriver] Callback error: {cb_err}")
                    else:
                        time.sleep(0.02)
                else:
                    break
            except Exception as e:
                if self._running:
                    logger.debug(f"[LightControllerDriver] Serial read loop notice: {e}")
                break

    def send_raw_command(self, cmd_str: str) -> bool:
        """Sends raw string over serial."""
        with self.lock:
            if not self.is_connected or not self.ser:
                logger.warning("[LightControllerDriver] Attempted to send while disconnected")
                return False

            try:
                if not cmd_str.endswith("\n"):
                    cmd_str += "\n"
                self.ser.write(cmd_str.encode("utf-8"))
                self.ser.flush()
                return True
            except Exception as e:
                logger.error(f"[LightControllerDriver] Error sending command '{cmd_str.strip()}': {e}")
                return False

    def send_light_command(self, r_ue: float, g_ue: float, b_ue: float, w_ue: float, ir_val: int) -> tuple[bool, Dict[str, int]]:
        """
        Calculates calibrated PWM for RGBW and sends JSON command to serial.
        Returns (success_bool, pwm_dict).
        """
        pwm_cmd = {
            "r": self.get_pwm_val(r_ue),
            "g": self.get_pwm_val(g_ue),
            "b": self.get_pwm_val(b_ue),
            "w": self.get_pwm_val(w_ue),
            "ir": int(max(0, ir_val))
        }
        json_str = json.dumps(pwm_cmd)
        success = self.send_raw_command(json_str)
        return success, pwm_cmd

    def send_pulse_command(self, duration_ms: int, r_ue: float, g_ue: float, b_ue: float, w_ue: float, ir_val: int = 0) -> tuple[bool, Dict[str, Any]]:
        """
        Calculates calibrated PWM for RGBW and sends pulse JSON command to serial.
        Returns (success_bool, pulse_cmd_dict).
        """
        pulse_cmd = {
            "pulse": int(duration_ms),
            "r": self.get_pwm_val(r_ue),
            "g": self.get_pwm_val(g_ue),
            "b": self.get_pwm_val(b_ue),
            "w": self.get_pwm_val(w_ue),
            "ir": int(max(0, ir_val))
        }
        json_str = json.dumps(pulse_cmd)
        success = self.send_raw_command(json_str)
        return success, pulse_cmd

    def blackout(self) -> bool:
        """Sends all channels at 0."""
        return self.send_raw_command(_BLACKOUT_CMD)
