
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.file_explorer import file_explorer
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from core.app_state import app_state
from .frame_rate_scheduler import FrameRateScheduler
from .video_writer_engine import VideoWriterEngine


class VideoWriter_win(WindowBase):
    """
    High-Performance Video Writer Module Window.
    
    Features:
    - Smart real-time frame rate synchronization (Drop & Duplicate).
    - Prevents UI freezes with asynchronous bounded frame queueing.
    - Multi-backend video encoding (VidGear WriteGear, FFmpeg direct pipe, OpenCV fallback).
    - Native support for standard and exotic formats (8-bit BGR/Gray, 12-bit/16-bit Mono/HDR uint16, BGRA).
    - Hardware acceleration support (NVENC, QuickSync, AMF).
    """

    label = "Video Writer"
    description = "High-performance video writer with real-time frame rate synchronization (Drop & Duplicate)."
    win_width = display_scaling.scale(340)
    win_height = display_scaling.scale(420)

    # I/O Specifications
    accepted_input_types = [
        IOTypes.FRAME,
        IOTypes.FRAME12,
        IOTypes.FRAME16,
        IOTypes.MASK,
        IOTypes.FOLDER_PATH,
        IOTypes.FILE_PATH,
        IOTypes.TRIGGER,
        IOTypes.CMD_DICT,
    ]

    outputs = {
        "Frame": IOTypes.FRAME,
        "File Path": IOTypes.FILE_PATH,
        "Trigger": IOTypes.TRIGGER,
        "Status": IOTypes.STATUS_DICT,
    }

    # Persistent fields saved into pipeline JSON
    persistent_fields = [
        "label",
        "target_fps",
        "codec",
        "container",
        "preset",
        "bitrate_mbps",
        "sync_mode",
        "bitdepth_mode",
        "output_dir",
        "file_prefix",
    ]

    # Persistent field defaults
    target_fps: int = 30
    codec: str = VideoWriterEngine.CODEC_H264
    container: str = ".mp4"
    preset: str = "veryfast"
    bitrate_mbps: float = 8.0
    sync_mode: str = FrameRateScheduler.MODE_REALTIME
    bitdepth_mode: str = VideoWriterEngine.BITDEPTH_AUTOSCALE_8BIT
    output_dir: str = ""
    file_prefix: str = "recording"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Defensive extraction of persistent fields
        try:
            self.target_fps = max(1, int(getattr(self, "target_fps", kwargs.get("target_fps", 30)) or 30))
        except (ValueError, TypeError):
            self.target_fps = 30
        self.codec = str(getattr(self, "codec", kwargs.get("codec", VideoWriterEngine.CODEC_H264)) or VideoWriterEngine.CODEC_H264)
        self.container = str(getattr(self, "container", kwargs.get("container", ".mp4")) or ".mp4")
        self.preset = str(getattr(self, "preset", kwargs.get("preset", "veryfast")) or "veryfast")
        try:
            self.bitrate_mbps = float(getattr(self, "bitrate_mbps", kwargs.get("bitrate_mbps", 8.0)) or 8.0)
        except (ValueError, TypeError):
            self.bitrate_mbps = 8.0
        self.sync_mode = str(getattr(self, "sync_mode", kwargs.get("sync_mode", FrameRateScheduler.MODE_REALTIME)) or FrameRateScheduler.MODE_REALTIME)
        self.bitdepth_mode = str(getattr(self, "bitdepth_mode", kwargs.get("bitdepth_mode", VideoWriterEngine.BITDEPTH_AUTOSCALE_8BIT)) or VideoWriterEngine.BITDEPTH_AUTOSCALE_8BIT)
        folder_init = getattr(self, "output_dir", kwargs.get("output_dir", "")) or kwargs.get("folder", "") or kwargs.get("folder_path", "")
        self.output_dir = str(folder_init or os.path.join(os.path.expanduser("~"), "Videos"))
        self.file_prefix = str(getattr(self, "file_prefix", kwargs.get("file_prefix", "recording")) or "recording")

        # Core Engines
        self.engine = VideoWriterEngine()
        self.scheduler = FrameRateScheduler(target_fps=self.target_fps, mode=self.sync_mode)

        # Asynchronous Queue & Threading
        self.frame_queue: queue.Queue = queue.Queue(maxsize=120)
        self.is_recording: bool = False
        self.is_paused: bool = False
        self._is_closing: bool = False
        self.timed_record_duration: Optional[float] = None
        self.worker_thread: Optional[threading.Thread] = None
        self._stop_requested: threading.Event = threading.Event()
        self.current_output_file: str = ""

        # UI Element Tags
        self.winID = f"video_writer_win_{self.UUID}"
        self.status_text_tag = f"vw_status_{self.UUID}"
        self.rec_btn_tag = f"vw_rec_btn_{self.UUID}"
        self.pause_btn_tag = f"vw_pause_btn_{self.UUID}"
        self.stop_btn_tag = f"vw_stop_btn_{self.UUID}"
        self.fps_input_tag = f"vw_fps_{self.UUID}"
        self.codec_combo_tag = f"vw_codec_{self.UUID}"
        self.container_combo_tag = f"vw_container_{self.UUID}"
        self.bitrate_slider_tag = f"vw_bitrate_{self.UUID}"
        self.sync_combo_tag = f"vw_sync_{self.UUID}"
        self.bitdepth_combo_tag = f"vw_bitdepth_{self.UUID}"
        self.folder_text_tag = f"vw_folder_{self.UUID}"
        self.prefix_input_tag = f"vw_prefix_{self.UUID}"
        self.stats_group_tag = f"vw_stats_group_{self.UUID}"
        self.stat_fps_tag = f"vw_stat_fps_{self.UUID}"
        self.stat_duration_tag = f"vw_stat_duration_{self.UUID}"
        self.stat_frames_tag = f"vw_stat_frames_{self.UUID}"
        self.stat_drops_tag = f"vw_stat_drops_{self.UUID}"
        self.stat_size_tag = f"vw_stat_size_{self.UUID}"

        # Codec & Container Options
        self.codec_options = [
            VideoWriterEngine.CODEC_H264,
            VideoWriterEngine.CODEC_H264_NVENC,
            VideoWriterEngine.CODEC_H264_QSV,
            VideoWriterEngine.CODEC_H265,
            VideoWriterEngine.CODEC_H265_NVENC,
            VideoWriterEngine.CODEC_MJPEG,
            VideoWriterEngine.CODEC_PRORES,
            VideoWriterEngine.CODEC_FFV1,
            VideoWriterEngine.CODEC_OPENCV_MP4V,
        ]
        self.container_options = [".mp4", ".mkv", ".avi", ".mov"]
        self.sync_options = [
            FrameRateScheduler.MODE_REALTIME,
            FrameRateScheduler.MODE_PASSTHROUGH,
        ]
        self.bitdepth_options = [
            VideoWriterEngine.BITDEPTH_AUTOSCALE_8BIT,
            VideoWriterEngine.BITDEPTH_NATIVE_HIGH,
        ]

        # UI Construction
        scale = display_scaling.scale
        with self.window():
            # Recording Controls Banner
            with dpg.group(horizontal=True):
                dpg.add_button(label=" \uf04b REC ", tag=self.rec_btn_tag, callback=self.start_recording)
                dpg.add_button(label=" \uf04c ", tag=self.pause_btn_tag, callback=self.toggle_pause)
                dpg.add_button(label=" \uf04d STOP ", tag=self.stop_btn_tag, callback=self.stop_recording)
                dpg.add_text("  IDLE", tag=self.status_text_tag, color=[180, 180, 180])

            dpg.add_separator()

            # Output Destination
            with dpg.group(horizontal=True):
                dpg.add_button(label="Output Dir", callback=self._on_browse_folder)
                display_path = self._format_path(self.output_dir)
                dpg.add_text(display_path, tag=self.folder_text_tag)

            dpg.add_input_text(
                label="Prefix",
                default_value=self.file_prefix,
                tag=self.prefix_input_tag,
                callback=lambda s, v: setattr(self, "file_prefix", v),
            )

            # Video Parameters Header
            with dpg.collapsing_header(label="Encoding Settings", default_open=True):
                dpg.add_input_int(
                    label="Target FPS",
                    tag=self.fps_input_tag,
                    default_value=self.target_fps,
                    min_value=1,
                    max_value=240,
                    min_clamped=True,
                    max_clamped=True,
                    callback=self._on_fps_changed,
                )

                dpg.add_combo(
                    label="Sync Mode",
                    items=self.sync_options,
                    default_value=self.sync_mode,
                    tag=self.sync_combo_tag,
                    callback=lambda s, v: setattr(self, "sync_mode", v),
                )

                dpg.add_combo(
                    label="Codec",
                    items=self.codec_options,
                    default_value=self.codec if self.codec in self.codec_options else self.codec_options[0],
                    tag=self.codec_combo_tag,
                    callback=lambda s, v: setattr(self, "codec", v),
                )

                dpg.add_combo(
                    label="Container",
                    items=self.container_options,
                    default_value=self.container if self.container in self.container_options else ".mp4",
                    tag=self.container_combo_tag,
                    callback=lambda s, v: setattr(self, "container", v),
                )

                dpg.add_combo(
                    label="Bit-Depth Mode",
                    items=self.bitdepth_options,
                    default_value=self.bitdepth_mode,
                    tag=self.bitdepth_combo_tag,
                    callback=lambda s, v: setattr(self, "bitdepth_mode", v),
                )

                dpg.add_slider_float(
                    label="Bitrate (Mbps)",
                    tag=self.bitrate_slider_tag,
                    default_value=self.bitrate_mbps,
                    min_value=1.0,
                    max_value=50.0,
                    format="%.1f Mbps",
                    callback=lambda s, v: setattr(self, "bitrate_mbps", v),
                )

            # Live Stats & Telemetry
            with dpg.collapsing_header(label="Live Telemetry", default_open=True, tag=self.stats_group_tag):
                dpg.add_text("Ingest: 0.0 FPS | Output: 0.0 FPS", tag=self.stat_fps_tag)
                dpg.add_text("Duration: 00:00:00 (Real: 0.0s)", tag=self.stat_duration_tag)
                dpg.add_text("Frames: 0 written (0 received)", tag=self.stat_frames_tag)
                dpg.add_text("Dropped: 0 | Duplicated: 0", tag=self.stat_drops_tag)
                dpg.add_text("File Size: 0.00 MB", tag=self.stat_size_tag)

        # Apply icon font for FontAwesome glyphs
        try:
            from config.theme_manager import theme_manager

            theme_manager.apply_icon_font(self.rec_btn_tag)
            theme_manager.apply_icon_font(self.pause_btn_tag)
            theme_manager.apply_icon_font(self.stop_btn_tag)
        except Exception as e:
            logger.debug(f"Could not apply icon font in {self.winID}: {e}")

        self.update_permission()

    def update_permission(self) -> None:
        """Adapts UI according to application mode."""
        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _on_fps_changed(self, sender: Any = None, value: Any = None, *args: Any, **kwargs: Any) -> None:
        try:
            val = int(value if value is not None else (dpg.get_value(self.fps_input_tag) if dpg.does_item_exist(self.fps_input_tag) else self.target_fps))
        except (ValueError, TypeError):
            val = 30
        self.target_fps = max(1, val)
        self.scheduler.target_fps = self.target_fps
        self.scheduler.slot_duration = 1.0 / self.target_fps

    def _on_browse_folder(self, *args: Any, **kwargs: Any) -> None:
        file_explorer.select_folder(
            default_path=self.output_dir if os.path.isdir(self.output_dir) else "",
            callback=self._on_folder_selected,
        )

    def set_output_dir(self, folder_path: str) -> bool:
        """Sets output working directory for recorded videos, creating it if needed."""
        if not folder_path:
            return False
        folder_str = str(folder_path).strip("'\"")
        try:
            os.makedirs(folder_str, exist_ok=True)
        except Exception as e:
            logger.warning(f"VideoWriter: Could not create output directory '{folder_str}': {e}")

        self.output_dir = os.path.abspath(folder_str)
        if dpg.does_item_exist(self.folder_text_tag):
            dpg.set_value(self.folder_text_tag, self._format_path(self.output_dir))
        logger.info(f"VideoWriter output directory set to: {self.output_dir}")
        return True

    def _on_folder_selected(self, folder_path: str, *args: Any, **kwargs: Any) -> None:
        if folder_path:
            self.set_output_dir(folder_path)

    def _format_path(self, path: str, max_chars: int = 24) -> str:
        if not path:
            return "No folder selected"
        return path if len(path) <= max_chars else f"...{path[-(max_chars - 3):]}"

    def _generate_filepath(self, base_name: Optional[str] = None) -> str:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        prefix = base_name if base_name else self.file_prefix
        if not prefix:
            prefix = "recording"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ext = self.container if self.container.startswith(".") else f".{self.container}"
        filename = f"{prefix}_{timestamp}{ext}"
        return os.path.join(self.output_dir, filename)

    def start_recording(self, *args: Any, duration: Optional[float] = None, **kwargs: Any) -> None:
        """Starts video recording worker thread with optional automatic duration limit in seconds."""
        # If already recording, stop first to finalize previous file cleanly
        if self.is_recording:
            self.stop_recording()
            time.sleep(0.05)

        # Check if duration passed in kwargs or args
        if duration is not None:
            self.timed_record_duration = float(duration) if float(duration) > 0 else None
        elif args and isinstance(args[0], (int, float)) and args[0] > 0:
            self.timed_record_duration = float(args[0])
        else:
            self.timed_record_duration = None

        self.is_recording = True
        self.is_paused = False
        self._stop_requested.clear()

        # Clear existing queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        # Generate destination path
        self.current_output_file = self._generate_filepath()
        self.scheduler.reset(target_fps=self.target_fps, mode=self.sync_mode)

        # Update UI state
        dur_hint = f" ({self.timed_record_duration:.0f}s)" if self.timed_record_duration else ""
        dpg.set_value(self.status_text_tag, f" \u25cf RECORDING{dur_hint}")
        dpg.configure_item(self.status_text_tag, color=[255, 60, 60])

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._recording_worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info(f"Video recording requested: {self.current_output_file} (Target FPS: {self.target_fps}, Max duration: {self.timed_record_duration})")

    def toggle_pause(self, *args: Any, **kwargs: Any) -> None:
        """Toggles pause/resume state of recording."""
        if not self.is_recording:
            return

        self.is_paused = not self.is_paused
        if self.is_paused:
            dpg.set_value(self.status_text_tag, " \u2016 PAUSED")
            dpg.configure_item(self.status_text_tag, color=[255, 200, 50])
            self._trigger_cb(event="RECORD_PAUSE")
        else:
            dpg.set_value(self.status_text_tag, " \u25cf RECORDING")
            dpg.configure_item(self.status_text_tag, color=[255, 60, 60])
            self._trigger_cb(event="RECORD_RESUME")

    def stop_recording(self, *args: Any, **kwargs: Any) -> None:
        """Stops video recording and finalizes output file."""
        if not self.is_recording:
            return

        self.is_recording = False
        self.is_paused = False
        self.timed_record_duration = None
        self._stop_requested.set()

        if self.worker_thread and self.worker_thread.is_alive() and threading.current_thread() != self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        self.worker_thread = None

        self.engine.close()

        # Update UI state
        dpg.set_value(self.status_text_tag, "  IDLE")
        dpg.configure_item(self.status_text_tag, color=[180, 180, 180])

        # Send outputs
        if self.current_output_file and os.path.exists(self.current_output_file):
            self._emit_file_path(self.current_output_file)

        self._trigger_cb(event="RECORD_STOP")
        # Emit READY signal for sequence synchronizer
        self._trigger_cb(event="READY")
        logger.success(f"Video recording saved: {self.current_output_file}")

    def _recording_worker_loop(self) -> None:
        """Dedicated background recording thread consuming frames from queue."""
        last_ui_update = time.perf_counter()
        timed_record_start = None

        while not self._stop_requested.is_set():
            now = time.perf_counter()

            # Check if timed recording duration has elapsed (handles slow/waiting queue)
            if self.timed_record_duration is not None and timed_record_start is not None:
                elapsed_timed = now - timed_record_start
                if elapsed_timed >= self.timed_record_duration:
                    logger.info(f"Timed recording duration reached (timer): {elapsed_timed:.2f}s / {self.timed_record_duration}s")
                    break

            try:
                item = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                self._update_telemetry_ui(now)
                continue

            if item is None:
                break

            frame, frame_ts, metadata = item

            # If paused, skip processing to pause video
            if self.is_paused:
                continue

            # Lazy-open video engine on first valid frame
            if not self.engine.is_open:
                h, w = frame.shape[:2]
                success = self.engine.open(
                    filepath=self.current_output_file,
                    width=w,
                    height=h,
                    fps=float(self.target_fps),
                    codec=self.codec,
                    preset=self.preset,
                    bitrate_mbps=self.bitrate_mbps,
                    bitdepth_mode=self.bitdepth_mode,
                    first_frame=frame,
                )
                if not success:
                    logger.error(f"Failed to open video writer for {self.current_output_file}")
                    break
                timed_record_start = time.perf_counter()
                self._trigger_cb(event="RECORD_START")
                self._trigger_cb(event="RECORD_STARTED")
                self._trigger_cb(event="READY")
                logger.info(f"Video recording started and ready: {self.current_output_file}")

            # Schedule frames (handles drop/duplication)
            frames_to_write = self.scheduler.process_frame(frame=frame, timestamp=frame_ts, metadata=metadata)

            for fw, _ in frames_to_write:
                self.engine.write(fw)

            # Update telemetry UI throttled at ~4Hz
            if now - last_ui_update >= 0.25:
                last_ui_update = now
                self._update_telemetry_ui(now)

            # Check if timed recording duration has elapsed
            if self.timed_record_duration is not None:
                vid_dur = self.scheduler.frames_written / self.scheduler.target_fps
                if vid_dur >= self.timed_record_duration:
                    logger.info(f"Timed recording duration reached: {vid_dur:.1f}s / {self.timed_record_duration}s")
                    break

        # Process any remaining frames in queue before finishing
        while not self.frame_queue.empty():
            try:
                item = self.frame_queue.get_nowait()
                if item is not None and not self.is_paused and self.engine.is_open:
                    frame, frame_ts, metadata = item
                    frames_to_write = self.scheduler.process_frame(frame=frame, timestamp=frame_ts, metadata=metadata)
                    for fw, _ in frames_to_write:
                        self.engine.write(fw)
            except queue.Empty:
                break

        self.engine.close()
        self.is_recording = False
        self.timed_record_duration = None

        # Update UI state
        if dpg.does_item_exist(self.status_text_tag):
            dpg.set_value(self.status_text_tag, "  IDLE")
            dpg.configure_item(self.status_text_tag, color=[180, 180, 180])

        if self.current_output_file and os.path.exists(self.current_output_file):
            self._emit_file_path(self.current_output_file)

        self._trigger_cb(event="RECORD_STOP")
        self._trigger_cb(event="READY")
        self._update_telemetry_ui(time.perf_counter(), force=True)

    def _update_telemetry_ui(self, now: float, force: bool = False) -> None:
        """Refreshes live statistics labels in the DearPyGui window."""
        stats = self.scheduler.get_stats()
        file_size_mb = self.engine.get_file_size_mb()

        # Format video duration into HH:MM:SS
        dur_sec = int(stats["video_duration_sec"])
        hh = dur_sec // 3600
        mm = (dur_sec % 3600) // 60
        ss = dur_sec % 60
        dur_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        if dpg.does_item_exist(self.stat_fps_tag):
            dpg.set_value(self.stat_fps_tag, f"Ingest: {stats['input_fps']:.1f} FPS | Output: {stats['output_fps']:.1f} FPS")

        if dpg.does_item_exist(self.stat_duration_tag):
            dur_target = f" / {self.timed_record_duration:.0f}s" if self.timed_record_duration else ""
            dpg.set_value(self.stat_duration_tag, f"Duration: {dur_str}{dur_target} (Real: {stats['elapsed_sec']:.1f}s)")

        if dpg.does_item_exist(self.stat_frames_tag):
            dpg.set_value(self.stat_frames_tag, f"Frames: {stats['frames_written']} written ({stats['frames_received']} in)")

        if dpg.does_item_exist(self.stat_drops_tag):
            dpg.set_value(self.stat_drops_tag, f"Dropped: {stats['frames_dropped']} | Duplicated: {stats['frames_duplicated']}")

        if dpg.does_item_exist(self.stat_size_tag):
            dpg.set_value(self.stat_size_tag, f"File Size: {file_size_mb:.2f} MB")

        # Emit status dictionary to downstream connections
        if self.connections.get("Status"):
            self._emit_status({**stats, "file_size_mb": file_size_mb, "is_recording": self.is_recording})

    def _enqueue_and_emit_frame(self, frame: np.ndarray, metadata: Any = None) -> None:
        """Pass-through frame downstream and enqueue for background video recording."""
        self._emit_frame(frame, metadata)
        if self.is_recording and not self.is_paused:
            ts = time.perf_counter()
            try:
                # Fast non-blocking push (< 0.05ms)
                self.frame_queue.put_nowait((frame, ts, metadata))
            except queue.Full:
                # If buffer is congested, drop oldest frame to prevent RAM bloat
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait((frame, ts, metadata))

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Receives data from upstream nodes (Frame, Folder, Trigger, Commands).
        Non-blocking ingestion into thread-safe frame queue with passthrough output.
        """
        data = kwargs.get("data") if kwargs.get("data") is not None else (args[0] if args else None)

        # Direct keyword arguments for folder or folder_path
        direct_folder = kwargs.get("folder_path") or kwargs.get("folder") or kwargs.get("output_dir") or kwargs.get("path")
        if direct_folder and isinstance(direct_folder, str):
            if self.set_output_dir(direct_folder):
                return

        # Fast-path for incoming frames (most frequent call path)
        if isinstance(data, np.ndarray):
            self._enqueue_and_emit_frame(data, None)
            return
        if isinstance(data, tuple) and len(data) >= 2 and isinstance(data[0], np.ndarray):
            self._enqueue_and_emit_frame(data[0], data[1])
            return

        data_type = kwargs.get("data_type")

        # 1. Handle commands
        if isinstance(data, dict) or data_type == IOTypes.CMD_DICT:
            if isinstance(data, dict) and "image" in data and isinstance(data["image"], np.ndarray):
                self._enqueue_and_emit_frame(data["image"], data.get("filename"))
                return
            cmd = data if isinstance(data, dict) else {}
            if "prefix" in cmd or "group" in cmd or "name" in cmd or "animation" in cmd:
                prefix_val = cmd.get("prefix") or cmd.get("group") or cmd.get("name") or cmd.get("animation")
                if prefix_val:
                    self.file_prefix = str(prefix_val)
                    if dpg.does_item_exist(self.prefix_input_tag):
                        dpg.set_value(self.prefix_input_tag, self.file_prefix)

            cmd_folder = cmd.get("folder") or cmd.get("folder_path") or cmd.get("output_dir") or cmd.get("path") or cmd.get("working_dir")
            if cmd_folder and isinstance(cmd_folder, str):
                self.set_output_dir(cmd_folder)

            ev = str(cmd.get("event", "")).upper()
            if cmd.get("start") or cmd.get("record_video") or ev in ("START", "RECORD", "RECORD_START") or ("duration" in cmd and cmd.get("record")):
                dur = cmd.get("duration") or cmd.get("record_for") or cmd.get("time")
                self.start_recording(duration=dur)
            elif cmd.get("stop") or cmd.get("stop_video") or ev in ("STOP", "RECORD_STOP"):
                self.stop_recording()
            elif cmd.get("pause") or ev == "PAUSE":
                self.toggle_pause()
            elif "fps" in cmd:
                self._on_fps_changed(None, cmd["fps"])
            return

        # 1.5 Handle Trigger signals (e.g. START, STOP, RECORD)
        if data_type == IOTypes.TRIGGER or (isinstance(data, str) and data.upper() in ("START", "STOP", "RECORD", "RECORD_START", "RECORD_STOP", "PAUSE")):
            ev = str(data).strip().upper()
            if ev in ("START", "RECORD", "RECORD_START"):
                self.start_recording()
                return
            elif ev in ("STOP", "RECORD_STOP"):
                self.stop_recording()
                return
            elif ev == "PAUSE":
                self.toggle_pause()
                return

        # 2. Handle folder path
        if data_type == IOTypes.FOLDER_PATH:
            target_dir = None
            if isinstance(data, str):
                target_dir = data
            elif isinstance(data, (tuple, list)) and len(data) > 0 and isinstance(data[0], str):
                target_dir = data[0]
            elif isinstance(data, dict):
                target_dir = data.get("folder_path") or data.get("folder") or data.get("output_dir") or data.get("path")
            if target_dir:
                if self.set_output_dir(target_dir):
                    return

        # String or tuple folder path
        if isinstance(data, str):
            if os.path.isdir(data) or (data_type == IOTypes.FOLDER_PATH):
                if self.set_output_dir(data):
                    return

        if isinstance(data, (tuple, list)) and len(data) > 0 and isinstance(data[0], str):
            if os.path.isdir(data[0]):
                if self.set_output_dir(data[0]):
                    return

    def _emit_frame(self, frame: np.ndarray, metadata: Any = None) -> None:
        """Re-emits incoming frame downstream to connected modules (Pass-through)."""
        targets = self.connections.get("Frame")
        if not targets:
            return
        payload = (frame, metadata) if metadata is not None else frame
        for module in targets:
            try:
                module.input_cb(data=payload, data_type=IOTypes.FRAME)
            except Exception as e:
                logger.error(f"VideoWriter emit frame error: {e}")

    def _emit_file_path(self, filepath: str) -> None:
        """Sends saved video filepath to downstream connected modules."""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        payload = (filepath, basename)
        for module in self.connections.get("File Path", []):
            try:
                module.input_cb(data=payload, data_type=IOTypes.FILE_PATH)
            except Exception as e:
                logger.error(f"VideoWriter emit file path error: {e}")

    def _trigger_cb(self, event: str) -> None:
        """Emits event trigger to downstream modules."""
        for module in self.connections.get("Trigger", []):
            try:
                module.input_cb(data=event, data_type=IOTypes.TRIGGER)
            except Exception as e:
                logger.error(f"VideoWriter trigger emit error: {e}")

    def _emit_status(self, status: dict) -> None:
        """Emits live status dictionary to downstream modules."""
        for module in self.connections.get("Status", []):
            try:
                module.input_cb(data=status, data_type=IOTypes.STATUS_DICT)
            except Exception as e:
                pass

    def is_ready(self) -> bool:
        """Indicate whether the video writer module is ready."""
        return True

    def on_close(self) -> None:
        """Clean resource release upon window closure."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        self.stop_recording()
        super().on_close()

    def close(self) -> None:
        """Close window and unregister."""
        self.on_close()
        super().close()

    def destroy(self) -> None:
        """Destroy window and release resources."""
        self.close()
        super().destroy()


VideoWriterWin = VideoWriter_win
EXPORTED_CLASS = VideoWriter_win
EXPORTED_NAME = "Video Writer"
