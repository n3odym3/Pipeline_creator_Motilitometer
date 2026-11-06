from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

# Try importing vidgear
try:
    from vidgear.gears import WriteGear

    VIDGEAR_AVAILABLE = True
except ImportError:
    VIDGEAR_AVAILABLE = False


class VideoWriterEngine:
    """
    High-Performance Multi-Backend Video Writer Engine.

    Supports:
    - Backend 1: VidGear WriteGear (FFmpeg wrapper with hardware acceleration)
    - Backend 2: Direct FFmpeg Subprocess Pipe (zero Python GIL overhead, native high-depth)
    - Backend 3: OpenCV VideoWriter (Universal fallback)

    Image format & bitdepth handling:
    - 8-bit BGR / RGB / BGRA
    - 8-bit Grayscale (Mono)
    - 12-bit Mono / HDR (uint16, 0-4095)
    - 16-bit Mono / HDR (uint16, 0-65535)
    - Fast SIMD tone-mapping to 8-bit OR native high bit-depth encoding (FFV1 / ProRes / HEVC-10).
    """

    # Bit-depth conversion modes
    BITDEPTH_AUTOSCALE_8BIT = "8-bit Tone-mapped (Universal MP4)"
    BITDEPTH_NATIVE_HIGH = "Native Full Bit-Depth (FFV1 / ProRes / HEVC)"

    # Codec definitions
    CODEC_H264 = "H.264 / AVC (Fast & Universal)"
    CODEC_H264_NVENC = "H.264 NVIDIA NVENC (GPU)"
    CODEC_H264_QSV = "H.264 Intel QuickSync (GPU)"
    CODEC_H265 = "H.265 / HEVC (High Efficiency)"
    CODEC_H265_NVENC = "H.265 NVIDIA NVENC (GPU)"
    CODEC_MJPEG = "MJPEG (Low CPU / Intra-frame)"
    CODEC_PRORES = "Apple ProRes (Editing / 10-12 bit)"
    CODEC_FFV1 = "FFV1 (Lossless Scientific Archival)"
    CODEC_OPENCV_MP4V = "OpenCV MP4V"

    _FFMPEG_AVAILABLE: Optional[bool] = None

    def __init__(self) -> None:
        """Initialize encoder backend references and configuration state."""
        self.is_open: bool = False
        self.filepath: str = ""
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 30.0
        self.backend_type: str = "none"

        # Active instances
        self._vidgear_writer: Optional[Any] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._stderr_lines: List[str] = []
        self._stderr_thread: Optional[threading.Thread] = None
        self._cv2_writer: Optional[cv2.VideoWriter] = None

        # Settings
        self.codec: str = self.CODEC_H264
        self.preset: str = "veryfast"
        self.bitrate_mbps: float = 8.0
        self.bitdepth_mode: str = self.BITDEPTH_AUTOSCALE_8BIT
        self.hardware_accel: bool = False
        self.input_pix_fmt: str = "bgr24"
        self._uint16_shift: Optional[int] = None
        self._last_raw_frame_ref: Optional[np.ndarray] = None
        self._last_prepared_frame: Optional[np.ndarray] = None

    @staticmethod
    def is_ffmpeg_installed() -> bool:
        """Check if FFmpeg executable is present in PATH."""
        if VideoWriterEngine._FFMPEG_AVAILABLE is None:
            VideoWriterEngine._FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
        return VideoWriterEngine._FFMPEG_AVAILABLE

    @staticmethod
    def is_vidgear_installed() -> bool:
        """Check if VidGear library is installed."""
        return VIDGEAR_AVAILABLE

    def open(
        self,
        filepath: str,
        width: int,
        height: int,
        fps: float,
        codec: str = CODEC_H264,
        preset: str = "veryfast",
        bitrate_mbps: float = 8.0,
        bitdepth_mode: str = BITDEPTH_AUTOSCALE_8BIT,
        hardware_accel: bool = False,
        first_frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Initializes and opens the video encoder backend for writing.

        Args:
            filepath: Destination output video filepath.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Frame rate.
            codec: Codec identifier string.
            preset: Encoding preset.
            bitrate_mbps: Target video bitrate in Mbps.
            bitdepth_mode: Tone-mapping vs native high bitdepth.
            hardware_accel: Enable GPU acceleration if available.
            first_frame: Sample frame used for format inspection.

        Returns:
            True if an encoder was initialized successfully.
        """
        self.close()

        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = max(1.0, float(fps))
        self.codec = codec
        self.preset = preset
        self.bitrate_mbps = max(0.5, float(bitrate_mbps))
        self.bitdepth_mode = bitdepth_mode
        self.hardware_accel = hardware_accel
        self._uint16_shift = None
        self._last_raw_frame_ref = None
        self._last_prepared_frame = None

        # Ensure target directory exists
        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Detect input frame characteristics
        is_uint16 = False
        channels = 3
        if first_frame is not None:
            is_uint16 = first_frame.dtype == np.uint16
            channels = 1 if len(first_frame.shape) == 2 else first_frame.shape[2]
            if is_uint16 and self.bitdepth_mode == self.BITDEPTH_AUTOSCALE_8BIT:
                max_val = np.max(first_frame) if first_frame.size > 0 else 0
                self._uint16_shift = 4 if max_val <= 4095 else 8

        # 1. VidGear WriteGear backend
        if VIDGEAR_AVAILABLE:
            success = self._open_vidgear(is_uint16, channels)
            if success:
                self.is_open = True
                self.backend_type = "vidgear"
                logger.info(
                    f"VideoWriter initialized via VidGear WriteGear: {filepath} ({width}x{height} @ {fps}fps)"
                )
                return True

        # 2. Direct FFmpeg subprocess pipe backend
        if self.is_ffmpeg_installed():
            success = self._open_ffmpeg_subprocess(is_uint16, channels)
            if success:
                self.is_open = True
                self.backend_type = "ffmpeg_direct"
                logger.info(
                    f"VideoWriter initialized via Direct FFmpeg: {filepath} ({width}x{height} @ {fps}fps)"
                )
                return True

        # 3. OpenCV fallback backend
        success = self._open_opencv(is_uint16, channels)
        if success:
            self.is_open = True
            self.backend_type = "opencv"
            logger.info(f"VideoWriter initialized via OpenCV: {filepath} ({width}x{height} @ {fps}fps)")
            return True

        logger.error(f"Failed to initialize any video writer backend for {filepath}")
        return False

    def _open_vidgear(self, is_uint16: bool, channels: int) -> bool:
        """Initialize VidGear WriteGear backend."""
        try:
            output_params: Dict[str, Any] = {
                "-input_framerate": float(self.fps),
                "-r": str(int(self.fps)) if self.fps.is_integer() else str(self.fps),
                "-b:v": f"{int(self.bitrate_mbps * 1000)}k",
            }

            # Map codec to FFmpeg parameters
            if self.codec == self.CODEC_H264_NVENC:
                output_params["-vcodec"] = "h264_nvenc"
                output_params["-preset"] = "fast"
                output_params["-pix_fmt"] = "yuv420p"
            elif self.codec == self.CODEC_H264_QSV:
                output_params["-vcodec"] = "h264_qsv"
                output_params["-preset"] = "veryfast"
                output_params["-pix_fmt"] = "nv12"
            elif self.codec == self.CODEC_H265_NVENC:
                output_params["-vcodec"] = "hevc_nvenc"
                output_params["-preset"] = "fast"
                output_params["-pix_fmt"] = "yuv420p"
            elif self.codec == self.CODEC_H265:
                output_params["-vcodec"] = "libx265"
                output_params["-preset"] = self.preset
                output_params["-pix_fmt"] = "yuv420p"
            elif self.codec == self.CODEC_MJPEG:
                output_params["-vcodec"] = "mjpeg"
                output_params["-q:v"] = "2"
                output_params["-pix_fmt"] = "yuvj420p"
            elif self.codec == self.CODEC_PRORES:
                output_params["-vcodec"] = "prores_ks"
                output_params["-profile:v"] = "3"
                output_params["-pix_fmt"] = "yuv422p10le"
            elif self.codec == self.CODEC_FFV1:
                output_params["-vcodec"] = "ffv1"
                output_params["-level"] = "3"
                output_params["-pix_fmt"] = "yuv420p" if not is_uint16 else "gray16le"
            else:
                # Default H.264
                output_params["-vcodec"] = "libx264"
                output_params["-preset"] = self.preset
                output_params["-pix_fmt"] = "yuv420p"

            # Native high bitdepth adjustments
            if is_uint16 and self.bitdepth_mode == self.BITDEPTH_NATIVE_HIGH:
                if self.codec in [self.CODEC_FFV1, self.CODEC_PRORES]:
                    output_params["-pix_fmt"] = "gray16le" if channels == 1 else "yuv422p10le"
                elif self.codec in [self.CODEC_H265, self.CODEC_H265_NVENC]:
                    output_params["-pix_fmt"] = "yuv420p10le"

            self._vidgear_writer = WriteGear(
                output=self.filepath,
                compression_mode=True,
                logging=False,
                **output_params,
            )
            return True
        except Exception as e:
            logger.warning(f"VidGear WriteGear init failed: {e}. Falling back...")
            self._vidgear_writer = None
            return False

    def _open_ffmpeg_subprocess(self, is_uint16: bool, channels: int) -> bool:
        """Initialize direct FFmpeg pipe backend with background stderr monitor."""
        try:
            # Determine input pixel format
            if is_uint16 and self.bitdepth_mode == self.BITDEPTH_NATIVE_HIGH:
                in_pix_fmt = "gray16le" if channels == 1 else "bgr48le"
            elif channels == 1:
                in_pix_fmt = "gray"
            else:
                in_pix_fmt = "bgr24"

            self.input_pix_fmt = in_pix_fmt

            # Build FFmpeg command line
            fps_str = str(int(self.fps)) if self.fps.is_integer() else str(self.fps)
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",  # Suppress informational stats to prevent pipe buffer saturation
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.width}x{self.height}",
                "-pix_fmt",
                in_pix_fmt,
                "-framerate",
                fps_str,
                "-i",
                "-",
            ]

            # Codec settings
            if self.codec == self.CODEC_H264_NVENC:
                cmd.extend(
                    [
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        "p4",
                        "-b:v",
                        f"{int(self.bitrate_mbps * 1000)}k",
                        "-pix_fmt",
                        "yuv420p",
                    ]
                )
            elif self.codec == self.CODEC_H264_QSV:
                cmd.extend(
                    [
                        "-c:v",
                        "h264_qsv",
                        "-preset",
                        "veryfast",
                        "-b:v",
                        f"{int(self.bitrate_mbps * 1000)}k",
                        "-pix_fmt",
                        "nv12",
                    ]
                )
            elif self.codec == self.CODEC_H265_NVENC:
                cmd.extend(
                    [
                        "-c:v",
                        "hevc_nvenc",
                        "-preset",
                        "p4",
                        "-b:v",
                        f"{int(self.bitrate_mbps * 1000)}k",
                        "-pix_fmt",
                        "yuv420p",
                    ]
                )
            elif self.codec == self.CODEC_H265:
                cmd.extend(
                    [
                        "-c:v",
                        "libx265",
                        "-preset",
                        self.preset,
                        "-b:v",
                        f"{int(self.bitrate_mbps * 1000)}k",
                        "-pix_fmt",
                        "yuv420p",
                    ]
                )
            elif self.codec == self.CODEC_MJPEG:
                cmd.extend(["-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj420p"])
            elif self.codec == self.CODEC_PRORES:
                cmd.extend(["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"])
            elif self.codec == self.CODEC_FFV1:
                pix = (
                    "gray16le"
                    if (is_uint16 and self.bitdepth_mode == self.BITDEPTH_NATIVE_HIGH)
                    else "yuv420p"
                )
                cmd.extend(["-c:v", "ffv1", "-level", "3", "-pix_fmt", pix])
            else:
                # Default H.264
                cmd.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-preset",
                        self.preset,
                        "-b:v",
                        f"{int(self.bitrate_mbps * 1000)}k",
                        "-pix_fmt",
                        "yuv420p",
                    ]
                )

            # Native high bitdepth adjustments
            if is_uint16 and self.bitdepth_mode == self.BITDEPTH_NATIVE_HIGH:
                if self.codec in [self.CODEC_H265, self.CODEC_H265_NVENC]:
                    cmd.extend(["-pix_fmt", "yuv420p10le"])

            # Output framerate constraint
            cmd.extend(["-r", str(self.fps)])
            cmd.append(self.filepath)

            self._stderr_lines = []
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=10**7,
            )

            # Dedicated thread to continuously drain stderr and prevent pipe deadlocks
            def _drain_stderr() -> None:
                if self._ffmpeg_proc and self._ffmpeg_proc.stderr:
                    for line in iter(self._ffmpeg_proc.stderr.readline, b""):
                        if line:
                            decoded = line.decode(errors="ignore").strip()
                            self._stderr_lines.append(decoded)
                            if len(self._stderr_lines) > 50:
                                self._stderr_lines.pop(0)

            self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            self._stderr_thread.start()

            return True
        except Exception as e:
            logger.warning(f"Direct FFmpeg init failed: {e}. Falling back to OpenCV...")
            self._ffmpeg_proc = None
            return False

    def _open_opencv(self, is_uint16: bool, channels: int) -> bool:
        """Initialize OpenCV VideoWriter backend."""
        try:
            ext = os.path.splitext(self.filepath)[1].lower()
            if ext == ".avi":
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            elif ext in [".mp4", ".m4v"]:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            is_color = channels > 1
            self._cv2_writer = cv2.VideoWriter(
                self.filepath,
                fourcc,
                self.fps,
                (self.width, self.height),
                is_color,
            )

            if not self._cv2_writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._cv2_writer = cv2.VideoWriter(
                    self.filepath,
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                    is_color,
                )

            return self._cv2_writer.isOpened()
        except Exception as e:
            logger.error(f"OpenCV VideoWriter open failed: {e}")
            self._cv2_writer = None
            return False

    def write(self, frame: np.ndarray) -> bool:
        """
        Writes a frame to the active video encoder with automatic format & bitdepth adaptation.

        Args:
            frame: Input image array.

        Returns:
            True if written successfully.
        """
        if not self.is_open or frame is None:
            return False

        try:
            processed_frame = self._prepare_frame(frame)
            if processed_frame is None:
                return False

            if self.backend_type == "vidgear" and self._vidgear_writer is not None:
                self._vidgear_writer.write(processed_frame)
                return True

            elif self.backend_type == "ffmpeg_direct" and self._ffmpeg_proc is not None:
                if self._ffmpeg_proc.poll() is not None:
                    err_summary = "\n".join(self._stderr_lines[-5:]) if self._stderr_lines else "Unknown"
                    logger.error(f"FFmpeg process died unexpectedly: {err_summary}")
                    return False
                if self._ffmpeg_proc.stdin:
                    self._ffmpeg_proc.stdin.write(memoryview(np.ascontiguousarray(processed_frame)))
                return True

            elif self.backend_type == "opencv" and self._cv2_writer is not None:
                if processed_frame.dtype != np.uint8:
                    processed_frame = cv2.convertScaleAbs(processed_frame, alpha=1.0 / 256.0)
                self._cv2_writer.write(processed_frame)
                return True

            return False
        except Exception as e:
            logger.error(f"Error writing frame to video: {e}")
            return False

    def _prepare_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts/adapts frame dimensions and bitdepth according to configuration.

        Args:
            frame: Raw incoming image frame.

        Returns:
            Preprocessed NumPy array ready for encoder consumption.
        """
        if frame is self._last_raw_frame_ref and self._last_prepared_frame is not None:
            return self._last_prepared_frame

        raw_ref = frame
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        is_uint16 = frame.dtype == np.uint16

        # 1. 12-bit / 16-bit tone mapping
        if is_uint16:
            if self.bitdepth_mode == self.BITDEPTH_AUTOSCALE_8BIT:
                shift = self._uint16_shift
                if shift is None:
                    max_val = np.max(frame) if frame.size > 0 else 0
                    shift = 4 if max_val <= 4095 else 8
                    self._uint16_shift = shift
                frame = cv2.convertScaleAbs(frame, alpha=1.0 / (1 << shift))
            else:
                self._last_raw_frame_ref = raw_ref
                self._last_prepared_frame = frame
                return frame

        # 2. Channel normalization for 8-bit
        if len(frame.shape) == 2:
            # Grayscale 8-bit
            if self.backend_type == "opencv" and self._cv2_writer is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif self.backend_type == "vidgear":
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif (
                self.backend_type == "ffmpeg_direct"
                and self.input_pix_fmt != "gray"
                and self.input_pix_fmt != "gray16le"
            ):
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3:
            if frame.shape[2] == 4:
                # BGRA -> BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.shape[2] == 1:
                frame = frame.squeeze(2)
                if self.backend_type in ("opencv", "vidgear") or (
                    self.backend_type == "ffmpeg_direct"
                    and self.input_pix_fmt not in ("gray", "gray16le")
                ):
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        self._last_raw_frame_ref = raw_ref
        self._last_prepared_frame = frame
        return frame

    def close(self) -> None:
        """Closes and releases the active video encoder."""
        if not self.is_open:
            return

        try:
            if self._vidgear_writer is not None:
                self._vidgear_writer.close()
                self._vidgear_writer = None

            if self._ffmpeg_proc is not None:
                try:
                    if self._ffmpeg_proc.stdin:
                        try:
                            self._ffmpeg_proc.stdin.flush()
                            self._ffmpeg_proc.stdin.close()
                        except (OSError, BrokenPipeError):
                            pass
                    self._ffmpeg_proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._ffmpeg_proc.kill()
                    try:
                        self._ffmpeg_proc.wait(timeout=1.0)
                    except Exception:
                        pass
                if self._ffmpeg_proc.stderr:
                    try:
                        self._ffmpeg_proc.stderr.close()
                    except Exception:
                        pass
                self._ffmpeg_proc = None

            if self._cv2_writer is not None:
                self._cv2_writer.release()
                self._cv2_writer = None

            logger.info(f"VideoWriter closed and saved: {self.filepath}")
        except Exception as e:
            logger.error(f"Error closing video writer: {e}")
        finally:
            self.is_open = False
            self.backend_type = "none"
            self._uint16_shift = None
            self._last_raw_frame_ref = None
            self._last_prepared_frame = None

    def get_file_size_mb(self) -> float:
        """
        Returns current recorded file size in megabytes.

        Returns:
            File size in MB, or 0.0 if file doesn't exist.
        """
        if self.filepath and os.path.exists(self.filepath):
            try:
                return round(os.path.getsize(self.filepath) / (1024.0 * 1024.0), 2)
            except OSError:
                return 0.0
        return 0.0
