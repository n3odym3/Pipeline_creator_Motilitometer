from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class FrameRateScheduler:
    """
    Intelligent Frame Rate Scheduler for Video Recording.

    Ensures that recorded video speed and duration match real-world elapsed time (1:1 sync),
    even when input frame rate differs from target output framerate (e.g. 90 FPS camera recorded
    at 30 FPS, or 24 FPS pipeline recorded at 30 FPS, or pipeline jitter/lag).

    Modes:
        - REALTIME (default): Intelligently drops frames if input FPS > target FPS,
          and duplicates frames if input FPS < target FPS or if processing lags.
        - PASSTHROUGH: Writes every incoming frame sequentially without dropping or duplicating.
    """

    MODE_REALTIME = "Realtime (Drop / Duplicate)"
    MODE_PASSTHROUGH = "Pass-through (Sequential 1:1)"

    def __init__(self, target_fps: float = 30.0, mode: str = MODE_REALTIME) -> None:
        """
        Initialize the FrameRateScheduler.

        Args:
            target_fps: Target video encoding framerate.
            mode: Synchronization mode (REALTIME or PASSTHROUGH).
        """
        self._lock = threading.Lock()
        self.target_fps: float = max(1.0, float(target_fps))
        self.slot_duration: float = 1.0 / self.target_fps
        self.mode: str = mode

        self.start_time: Optional[float] = None
        self.last_slot_index: int = -1
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_metadata: Any = None

        # Statistics
        self.frames_received: int = 0
        self.frames_written: int = 0
        self.frames_dropped: int = 0
        self.frames_duplicated: int = 0

        # Rolling timestamp windows
        self._input_timestamps: deque[float] = deque(maxlen=300)
        self._output_timestamps: deque[float] = deque(maxlen=300)
        self._last_stats_calc: float = 0.0
        self._cached_input_fps: float = 0.0
        self._cached_output_fps: float = 0.0

    def reset(self, target_fps: Optional[float] = None, mode: Optional[str] = None) -> None:
        """
        Reset the scheduler state for a new recording session.

        Args:
            target_fps: Optional new target framerate.
            mode: Optional new synchronization mode.
        """
        with self._lock:
            if target_fps is not None:
                self.target_fps = max(1.0, float(target_fps))
                self.slot_duration = 1.0 / self.target_fps
            if mode is not None:
                self.mode = mode

            self.start_time = None
            self.last_slot_index = -1
            self.last_frame = None
            self.last_frame_metadata = None

            self.frames_received = 0
            self.frames_written = 0
            self.frames_dropped = 0
            self.frames_duplicated = 0

            self._input_timestamps.clear()
            self._output_timestamps.clear()
            self._last_stats_calc = 0.0
            self._cached_input_fps = 0.0
            self._cached_output_fps = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        metadata: Any = None,
    ) -> List[Tuple[np.ndarray, Any]]:
        """
        Processes an incoming frame and returns a list of (frame, metadata) tuples
        that should be written to the video encoder for the current time step.

        Args:
            frame: Incoming image array.
            timestamp: Optional high-precision timestamp (defaults to time.perf_counter()).
            metadata: Optional attached frame name or metadata.

        Returns:
            List of (frame, metadata) tuples to write (0, 1, or multiple in case of duplication).
        """
        now = time.perf_counter() if timestamp is None else timestamp

        with self._lock:
            self.frames_received += 1
            self._record_input_timestamp(now)

            # Mode 1: Passthrough mode (no timing modification)
            if self.mode == self.MODE_PASSTHROUGH:
                self.frames_written += 1
                self._record_output_timestamp(now)
                self.last_frame = frame
                self.last_frame_metadata = metadata
                return [(frame, metadata)]

            # Mode 2: Real-time time-slot synchronization
            if self.start_time is None:
                self.start_time = now
                self.last_slot_index = 0
                self.last_frame = frame
                self.last_frame_metadata = metadata
                self.frames_written += 1
                self._record_output_timestamp(now)
                return [(frame, metadata)]

            # Calculate current time slot index based on elapsed real-world time
            elapsed = max(0.0, now - self.start_time)
            current_slot = int(elapsed * self.target_fps)

            frames_to_write: List[Tuple[np.ndarray, Any]] = []

            if current_slot <= self.last_slot_index:
                # We are still in the same time slot (Input FPS > Target FPS) -> Drop intermediate frame
                self.frames_dropped += 1
                self.last_frame = frame
                self.last_frame_metadata = metadata
                return []

            elif current_slot == self.last_slot_index + 1:
                # Perfect progression to the next adjacent slot
                frames_to_write.append((frame, metadata))
                self.frames_written += 1
                self._record_output_timestamp(now)
                self.last_slot_index = current_slot
                self.last_frame = frame
                self.last_frame_metadata = metadata
                return frames_to_write

            else:
                # Time skipped ahead by multiple slots (Input FPS < Target FPS or lag)
                # Duplicate previous frame for missing slots to preserve exact 1:1 elapsed time.
                missing_slots = current_slot - self.last_slot_index - 1
                if self.last_frame is not None and missing_slots > 0:
                    # Cap duplication in case of extreme lag (max 60 seconds of duplicate frames)
                    max_dupes = int(self.target_fps * 60)
                    dupes_count = min(missing_slots, max_dupes)
                    for _ in range(dupes_count):
                        frames_to_write.append((self.last_frame, self.last_frame_metadata))

                    self.frames_duplicated += dupes_count
                    self.frames_written += dupes_count
                    self._record_output_timestamp(now)

                # Write the new current frame
                frames_to_write.append((frame, metadata))
                self.frames_written += 1
                self._record_output_timestamp(now)
                self.last_slot_index = current_slot
                self.last_frame = frame
                self.last_frame_metadata = metadata
                return frames_to_write

    def flush(self) -> List[Tuple[np.ndarray, Any]]:
        """
        Called when recording stops to perform any final flush operations.

        Returns:
            List of trailing frames to write if any.
        """
        return []

    def _record_input_timestamp(self, ts: float) -> None:
        """Record input timestamp and maintain rolling window."""
        self._input_timestamps.append(ts)
        cutoff = ts - 2.0
        while self._input_timestamps and self._input_timestamps[0] < cutoff:
            self._input_timestamps.popleft()

    def _record_output_timestamp(self, ts: float) -> None:
        """Record output timestamp and maintain rolling window."""
        self._output_timestamps.append(ts)
        cutoff = ts - 2.0
        while self._output_timestamps and self._output_timestamps[0] < cutoff:
            self._output_timestamps.popleft()

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns live telemetry statistics regarding FPS, frame counts, and duration.

        Returns:
            Dictionary containing metrics.
        """
        now = time.perf_counter()

        with self._lock:
            if now - self._last_stats_calc >= 0.25:
                self._last_stats_calc = now
                if len(self._input_timestamps) > 1:
                    dt = self._input_timestamps[-1] - self._input_timestamps[0]
                    self._cached_input_fps = (len(self._input_timestamps) - 1) / dt if dt > 0 else 0.0
                else:
                    self._cached_input_fps = 0.0

                if len(self._output_timestamps) > 1:
                    dt = self._output_timestamps[-1] - self._output_timestamps[0]
                    self._cached_output_fps = (len(self._output_timestamps) - 1) / dt if dt > 0 else 0.0
                else:
                    self._cached_output_fps = 0.0

            elapsed_sec = (now - self.start_time) if self.start_time is not None else 0.0
            video_duration_sec = self.frames_written / self.target_fps if self.target_fps > 0 else 0.0

            return {
                "input_fps": round(self._cached_input_fps, 1),
                "output_fps": round(self._cached_output_fps, 1),
                "target_fps": self.target_fps,
                "frames_received": self.frames_received,
                "frames_written": self.frames_written,
                "frames_dropped": self.frames_dropped,
                "frames_duplicated": self.frames_duplicated,
                "elapsed_sec": round(elapsed_sec, 2),
                "video_duration_sec": round(video_duration_sec, 2),
            }
