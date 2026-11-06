"""
Drift Correction Engine for Motilitometer Counter Tracker.

Encapsulates:
- Spatial grid subdivision (Full 1x1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64) for local drift capture
- Automatic dominant cluster (2D angle + magnitude mode) detection for sedimenting/non-motile algae
- Optional global direction hint filtering ("Give Hint")
- Temporal decision locking (live checkbox and batch/live frame range nomenclature e.g. [0-500],[1000-1500])
- Vector subtraction correction (V_corrected = V - V_drift)
- Visual overlays for drift vectors, pre/post correction vectors, and filtered drift items
"""

from __future__ import annotations

from bisect import bisect_left
import csv
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np


class DriftProcessor:
    """Manages sedimentation/drift vector estimation, locking, subtraction, and visualization."""

    GRID_MODES: Dict[str, Tuple[int, int]] = {
        "Full (1x1)": (1, 1),
        "Full": (1, 1),
        "1/1": (1, 1),
        "1/2 (1x2)": (1, 2),
        "1/2 (2x1)": (2, 1),
        "1/2": (1, 2),
        "1/4 (2x2)": (2, 2),
        "1/4": (2, 2),
        "1/8 (2x4)": (2, 4),
        "1/8 (4x2)": (4, 2),
        "1/8": (2, 4),
        "1/16 (4x4)": (4, 4),
        "1/16": (4, 4),
        "1/32 (4x8)": (4, 8),
        "1/32 (8x4)": (8, 4),
        "1/32": (4, 8),
        "1/64 (8x8)": (8, 8),
        "1/64": (8, 8),
    }

    HINT_PRESETS: Dict[str, float] = {
        "Down (90°)": 90.0,
        "Up (270°)": 270.0,
        "Right (0°)": 0.0,
        "Left (180°)": 180.0,
    }

    _CSV_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    _fake_cell_offsets: Dict[Tuple[int, int], Tuple[float, float]] = {}
    _fake_last_frame_idx: Optional[int] = None
    _fake_grid_key: Optional[Tuple[int, int, int, int]] = None

    def __init__(
        self,
        precomputed_lock_states: Optional[Dict[int, Dict[Tuple[int, int], Dict[str, Any]]]] = None,
        preprocessed_drift_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._cell_states: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._grid_shape: Tuple[int, int] = (1, 1)
        self._last_reset_trigger: int = 0
        self._precomputed_lock_states: Dict[int, Dict[Tuple[int, int], Dict[str, Any]]] = (
            dict(precomputed_lock_states) if precomputed_lock_states else {}
        )
        self._preprocessed_drift_map: Optional[Dict[str, Any]] = preprocessed_drift_map

    @staticmethod
    def drift_path_for(video_path: str) -> str:
        """Returns `<video>_drift.csv` path in the same directory as `video_path`."""
        base, _ = os.path.splitext(video_path)
        if base.endswith("_drift"):
            return f"{base}.csv"
        return f"{base}_drift.csv"

    @classmethod
    def save_drift_csv(
        cls,
        path: str,
        frame_indices: Sequence[int | float],
        drift_infos: Sequence[Optional[Dict[str, Any]]],
        scale: float = 1.0,
        default_grid_shape: Tuple[int, int] = (1, 1),
    ) -> bool:
        """
        Exports per-frame drift speed, angle, dx, dy (and per-subcell values when grid > 1x1)
        to `<video>_drift.csv`. Returns True on success.
        """
        try:
            rows, cols = default_grid_shape
            for di in drift_infos:
                if isinstance(di, dict) and "grid_shape" in di:
                    gs = di["grid_shape"]
                    if isinstance(gs, (tuple, list)) and len(gs) == 2:
                        rows, cols = int(gs[0]), int(gs[1])
                        break

            has_subcells = (rows > 1 or cols > 1)
            header = [
                "frame_idx",
                "grid_rows",
                "grid_cols",
                "drift_speed_um_s",
                "drift_angle_deg",
                "dx_px",
                "dy_px",
            ]
            if has_subcells:
                for r in range(rows):
                    for c in range(cols):
                        prefix = f"r{r}_c{c}"
                        header.extend([
                            f"{prefix}_speed_um_s",
                            f"{prefix}_angle_deg",
                            f"{prefix}_dx_px",
                            f"{prefix}_dy_px",
                        ])

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for f_idx, di in zip(frame_indices, drift_infos):
                    di_dict = di if isinstance(di, dict) else {}
                    mean_spd = round(float(di_dict.get("mean_mag_um_s", 0.0)), 4)
                    mean_ang = round(float(di_dict.get("mean_ang_deg", 0.0)), 2)
                    mean_dx = round(float(di_dict.get("mean_dx", 0.0)), 6)
                    mean_dy = round(float(di_dict.get("mean_dy", 0.0)), 6)
                    row_vals: List[Any] = [int(round(float(f_idx))), rows, cols, mean_spd, mean_ang, mean_dx, mean_dy]

                    if has_subcells:
                        cells = di_dict.get("cells", {})
                        for r in range(rows):
                            for c in range(cols):
                                cell = (cells.get((r, c)) or cells.get(f"{r},{c}") or {}) if isinstance(cells, dict) else {}
                                c_dx = float(cell.get("dx", 0.0))
                                c_dy = float(cell.get("dy", 0.0))
                                c_mag = float(cell.get("mag_px", np.hypot(c_dx, c_dy)))
                                c_spd = round(c_mag * scale, 4)
                                c_ang = (
                                    round(float(np.rad2deg(cell.get("ang_rad", np.arctan2(c_dy, c_dx) % (2.0 * np.pi)))), 2)
                                    if c_mag > 0
                                    else 0.0
                                )
                                row_vals.extend([c_spd, c_ang, round(c_dx, 6), round(c_dy, 6)])
                    writer.writerow(row_vals)

            # Invalidate cache for this path if it was cached
            abs_p = os.path.abspath(path)
            cls._CSV_CACHE.pop(abs_p, None)
            return True
        except Exception:
            return False

    @classmethod
    def _fill_empty_drift_frames(cls, frames: Dict[int, Dict[str, Any]], sorted_keys: List[int]) -> None:
        """
        Backfills leading zero-drift frames (such as Frame 0, which has no optical-flow/tracking
        predecessor) from the first valid frame, and forward-fills any intermediate zero-drift frames
        both at the frame level and per grid cell `(r, c)`.
        """
        if not frames or not sorted_keys:
            return

        def _is_frame_valid(entry: Dict[str, Any]) -> bool:
            if entry.get("mean_mag_px", 0.0) > 0 or entry.get("mean_mag_um_s", 0.0) > 0:
                return True
            return any(bool(c.get("valid", False)) for c in entry.get("cells", {}).values())

        first_valid: Optional[Dict[str, Any]] = None
        for k in sorted_keys:
            if _is_frame_valid(frames[k]):
                first_valid = frames[k]
                break
        if first_valid is None:
            return

        last_valid = first_valid
        for k in sorted_keys:
            if _is_frame_valid(frames[k]):
                last_valid = frames[k]
            else:
                frames[k] = {
                    **last_valid,
                    "cells": {ck: dict(cv) for ck, cv in last_valid.get("cells", {}).items()},
                }

        # Also backfill/forward-fill individual subcells (r, c) across frames when grid > 1x1
        sample_cells = first_valid.get("cells", {})
        for cell_key in list(sample_cells.keys()):
            first_c_valid: Optional[Dict[str, Any]] = None
            for k in sorted_keys:
                c_entry = frames[k].get("cells", {}).get(cell_key)
                if c_entry and c_entry.get("valid", False):
                    first_c_valid = c_entry
                    break
            if first_c_valid is not None:
                last_c_valid = first_c_valid
                for k in sorted_keys:
                    c_entry = frames[k].get("cells", {}).get(cell_key)
                    if c_entry and c_entry.get("valid", False):
                        last_c_valid = c_entry
                    elif "cells" in frames[k]:
                        frames[k]["cells"][cell_key] = dict(last_c_valid)

    @classmethod
    def load_drift_csv(cls, path: str) -> Optional[Dict[str, Any]]:
        """
        Loads a `<video>_drift.csv` file and returns a dictionary mapping frame indices
        to global and per-cell drift vectors. Uses an mtime-aware memory cache.
        """
        if not path or not os.path.isfile(path):
            return None
        abs_p = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(abs_p)
        except OSError:
            return None

        cached = cls._CSV_CACHE.get(abs_p)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        try:
            frames: Dict[int, Dict[str, Any]] = {}
            grid_rows, grid_cols = 1, 1
            with open(abs_p, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row or "frame_idx" not in row:
                        continue
                    f_idx = int(float(row["frame_idx"]))
                    r_cnt = max(1, int(float(row.get("grid_rows") or 1)))
                    c_cnt = max(1, int(float(row.get("grid_cols") or 1)))
                    grid_rows, grid_cols = r_cnt, c_cnt

                    mean_spd = float(row.get("drift_speed_um_s") or 0.0)
                    mean_ang = float(row.get("drift_angle_deg") or 0.0)
                    mean_dx = float(row.get("dx_px") or 0.0)
                    mean_dy = float(row.get("dy_px") or 0.0)
                    mean_mag_px = float(np.hypot(mean_dx, mean_dy))

                    cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
                    if r_cnt > 1 or c_cnt > 1:
                        for r in range(r_cnt):
                            for c in range(c_cnt):
                                prefix = f"r{r}_c{c}"
                                if f"{prefix}_dx_px" in row and row[f"{prefix}_dx_px"] not in (None, ""):
                                    c_dx = float(row[f"{prefix}_dx_px"])
                                    c_dy = float(row[f"{prefix}_dy_px"])
                                    c_spd = float(row.get(f"{prefix}_speed_um_s") or 0.0)
                                    c_ang_deg = float(row.get(f"{prefix}_angle_deg") or 0.0)
                                else:
                                    c_dx, c_dy, c_spd, c_ang_deg = mean_dx, mean_dy, mean_spd, mean_ang
                                c_mag_px = float(np.hypot(c_dx, c_dy))
                                c_ang_rad = float(np.arctan2(c_dy, c_dx) % (2.0 * np.pi)) if c_mag_px > 0 else float(np.deg2rad(c_ang_deg % 360.0))
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
                        c_ang_rad = float(np.arctan2(mean_dy, mean_dx) % (2.0 * np.pi)) if mean_mag_px > 0 else float(np.deg2rad(mean_ang % 360.0))
                        cells[(0, 0)] = {
                            "dx": mean_dx,
                            "dy": mean_dy,
                            "mag_px": mean_mag_px,
                            "ang_rad": c_ang_rad,
                            "speed_um_s": mean_spd,
                            "ang_deg": mean_ang,
                            "valid": mean_mag_px > 0 or mean_spd > 0,
                            "count": 0,
                        }

                    frames[f_idx] = {
                        "grid_shape": (r_cnt, c_cnt),
                        "mean_mag_um_s": mean_spd,
                        "mean_ang_deg": mean_ang,
                        "mean_dx": mean_dx,
                        "mean_dy": mean_dy,
                        "mean_mag_px": mean_mag_px,
                        "cells": cells,
                    }

            if not frames:
                return None

            sorted_keys = sorted(frames.keys())
            cls._fill_empty_drift_frames(frames, sorted_keys)
            result = {
                "path": abs_p,
                "grid_shape": (grid_rows, grid_cols),
                "frames": frames,
                "sorted_keys": sorted_keys,
            }
            cls._CSV_CACHE[abs_p] = (mtime, result)
            return result
        except Exception:
            return None

    @classmethod
    def find_and_load_drift_for_video(cls, video_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Finds and loads preprocessed drift for `video_path`.
        1. Checks `<video>_drift.csv` first.
        2. Falls back to `<video>.npz` if it contains `drift_speed` / `drift_dx` arrays.
        3. Falls back to `<video>.csv` if it contains `DriftSpeed_*` and `DriftAngle_deg_*` columns.
        Returns `(loaded_file_path, preproc_map)` or `(None, None)`.
        """
        if not video_path:
            return None, None

        # 1. Primary: <video>_drift.csv
        drift_csv = cls.drift_path_for(video_path)
        if os.path.isfile(drift_csv):
            dmap = cls.load_drift_csv(drift_csv)
            if dmap is not None:
                return drift_csv, dmap

        base = os.path.splitext(video_path)[0]

        # 2. Fallback: <video>.npz
        npz_path = base + ".npz"
        if os.path.isfile(npz_path):
            abs_npz = os.path.abspath(npz_path)
            try:
                mtime = os.path.getmtime(abs_npz)
                cached = cls._CSV_CACHE.get(abs_npz)
                if cached is not None and cached[0] == mtime:
                    return abs_npz, cached[1]
                with np.load(abs_npz, allow_pickle=False) as z:
                    if "drift_speed" in z.files and "frame_idx" in z.files:
                        f_idxs = z["frame_idx"]
                        d_spd = z["drift_speed"]
                        d_ang = z["drift_angle_deg"] if "drift_angle_deg" in z.files else np.zeros_like(d_spd)
                        d_dx = z["drift_dx"] if "drift_dx" in z.files else np.zeros_like(d_spd)
                        d_dy = z["drift_dy"] if "drift_dy" in z.files else np.zeros_like(d_spd)
                        c_spd_arr = z["drift_cell_speed"] if "drift_cell_speed" in z.files else None
                        c_ang_arr = z["drift_cell_angle_deg"] if "drift_cell_angle_deg" in z.files else None
                        c_dx_arr = z["drift_cell_dx"] if "drift_cell_dx" in z.files else None
                        c_dy_arr = z["drift_cell_dy"] if "drift_cell_dy" in z.files else None

                        r_cnt, c_cnt = (int(c_spd_arr.shape[1]), int(c_spd_arr.shape[2])) if (c_spd_arr is not None and c_spd_arr.ndim == 3) else (1, 1)
                        frames: Dict[int, Dict[str, Any]] = {}
                        for i in range(len(f_idxs)):
                            fi = int(round(float(f_idxs[i])))
                            m_spd = float(d_spd[i]) if np.isfinite(d_spd[i]) else 0.0
                            m_ang = float(d_ang[i]) if np.isfinite(d_ang[i]) else 0.0
                            m_dx = float(d_dx[i]) if np.isfinite(d_dx[i]) else 0.0
                            m_dy = float(d_dy[i]) if np.isfinite(d_dy[i]) else 0.0
                            m_mag_px = float(np.hypot(m_dx, m_dy))
                            cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
                            if r_cnt > 1 or c_cnt > 1:
                                for r in range(r_cnt):
                                    for c in range(c_cnt):
                                        cs = float(c_spd_arr[i, r, c]) if (c_spd_arr is not None and np.isfinite(c_spd_arr[i, r, c])) else m_spd
                                        ca = float(c_ang_arr[i, r, c]) if (c_ang_arr is not None and np.isfinite(c_ang_arr[i, r, c])) else m_ang
                                        cdx = float(c_dx_arr[i, r, c]) if (c_dx_arr is not None and np.isfinite(c_dx_arr[i, r, c])) else m_dx
                                        cdy = float(c_dy_arr[i, r, c]) if (c_dy_arr is not None and np.isfinite(c_dy_arr[i, r, c])) else m_dy
                                        cmag = float(np.hypot(cdx, cdy))
                                        crad = float(np.arctan2(cdy, cdx) % (2.0 * np.pi)) if cmag > 0 else float(np.deg2rad(ca % 360.0))
                                        cells[(r, c)] = {
                                            "dx": cdx, "dy": cdy, "mag_px": cmag, "ang_rad": crad,
                                            "speed_um_s": cs, "ang_deg": ca, "valid": cmag > 0 or cs > 0, "count": 0,
                                        }
                            else:
                                crad = float(np.arctan2(m_dy, m_dx) % (2.0 * np.pi)) if m_mag_px > 0 else float(np.deg2rad(m_ang % 360.0))
                                cells[(0, 0)] = {
                                    "dx": m_dx, "dy": m_dy, "mag_px": m_mag_px, "ang_rad": crad,
                                    "speed_um_s": m_spd, "ang_deg": m_ang, "valid": m_mag_px > 0 or m_spd > 0, "count": 0,
                                }
                            frames[fi] = {
                                "grid_shape": (r_cnt, c_cnt),
                                "mean_mag_um_s": m_spd,
                                "mean_ang_deg": m_ang,
                                "mean_dx": m_dx,
                                "mean_dy": m_dy,
                                "mean_mag_px": m_mag_px,
                                "cells": cells,
                            }
                        if frames:
                            s_keys = sorted(frames.keys())
                            cls._fill_empty_drift_frames(frames, s_keys)
                            res = {"path": abs_npz, "grid_shape": (r_cnt, c_cnt), "frames": frames, "sorted_keys": s_keys}
                            cls._CSV_CACHE[abs_npz] = (mtime, res)
                            return abs_npz, res
            except Exception:
                pass

        return None, None

    @staticmethod
    def _lookup_preprocessed_frame(preproc_map: Dict[str, Any], frame_idx: Optional[int]) -> Optional[Dict[str, Any]]:
        """Finds the exact or closest frame entry in `preproc_map`."""
        frames = preproc_map.get("frames")
        if not frames:
            return None
        if frame_idx is None:
            first_k = preproc_map["sorted_keys"][0]
            return frames[first_k]
        f = int(frame_idx)
        if f in frames:
            return frames[f]
        keys = preproc_map.get("sorted_keys") or sorted(frames.keys())
        pos = bisect_left(keys, f)
        if pos == 0:
            return frames[keys[0]]
        if pos >= len(keys):
            return frames[keys[-1]]
        before, after = keys[pos - 1], keys[pos]
        return frames[before] if (f - before) <= (after - f) else frames[after]

    def reset(self) -> None:
        """Clears cached drift vectors across all grid cells."""
        self._cell_states.clear()
        DriftProcessor._fake_cell_offsets.clear()
        DriftProcessor._fake_last_frame_idx = None

    def check_reset_trigger(self, trigger_val: int) -> None:
        """Resets state if the reset trigger counter has been incremented."""
        if trigger_val > self._last_reset_trigger:
            self.reset()
            self._last_reset_trigger = trigger_val

    @classmethod
    def parse_grid_mode(cls, grid_mode: Any) -> Tuple[int, int]:
        """Returns (rows, cols) for the given grid mode string."""
        if isinstance(grid_mode, str) and grid_mode in cls.GRID_MODES:
            return cls.GRID_MODES[grid_mode]
        if isinstance(grid_mode, str):
            m = re.search(r"(\d+)\s*x\s*(\d+)", grid_mode)
            if m:
                return max(1, int(m.group(1))), max(1, int(m.group(2)))
        return (1, 1)

    @staticmethod
    def parse_frame_ranges(ranges_str: Any) -> List[Tuple[int, int]]:
        """
        Parses frame range nomenclature such as '[0-500],[1000-1500]' into sorted (start, end) intervals.
        Also supports single integer '500' as '[0-500]'.
        """
        if not ranges_str or not isinstance(ranges_str, str):
            return []
        s = ranges_str.strip()
        if not s:
            return []

        # If a single integer X is given, interpret as [0-X] ("stop recalculating after frame X")
        if re.fullmatch(r"\d+", s):
            return [(0, int(s))]

        intervals: List[Tuple[int, int]] = []
        # Match patterns like [0-500], 0-500, [1000:1500], [500-]
        for match in re.finditer(r"\[?\s*(\d+)\s*[-:]\s*(\d*)\s*\]?", s):
            start_f = int(match.group(1))
            end_str = match.group(2)
            end_f = int(end_str) if end_str else 10**9
            if end_f < start_f:
                start_f, end_f = end_f, start_f
            intervals.append((start_f, end_f))

        intervals.sort(key=lambda x: x[0])
        return intervals

    @classmethod
    def is_frame_locked(cls, frame_idx: Optional[int], params: dict) -> Tuple[bool, Optional[int]]:
        """
        Determines whether drift estimation is locked on `frame_idx`.
        Returns:
            Tuple of (is_locked, lock_boundary_frame_idx)
        """
        if bool(params.get("drift_lock_live", False)):
            return True, None

        if bool(params.get("drift_use_ranges", False)) and frame_idx is not None:
            intervals = cls.parse_frame_ranges(params.get("drift_active_ranges", ""))
            if intervals:
                f = int(frame_idx)
                for start_f, end_f in intervals:
                    if start_f <= f <= end_f:
                        return False, None
                # Frame is outside active intervals; find the most recent active interval end before `f`
                prev_ends = [end_f for _, end_f in intervals if end_f < f]
                lock_frame = max(prev_ends) if prev_ends else intervals[0][0]
                return True, lock_frame

        return False, None

    @classmethod
    def get_hint_cone_rad(cls, params: dict) -> Optional[Tuple[float, float]]:
        """Returns (hint_angle_rad, hint_cone_rad) if 'Give Hint' is enabled, else None."""
        if not bool(params.get("drift_use_hint", False)):
            return None
        preset = str(params.get("drift_hint_preset", "Down (90°)"))
        if preset in cls.HINT_PRESETS:
            angle_deg = cls.HINT_PRESETS[preset]
        else:
            angle_deg = float(params.get("drift_hint_angle", 90.0))
        cone_deg = float(params.get("drift_hint_cone", 45.0))
        hint_rad = float(np.deg2rad(angle_deg % 360.0))
        cone_rad = float(np.deg2rad(np.clip(cone_deg, 1.0, 180.0)))
        return hint_rad, cone_rad

    @staticmethod
    def _circular_diff(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
        """Unsigned angular difference in [0, pi] between angles in radians."""
        return np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)

    @classmethod
    def _estimate_cell_drift(cls,
                             cell_vx: np.ndarray,
                             cell_vy: np.ndarray,
                             cell_mag: np.ndarray,
                             cell_ang: np.ndarray,
                             ang_tol_rad: float,
                             mag_tol_ratio: float,
                             hint: Optional[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
        """
        Finds the dominant drift cluster (angle + magnitude mode) in a grid cell.
        Returns dict with (dx, dy, mag_px, ang_rad, valid, count) or None if no valid cluster found.
        """
        n_total = len(cell_mag)
        if n_total == 0:
            return None

        # 1. Apply global direction hint filter if active
        if hint is not None:
            hint_rad, cone_rad = hint
            cand_mask = (cell_mag > 0) & (cls._circular_diff(cell_ang, hint_rad) <= cone_rad)
        else:
            cand_mask = cell_mag > 0

        cand_indices = np.nonzero(cand_mask)[0]
        n_cand = len(cand_indices)
        if n_cand == 0:
            return None

        cand_vx = cell_vx[cand_indices]
        cand_vy = cell_vy[cand_indices]
        cand_mag = cell_mag[cand_indices]
        cand_ang = cell_ang[cand_indices]

        # 2. Dominant 2D mode search in (angle, magnitude) space
        # Deterministic subsampling for high speed on dense pixel fields
        max_centers = 256
        max_eval = 2048
        c_stride = max(1, n_cand // max_centers)
        e_stride = max(1, n_cand // max_eval)

        center_ang = cand_ang[::c_stride][:max_centers]
        center_mag = cand_mag[::c_stride][:max_centers]
        eval_ang = cand_ang[::e_stride][:max_eval]
        eval_mag = cand_mag[::e_stride][:max_eval]

        ang_diff_mat = cls._circular_diff(eval_ang[:, None], center_ang[None, :])
        mag_thresh_row = np.maximum(center_mag[None, :] * mag_tol_ratio, 1e-6)
        mag_diff_mat = np.abs(eval_mag[:, None] - center_mag[None, :])

        inlier_mat = (ang_diff_mat <= ang_tol_rad) & (mag_diff_mat <= mag_thresh_row)
        counts = inlier_mat.sum(axis=0)
        best_c_idx = int(np.argmax(counts))
        if counts[best_c_idx] == 0:
            return None

        mode_ang = float(center_ang[best_c_idx])
        mode_mag = float(center_mag[best_c_idx])

        # 3. Gather all candidate inliers around (mode_ang, mode_mag) and compute mean drift vector
        inliers_cand = (
            (cls._circular_diff(cand_ang, mode_ang) <= ang_tol_rad)
            & (np.abs(cand_mag - mode_mag) <= max(mode_mag * mag_tol_ratio, 1e-6))
        )
        if not np.any(inliers_cand):
            return None

        dx = float(np.mean(cand_vx[inliers_cand]))
        dy = float(np.mean(cand_vy[inliers_cand]))
        mag_px = float(np.hypot(dx, dy))
        ang_rad = float(np.arctan2(dy, dx) % (2.0 * np.pi)) if mag_px > 0 else 0.0

        return {
            "dx": dx,
            "dy": dy,
            "mag_px": mag_px,
            "ang_rad": ang_rad,
            "valid": mag_px > 0,
            "count": int(np.count_nonzero(inliers_cand)),
        }

    def process_vectors(self,
                        xs: np.ndarray | List[float],
                        ys: np.ndarray | List[float],
                        vxs: np.ndarray | List[float],
                        vys: np.ndarray | List[float],
                        frame_shape: Tuple[int, int],
                        params: dict,
                        frame_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Categorizes drift vectors vs motile vectors per grid sub-section and computes
        drift-corrected vectors: V_corr = V - V_drift.
        """
        self.check_reset_trigger(int(params.get("reset_trigger", 0)))

        xs_arr = np.asarray(xs, dtype=np.float32)
        ys_arr = np.asarray(ys, dtype=np.float32)
        vxs_arr = np.asarray(vxs, dtype=np.float32)
        vys_arr = np.asarray(vys, dtype=np.float32)
        n_vecs = len(vxs_arr)

        mags = np.hypot(vxs_arr, vys_arr)
        angs = (np.arctan2(vys_arr, vxs_arr) + 2.0 * np.pi) % (2.0 * np.pi)

        enabled = bool(params.get("enable_drift_correction", False))
        H = max(1, int(frame_shape[0])) if frame_shape and len(frame_shape) >= 2 else 1
        W = max(1, int(frame_shape[1])) if frame_shape and len(frame_shape) >= 2 else 1
        rows, cols = self.parse_grid_mode(params.get("drift_grid_mode", "Full (1x1)"))

        if (rows, cols) != self._grid_shape:
            self._cell_states.clear()
            self._grid_shape = (rows, cols)

        calib = float(params.get("calibration", params.get("df_calibration", 1.0)))
        fps = float(params.get("fps", params.get("df_fps", 30.0)))
        scale = calib * fps if calib > 0 and fps > 0 else 1.0

        if not enabled:
            return {
                "enabled": False,
                "is_drift": np.zeros(n_vecs, dtype=bool),
                "corr_vx": vxs_arr.copy(),
                "corr_vy": vys_arr.copy(),
                "corr_mag": mags.copy(),
                "corr_ang": angs.copy(),
                "drift_vx": np.zeros(n_vecs, dtype=np.float32),
                "drift_vy": np.zeros(n_vecs, dtype=np.float32),
                "drift_info": {
                    "enabled": False,
                    "grid_shape": (rows, cols),
                    "frame_shape": (H, W),
                    "locked": False,
                    "cells": {},
                    "mean_dx": 0.0,
                    "mean_dy": 0.0,
                    "mean_mag_px": 0.0,
                    "mean_mag_um_s": 0.0,
                    "mean_ang_deg": 0.0,
                    "drift_count": 0,
                    "motile_count": n_vecs,
                    "total_count": n_vecs,
                },
            }

        ang_tol_deg = float(params.get("drift_angle_tol", 15.0))
        ang_tol_rad = float(np.deg2rad(np.clip(ang_tol_deg, 0.5, 180.0)))
        mag_tol_pct = float(params.get("drift_mag_tol", 25.0))
        mag_tol_ratio = float(np.clip(mag_tol_pct / 100.0, 0.01, 2.0))
        min_corr_speed_um_s = float(params.get("drift_min_corr_speed", 0.0))
        min_corr_px = (min_corr_speed_um_s / scale) if (min_corr_speed_um_s > 0.0 and scale > 0.0) else 0.0
        hint = self.get_hint_cone_rad(params)

        # Check if preprocessed drift reuse is enabled and available for this video
        reuse_requested = bool(params.get("drift_reuse_preprocessed", False))
        preproc_entry: Optional[Dict[str, Any]] = None
        if reuse_requested:
            preproc_map = params.get("_preprocessed_drift_map") or self._preprocessed_drift_map
            if preproc_map is None and params.get("_preprocessed_drift_path"):
                d_path = str(params["_preprocessed_drift_path"])
                if d_path.lower().endswith(".csv"):
                    preproc_map = self.load_drift_csv(d_path)
                if preproc_map is None:
                    _, preproc_map = self.find_and_load_drift_for_video(d_path)
            if preproc_map is None and params.get("_video_path"):
                _, preproc_map = self.find_and_load_drift_for_video(str(params["_video_path"]))
            if isinstance(preproc_map, dict):
                preproc_entry = self._lookup_preprocessed_frame(preproc_map, frame_idx)

        is_reused = preproc_entry is not None
        if is_reused and preproc_entry is not None:
            rows, cols = preproc_entry.get("grid_shape", (rows, cols))
            is_locked = False
            lock_boundary = None
            active_cell_states = preproc_entry.get("cells", {})
        else:
            is_locked, lock_boundary = self.is_frame_locked(frame_idx, params)
            # Check if precomputed batch lock state applies for this locked frame
            precomp_map = params.get("_batch_locked_drift_states") or self._precomputed_lock_states
            if is_locked and lock_boundary is not None and precomp_map and lock_boundary in precomp_map:
                active_cell_states = precomp_map[lock_boundary]
            else:
                active_cell_states = self._cell_states

        is_drift = np.zeros(n_vecs, dtype=bool)
        corr_vx = vxs_arr.copy()
        corr_vy = vys_arr.copy()
        drift_vx = np.zeros(n_vecs, dtype=np.float32)
        drift_vy = np.zeros(n_vecs, dtype=np.float32)

        if n_vecs > 0:
            cell_r = np.clip((ys_arr * rows / H).astype(np.int32), 0, rows - 1)
            cell_c = np.clip((xs_arr * cols / W).astype(np.int32), 0, cols - 1)
        else:
            cell_r = np.empty(0, dtype=np.int32)
            cell_c = np.empty(0, dtype=np.int32)

        cells_out: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for r in range(rows):
            for c in range(cols):
                key = (r, c)
                in_cell = (cell_r == r) & (cell_c == c)
                cell_idx = np.nonzero(in_cell)[0]

                # Update cell drift state if not reusing preprocessed and not locked (or if cell has never been initialized yet)
                if not is_reused and (not is_locked or key not in active_cell_states):
                    if len(cell_idx) > 0:
                        est = self._estimate_cell_drift(
                            vxs_arr[cell_idx],
                            vys_arr[cell_idx],
                            mags[cell_idx],
                            angs[cell_idx],
                            ang_tol_rad,
                            mag_tol_ratio,
                            hint,
                        )
                        if est is not None:
                            self._cell_states[key] = est

                cell_state = active_cell_states.get(key) if is_reused else (active_cell_states.get(key) or self._cell_states.get(key))
                if cell_state is not None and cell_state.get("valid", False):
                    d_dx = float(cell_state["dx"])
                    d_dy = float(cell_state["dy"])
                    d_mag = float(cell_state["mag_px"])
                    d_ang = float(cell_state["ang_rad"])
                    if d_mag <= 0.0 and cell_state.get("speed_um_s", 0.0) > 0.0 and scale > 0.0:
                        d_mag = float(cell_state["speed_um_s"]) / scale
                        d_dx = float(d_mag * np.cos(d_ang))
                        d_dy = float(d_mag * np.sin(d_ang))

                    if len(cell_idx) > 0:
                        c_angs = angs[cell_idx]
                        c_mags = mags[cell_idx]
                        c_inliers = (
                            (self._circular_diff(c_angs, d_ang) <= ang_tol_rad)
                            & (np.abs(c_mags - d_mag) <= max(d_mag * mag_tol_ratio, 1e-6))
                        )
                        if hint is not None and not is_reused:
                            hint_rad, cone_rad = hint
                            c_inliers &= (self._circular_diff(c_angs, hint_rad) <= cone_rad)

                        drift_idx = cell_idx[c_inliers]
                        motile_idx = cell_idx[~c_inliers]

                        is_drift[drift_idx] = True
                        corr_vx[drift_idx] = 0.0
                        corr_vy[drift_idx] = 0.0
                        drift_vx[cell_idx] = d_dx
                        drift_vy[cell_idx] = d_dy

                        # Vector subtraction on motile vectors: V_corr = V - V_drift
                        corr_vx[motile_idx] = vxs_arr[motile_idx] - d_dx
                        corr_vy[motile_idx] = vys_arr[motile_idx] - d_dy

                        # Post-correction minimum speed filter: ignore residual slow vectors
                        extra_drift_cnt = 0
                        if min_corr_px > 0.0 and len(motile_idx) > 0:
                            motile_corr_mag = np.hypot(corr_vx[motile_idx], corr_vy[motile_idx])
                            below_corr_min = motile_corr_mag < min_corr_px
                            if np.any(below_corr_min):
                                slow_corr_idx = motile_idx[below_corr_min]
                                is_drift[slow_corr_idx] = True
                                corr_vx[slow_corr_idx] = 0.0
                                corr_vy[slow_corr_idx] = 0.0
                                extra_drift_cnt = int(len(slow_corr_idx))

                        cells_out[key] = {
                            **cell_state,
                            "dx": d_dx,
                            "dy": d_dy,
                            "mag_px": d_mag,
                            "ang_rad": d_ang,
                            "count": int(len(drift_idx) + extra_drift_cnt),
                        }
                    else:
                        cells_out[key] = {
                            **cell_state,
                            "dx": d_dx,
                            "dy": d_dy,
                            "mag_px": d_mag,
                            "ang_rad": d_ang,
                            "count": 0,
                        }
                else:
                    cells_out[key] = {
                        "dx": 0.0,
                        "dy": 0.0,
                        "mag_px": 0.0,
                        "ang_rad": 0.0,
                        "valid": False,
                        "count": 0,
                    }

        corr_mag = np.hypot(corr_vx, corr_vy)
        corr_ang = (np.arctan2(corr_vy, corr_vx) + 2.0 * np.pi) % (2.0 * np.pi)

        valid_cells = [v for v in cells_out.values() if v.get("valid", False)]
        if valid_cells:
            mean_dx = float(np.mean([v["dx"] for v in valid_cells]))
            mean_dy = float(np.mean([v["dy"] for v in valid_cells]))
            mean_mag_px = float(np.hypot(mean_dx, mean_dy))
            mean_ang_rad = float(np.arctan2(mean_dy, mean_dx) % (2.0 * np.pi)) if mean_mag_px > 0 else 0.0
            mean_ang_deg = float(np.rad2deg(mean_ang_rad))
        else:
            mean_dx = 0.0
            mean_dy = 0.0
            mean_mag_px = 0.0
            mean_ang_deg = 0.0

        drift_count = int(np.count_nonzero(is_drift))
        motile_count = int(n_vecs - drift_count)

        drift_info = {
            "enabled": True,
            "frame_idx": frame_idx,
            "grid_shape": (rows, cols),
            "frame_shape": (H, W),
            "locked": is_locked,
            "reused": is_reused,
            "cells": cells_out,
            "mean_dx": mean_dx,
            "mean_dy": mean_dy,
            "mean_mag_px": mean_mag_px,
            "mean_mag_um_s": round(mean_mag_px * scale, 2),
            "mean_ang_deg": round(mean_ang_deg, 1),
            "drift_count": drift_count,
            "motile_count": motile_count,
            "total_count": n_vecs,
        }

        return {
            "enabled": True,
            "is_drift": is_drift,
            "corr_vx": corr_vx,
            "corr_vy": corr_vy,
            "corr_mag": corr_mag,
            "corr_ang": corr_ang,
            "drift_vx": drift_vx,
            "drift_vy": drift_vy,
            "drift_info": drift_info,
        }

    @classmethod
    def draw_fake_drifters(cls,
                           canvas: np.ndarray,
                           drift_info: Dict[str, Any]) -> None:
        """
        Draws a uniform grid of synthetic 'fake algae' across the frame that translate
        purely according to the local (or global) drift vector of their sub-section cell.
        """
        H, W = canvas.shape[:2]
        rows, cols = drift_info.get("grid_shape", (1, 1))
        cells: Dict[Tuple[int, int], Dict[str, Any]] = drift_info.get("cells", {})
        frame_idx = drift_info.get("frame_idx")

        grid_key = (H, W, rows, cols)
        if cls._fake_grid_key != grid_key:
            cls._fake_cell_offsets.clear()
            cls._fake_last_frame_idx = None
            cls._fake_grid_key = grid_key

        # Advance cumulative drift offset only when a new frame arrives (freezes when video is paused)
        should_advance = True
        if frame_idx is not None:
            f_int = int(frame_idx)
            if cls._fake_last_frame_idx is not None:
                if f_int == cls._fake_last_frame_idx:
                    should_advance = False
                elif f_int < cls._fake_last_frame_idx and f_int <= 2:
                    cls._fake_cell_offsets.clear()
            cls._fake_last_frame_idx = f_int

        mean_dx = float(drift_info.get("mean_dx", 0.0))
        mean_dy = float(drift_info.get("mean_dy", 0.0))
        base_step = max(36.0, round(min(W, H) / 10.0))

        for r in range(rows):
            for c in range(cols):
                key = (r, c)
                cdata = cells.get(key)
                if cdata is not None and cdata.get("valid", False):
                    dx = float(cdata.get("dx", 0.0))
                    dy = float(cdata.get("dy", 0.0))
                else:
                    dx, dy = mean_dx, mean_dy

                ox, oy = cls._fake_cell_offsets.get(key, (0.0, 0.0))
                if should_advance:
                    ox += dx
                    oy += dy
                    cls._fake_cell_offsets[key] = (ox, oy)

                x0 = c * W / cols
                x1 = (c + 1) * W / cols
                y0 = r * H / rows
                y1 = (r + 1) * H / rows
                cell_w = max(1.0, x1 - x0)
                cell_h = max(1.0, y1 - y0)

                step_x = min(base_step, max(16.0, cell_w / 2.0))
                step_y = min(base_step, max(16.0, cell_h / 2.0))

                start_x = x0 + (ox % step_x)
                start_y = y0 + (oy % step_y)
                mag_px = float(np.hypot(dx, dy))

                py = start_y
                while py < y1 - 1:
                    px = start_x
                    iy = int(round(py))
                    while px < x1 - 1:
                        ix = int(round(px))
                        if 0 <= ix < W and 0 <= iy < H:
                            if mag_px > 0.01:
                                # Short motion trail behind the fake drifter showing its drift direction
                                tail_scale = max(6.0, min(18.0, mag_px * 8.0))
                                tx = int(round(px - (dx / mag_px) * tail_scale))
                                ty = int(round(py - (dy / mag_px) * tail_scale))
                                cv2.line(canvas, (tx, ty), (ix, iy), (0, 0, 0), 3, cv2.LINE_AA)
                                cv2.line(canvas, (tx, ty), (ix, iy), (220, 220, 0), 1, cv2.LINE_AA)
                            # Synthetic alga body: dark outline + bright cyan disk + white core
                            cv2.circle(canvas, (ix, iy), 4, (0, 0, 0), -1, cv2.LINE_AA)
                            cv2.circle(canvas, (ix, iy), 3, (255, 255, 0), -1, cv2.LINE_AA)
                            cv2.circle(canvas, (ix, iy), 1, (255, 255, 255), -1, cv2.LINE_AA)
                        px += step_x
                    py += step_y

    @classmethod
    def draw_drift_overlay(cls,
                           canvas: Optional[np.ndarray],
                           drift_info: Optional[Dict[str, Any]],
                           arrow_scale: float,
                           params: dict,
                           min_arrow_len: float = 25.0) -> None:
        """
        Draws the estimated drift vector arrow(s) (and sub-section grid lines if > 1x1) on `canvas`
        with an amplified visual scale and minimum length so low-speed sedimentation remains clearly visible.
        Also draws synthetic 'fake drifters' if enabled.
        """
        if canvas is None or not drift_info or not drift_info.get("enabled", False):
            return
        show_vector = bool(params.get("drift_show_vector", True))
        show_fake = bool(params.get("drift_show_fake_drifters", False))
        if not show_vector and not show_fake:
            return

        H, W = canvas.shape[:2]
        rows, cols = drift_info.get("grid_shape", (1, 1))
        cells: Dict[Tuple[int, int], Dict[str, Any]] = drift_info.get("cells", {})
        is_locked = bool(drift_info.get("locked", False))

        calib = float(params.get("calibration", params.get("df_calibration", 1.0)))
        fps = float(params.get("fps", params.get("df_fps", 30.0)))
        scale = calib * fps if calib > 0 and fps > 0 else 1.0

        # Draw grid lines if subdivided
        if rows > 1 or cols > 1:
            for r in range(1, rows):
                y_line = int(round(r * H / rows))
                cv2.line(canvas, (0, y_line), (W, y_line), (90, 90, 90), 1, cv2.LINE_AA)
            for c in range(1, cols):
                x_line = int(round(c * W / cols))
                cv2.line(canvas, (x_line, 0), (x_line, H), (90, 90, 90), 1, cv2.LINE_AA)

        if show_fake:
            cls.draw_fake_drifters(canvas, drift_info)

        if not show_vector:
            return

        # Color: Magenta (255, 100, 255) when active, Gold (0, 215, 255) when locked
        arrow_color = (255, 100, 255) if not is_locked else (0, 215, 255)

        for (r, c), cdata in cells.items():
            if not cdata.get("valid", False):
                continue
            mag_px = float(cdata.get("mag_px", 0.0))
            if mag_px <= 0.0:
                continue
            ang_rad = float(cdata.get("ang_rad", 0.0))

            cx = int(round((c + 0.5) * W / cols))
            cy = int(round((r + 0.5) * H / rows))

            # Boosted scaling + base offset so slow drift (e.g. < 1 px/frame) is always clearly visible
            effective_scale = max(float(arrow_scale), 2.0) * 2.5
            arrow_len = max(min_arrow_len, 18.0 + mag_px * effective_scale)
            end_x = int(round(cx + arrow_len * np.cos(ang_rad)))
            end_y = int(round(cy + arrow_len * np.sin(ang_rad)))

            # Dark outline + bright foreground arrow for high visibility
            cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 3, arrow_color, -1, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.28)
            cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), arrow_color, 3, cv2.LINE_AA, tipLength=0.28)

            if rows <= 4 and cols <= 4:
                spd_um_s = mag_px * scale
                is_reused = bool(drift_info.get("reused", False))
                status_tag = " [REUSED]" if is_reused else (" [LOCK]" if is_locked else "")
                label = f"Drift: {spd_um_s:.1f} um/s{status_tag}"
                tx = max(4, min(W - 140, cx + 6))
                ty = max(14, min(H - 6, cy - 6))
                cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1, cv2.LINE_AA)
