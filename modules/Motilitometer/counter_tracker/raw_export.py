"""Compact per-frame raw data export (``<video>.npz``) and reader for Motilitometer.

Each frame keeps a *distribution* of its vectors instead of the vectors themselves:

* angular histogram (``ang_counts`` / ``ang_mag_sums``, 72 bins, image convention),
* speed histogram (``speed_counts`` over shared ``speed_edges``),
* exact per-frame summary: vector count, mean / std of speed, mean vector (vx, vy),
* the time series the Counter Tracker emits (``result``, ``surface``, ``count``, ``motility``).

``RawData`` reloads the file and gives everything the Motility Reprocessor needs to
replay the data synchronised with the video.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .angle_export import N_BINS, TWO_PI, _clean, frame_histograms

SCHEMA_VERSION = 2
SPEED_BINS = 64


def raw_path_for(video_path: str) -> str:
    """``video.mp4`` -> ``video.npz`` (same folder, same name)."""
    return os.path.splitext(video_path)[0] + ".npz"


def _speed_edges_from_cleaned(cleaned: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    mags = [mag for _, mag in cleaned if mag.size]
    hi = float(np.percentile(np.concatenate(mags), 99.9)) if mags else 0.0
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    return np.linspace(0.0, hi, SPEED_BINS + 1, dtype=np.float32)


def _speed_edges(frame_vectors: Sequence[Tuple[Any, Any]]) -> np.ndarray:
    return _speed_edges_from_cleaned([_clean(a, m) for a, m in frame_vectors])


def _padded(values: Any, n: int) -> np.ndarray:
    """float32 array of length n (NaN when the series is absent / shorter)."""
    out = np.full(n, np.nan, dtype=np.float32)
    if values is not None and len(values) == n:
        out[:] = np.asarray([np.nan if v is None else v for v in values], dtype=np.float32)
    return out


def build_raw_arrays(frame_vectors: Sequence[Tuple[Any, Any]]) -> Dict[str, np.ndarray]:
    """frame_vectors: [(angles, magnitudes), ...] one entry per frame."""
    n = len(frame_vectors)
    cleaned = [_clean(a, m) for a, m in frame_vectors]
    edges = _speed_edges_from_cleaned(cleaned)
    top = float(edges[-1]) * (1.0 - 1e-6)
    ang_edges = np.linspace(0.0, TWO_PI, N_BINS + 1)
    ang_counts = np.zeros((n, N_BINS), dtype=np.uint16)
    ang_msums = np.zeros((n, N_BINS), dtype=np.float32)
    spd_counts = np.zeros((n, SPEED_BINS), dtype=np.uint16)
    n_vec = np.zeros(n, dtype=np.int32)
    nan = lambda: np.full(n, np.nan, dtype=np.float32)
    spd_mean, spd_std, vx, vy = nan(), nan(), nan(), nan()

    for i, (ang, mag) in enumerate(cleaned):
        n_vec[i] = ang.size
        if ang.size == 0:
            continue
        c_hist, _ = np.histogram(ang, bins=ang_edges)
        m_hist, _ = np.histogram(ang, bins=ang_edges, weights=mag)
        ang_counts[i] = np.minimum(c_hist, 65535).astype(np.uint16)
        ang_msums[i] = m_hist.astype(np.float32)
        spd_mean[i], spd_std[i] = mag.mean(), mag.std()
        vx[i], vy[i] = (mag * np.cos(ang)).mean(), (mag * np.sin(ang)).mean()
        h, _ = np.histogram(np.minimum(mag, top), bins=edges)
        spd_counts[i] = np.minimum(h, 65535)

    return {"ang_counts": ang_counts, "ang_mag_sums": ang_msums, "speed_counts": spd_counts,
            "speed_edges": edges, "n_vec": n_vec, "vec_speed_mean": spd_mean, "vec_speed_std": spd_std,
            "vx_mean": vx, "vy_mean": vy}


def save_raw_npz(path: str, *, frame_idx: Sequence[float], fps: float, mode: str, source: str,
                 frame_vectors: Sequence[Tuple[Any, Any]], result: Any, surface: Any = None,
                 count: Any = None, motility: Any = None, phototaxis: Any = None,
                 metric_kind: str = "surface",
                 drift_speed: Any = None, drift_angle_deg: Any = None,
                 drift_dx: Any = None, drift_dy: Any = None,
                 drift_cell_speed: Any = None, drift_cell_angle_deg: Any = None,
                 drift_cell_dx: Any = None, drift_cell_dy: Any = None,
                 extra_meta: Optional[Dict[str, Any]] = None,
                 precomputed_arrays: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write the compact raw export. ``result`` is the series sent to the Result output
    (Speed in ``both`` mode); ``surface`` is only used in ``both`` mode."""
    n = len(frame_idx)
    fidx = np.asarray(frame_idx, dtype=np.int32)
    meta = {"mode": mode, "source": source, "fps": float(fps), "metric_kind": metric_kind,
            "magnitude_unit": "um/s",
            "vector_scaled": True,
            "angle_convention": "image: radians from +x, y down (cv2.cartToPolar); bins start at 0"}
    meta.update(extra_meta or {})
    arrays = precomputed_arrays if precomputed_arrays is not None else build_raw_arrays(frame_vectors)
    extra_arrays: Dict[str, np.ndarray] = {}
    if phototaxis is not None:
        extra_arrays["phototaxis"] = _padded(phototaxis, n)
    if drift_speed is not None:
        extra_arrays["drift_speed"] = _padded(drift_speed, n)
    if drift_angle_deg is not None:
        extra_arrays["drift_angle_deg"] = _padded(drift_angle_deg, n)
    if drift_dx is not None:
        extra_arrays["drift_dx"] = _padded(drift_dx, n)
    if drift_dy is not None:
        extra_arrays["drift_dy"] = _padded(drift_dy, n)
    if drift_cell_speed is not None:
        extra_arrays["drift_cell_speed"] = np.asarray(drift_cell_speed, dtype=np.float32)
    if drift_cell_angle_deg is not None:
        extra_arrays["drift_cell_angle_deg"] = np.asarray(drift_cell_angle_deg, dtype=np.float32)
    if drift_cell_dx is not None:
        extra_arrays["drift_cell_dx"] = np.asarray(drift_cell_dx, dtype=np.float32)
    if drift_cell_dy is not None:
        extra_arrays["drift_cell_dy"] = np.asarray(drift_cell_dy, dtype=np.float32)
    np.savez_compressed(
        path,
        schema_version=np.int32(SCHEMA_VERSION),
        frame_idx=fidx,
        time_s=(fidx / float(fps or 1.0)).astype(np.float32),
        result=_padded(result, n), surface=_padded(surface, n),
        count=_padded(count, n), motility=_padded(motility, n),
        meta_json=np.str_(json.dumps(meta, default=str)),
        **arrays,
        **extra_arrays,
    )


class RawData:
    """Reader for ``<video>.npz`` raw exports."""

    def __init__(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as z:
            self._z = {k: z[k] for k in z.files}
        z = self._z
        self.path = path
        self.frame_idx = z["frame_idx"]
        self.time_s = z["time_s"]
        self.meta: Dict[str, Any] = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
        self.mode = str(self.meta.get("mode", z["mode"] if "mode" in z else "contour"))
        self.fps = float(self.meta.get("fps", z["fps"] if "fps" in z else 30.0))
        self.source = str(self.meta.get("source", ""))
        self.metric_kind = str(self.meta.get("metric_kind", "surface"))
        self.ang_counts = z["ang_counts"] if "ang_counts" in z else z["counts"]
        self.ang_mag_sums = z["ang_mag_sums"] if "ang_mag_sums" in z else z["mag_sums"]
        self.speed_counts = z.get("speed_counts")
        self.speed_edges = z.get("speed_edges")
        for k in ("result", "surface", "count", "motility", "phototaxis", "n_vec", "vec_speed_mean",
                  "vec_speed_std", "vx_mean", "vy_mean",
                  "drift_speed", "drift_angle_deg", "drift_dx", "drift_dy",
                  "drift_cell_speed", "drift_cell_angle_deg", "drift_cell_dx", "drift_cell_dy"):
            setattr(self, k, z.get(k))

        # Backward compatibility check:
        # Prior to this update, contour tracking stored vector magnitudes in px/frame instead of µm/s.
        # If vector_scaled is not True and mode is contour (or tracking-based), automatically scale
        # the angular magnitude sums, speed edges, vec_speed_mean, std, and mean vectors to µm/s.
        vector_scaled = bool(self.meta.get("vector_scaled", False))
        if not vector_scaled and self.mode in ("contour", "both_contour"):
            params = self.meta.get("parameters", {}) if isinstance(self.meta.get("parameters"), dict) else {}
            calib = float(params.get("calibration", self.meta.get("calibration", 1.0)) or 1.0)
            fps_val = float(params.get("fps", self.meta.get("fps", self.fps or 30.0)) or 30.0)
            scale = calib * fps_val if calib > 0 and fps_val > 0 else 1.0
            if scale > 0 and abs(scale - 1.0) > 1e-4:
                if self.ang_mag_sums is not None:
                    self.ang_mag_sums = (self.ang_mag_sums.astype(np.float64) * scale).astype(np.float32)
                if self.speed_edges is not None:
                    self.speed_edges = (self.speed_edges.astype(np.float64) * scale).astype(np.float32)
                if self.vec_speed_mean is not None:
                    self.vec_speed_mean = (self.vec_speed_mean.astype(np.float64) * scale).astype(np.float32)
                if self.vec_speed_std is not None:
                    self.vec_speed_std = (self.vec_speed_std.astype(np.float64) * scale).astype(np.float32)
                if self.vx_mean is not None:
                    self.vx_mean = (self.vx_mean.astype(np.float64) * scale).astype(np.float32)
                if self.vy_mean is not None:
                    self.vy_mean = (self.vy_mean.astype(np.float64) * scale).astype(np.float32)
                self.meta["vector_scaled"] = True

        mv = np.hypot(self.vx_mean, self.vy_mean) if self.vx_mean is not None else np.array([])
        mv = mv[np.isfinite(mv)]
        self.vector_ref = float(np.percentile(mv, 99)) if mv.size else 0.0

    # ── lookup ────────────────────────────────────────────────────────
    def row(self, frame_idx: int) -> Optional[int]:
        """Row of an exact frame index, or None if that frame was not processed."""
        i = int(np.searchsorted(self.frame_idx, frame_idx))
        return i if i < len(self.frame_idx) and self.frame_idx[i] == frame_idx else None

    def upto(self, frame_idx: int) -> int:
        """Number of processed frames with index <= frame_idx."""
        return int(np.searchsorted(self.frame_idx, frame_idx, side="right"))

    # ── per-frame reconstructions ─────────────────────────────────────
    def vectors(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Synthetic (angles, magnitudes) for a frame: bin centre repeated ``count`` times,
        magnitude = mean magnitude of the bin. Angles are in image convention."""
        counts = self.ang_counts[i].astype(np.int64)
        nb = len(counts)
        centers = (np.arange(nb) + 0.5) * (TWO_PI / nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_mag = np.where(counts > 0, self.ang_mag_sums[i] / np.maximum(counts, 1), 0.0)
        return np.repeat(centers, counts), np.repeat(mean_mag, counts).astype(np.float32)

    def raw_speeds(self, i: int, cap: int = 5000) -> np.ndarray:
        """Approximate per-vector speeds (bin centres repeated by count)."""
        if self.speed_counts is None:
            return np.array([], dtype=np.float32)
        counts = self.speed_counts[i].astype(np.int64)
        centers = (self.speed_edges[:-1] + self.speed_edges[1:]) / 2.0
        out = np.repeat(centers, counts)
        if out.size > cap:
            out = out[:: out.size // cap]
        return out

    def pooled(self, frame_range: Optional[Tuple[int, int]] = None, num_bins: int = 24,
               weight_by_mag: bool = True, invert_y: bool = True,
               normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """(bin_centers_rad, values) as PolarPlotProcessor computes them. frame_range is
        (first, last) frame indices inclusive; num_bins must divide the stored bin count."""
        n_src = self.ang_counts.shape[1]
        if n_src % num_bins:
            raise ValueError(f"num_bins must divide {n_src}")
        sel = np.ones(len(self.frame_idx), dtype=bool)
        if frame_range is not None:
            sel = (self.frame_idx >= frame_range[0]) & (self.frame_idx <= frame_range[1])
        src = (self.ang_mag_sums if weight_by_mag else self.ang_counts)[sel].sum(axis=0).astype(np.float64)
        if invert_y:  # plot angle = -image angle: bin k -> n_src - 1 - k
            src = src[::-1]
        vals = src.reshape(num_bins, n_src // num_bins).sum(axis=1)
        if normalize and vals.sum() > 0:
            vals = vals / vals.sum() * 100.0
        width = TWO_PI / num_bins
        return np.arange(num_bins) * width + width / 2.0, vals


PolarData = RawData  # backward-compatible alias
