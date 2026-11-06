"""Per-frame angle statistics and Phototaxis Index for the Motilitometer.

Vector orientations are ``angles`` in radians in image convention (as produced by
``cv2.cartToPolar``, in [0, 2*pi)) with their ``magnitudes`` (um/s).

* ``frame_circular_stats`` -> magnitude-weighted circular mean and resultant length R for the
  CSV summary, using the same default convention as ``PolarPlotProcessor`` (Y inverted,
  weighted by magnitude).
* ``frame_histograms`` -> per-frame angular histogram, stored by ``raw_export`` so rose
  diagrams and Phototaxis Index series can be rebuilt later.
* ``frame_phototaxis_index`` / ``hist_phototaxis_series`` -> Phototaxis Index in [-1, +1]
  measuring movement toward (+1) vs away from (-1) the light source.

Histograms use ``N_BINS`` bins whose edges start at 0 (image convention). 72 bins (5 deg)
re-bin exactly to 36, 24, 18, 12, 8, 6... bins.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

N_BINS = 72
TWO_PI = 2.0 * np.pi

PI_LIGHT_POSITIONS: Tuple[str, ...] = (
    "Bottom (↓)",
    "Top (↑)",
    "Right (→)",
    "Left (←)",
)
PI_DEFAULT_LIGHT_POS: str = "Bottom (↓)"

PI_PRESETS: Tuple[str, ...] = (
    "Half-Plane (180°)",
    "Cones 90°",
    "Cones 60°",
    "Custom Range",
)
PI_DEFAULT_PRESET: str = "Half-Plane (180°)"

_LIGHT_CENTER_DEG: Dict[str, Tuple[float, float]] = {
    "Bottom (↓)": (270.0, 90.0),
    "Top (↑)": (90.0, 270.0),
    "Right (→)": (0.0, 180.0),
    "Left (←)": (180.0, 0.0),
}

_LIGHT_ARROWS: Dict[str, Tuple[str, str]] = {
    "Bottom (↓)": ("↓", "↑"),
    "Top (↑)": ("↑", "↓"),
    "Right (→)": ("→", "←"),
    "Left (←)": ("←", "→"),
}

_PRESET_HALF_WIDTH_DEG: Dict[str, float] = {
    "Half-Plane (180°)": 90.0,
    "Cones 90°": 45.0,
    "Cones 60°": 30.0,
}


def normalize_pi_light_pos(light_pos: Any) -> str:
    """Normalize a light position string to one of ``PI_LIGHT_POSITIONS``."""
    s = str(light_pos or "").strip()
    if s in _LIGHT_CENTER_DEG:
        return s
    low = s.lower()
    if "top" in low or "up" in low or "haut" in low or "↑" in low:
        return "Top (↑)"
    if "right" in low or "droite" in low or "→" in low:
        return "Right (→)"
    if "left" in low or "gauche" in low or "←" in low:
        return "Left (←)"
    return PI_DEFAULT_LIGHT_POS


def normalize_pi_preset(preset: Any) -> str:
    """Normalize legacy or current preset names to ``PI_PRESETS``."""
    s = str(preset or "").strip()
    if s in PI_PRESETS:
        return s
    if s.startswith("Half-Plane"):
        return "Half-Plane (180°)"
    if s.startswith("Cones 90"):
        return "Cones 90°"
    if s.startswith("Cones 60"):
        return "Cones 60°"
    if "Custom" in s:
        return "Custom Range"
    return PI_DEFAULT_PRESET


def pi_light_arrows(light_pos: Any) -> Tuple[str, str]:
    """Return ``(toward_arrow, away_arrow)`` for a given light position."""
    return _LIGHT_ARROWS[normalize_pi_light_pos(light_pos)]


def _sector_bounds(center_deg: float, half_width_deg: float) -> Tuple[float, float]:
    lo = (center_deg - half_width_deg) % 360.0
    hi = (center_deg + half_width_deg) % 360.0
    if hi == 0.0 and lo > 0.0:
        hi = 360.0
    return float(lo), float(hi)


def pi_preset_ranges(
    preset: Any,
    light_pos: Any = PI_DEFAULT_LIGHT_POS,
) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(toward_min, toward_max, away_min, away_max)`` in screen degrees,
    or ``None`` when ``preset`` is ``"Custom Range"``."""
    norm_preset = normalize_pi_preset(preset)
    half_w = _PRESET_HALF_WIDTH_DEG.get(norm_preset)
    if half_w is None:
        return None
    toward_c, away_c = _LIGHT_CENTER_DEG[normalize_pi_light_pos(light_pos)]
    t_min, t_max = _sector_bounds(toward_c, half_w)
    a_min, a_max = _sector_bounds(away_c, half_w)
    return t_min, t_max, a_min, a_max


def in_angle_range(angles_deg: np.ndarray, min_deg: float, max_deg: float) -> np.ndarray:
    """Boolean mask for angles (in degrees) inside ``[min_deg, max_deg]`` with circular wrap."""
    a = np.asarray(angles_deg, dtype=np.float64) % 360.0
    span = float(max_deg) - float(min_deg)
    if abs(span) >= 360.0:
        return np.ones(a.shape, dtype=bool)
    mn = float(min_deg) % 360.0
    mx = 360.0 if (float(max_deg) == 360.0 and mn < 360.0) else (float(max_deg) % 360.0)
    if mn <= mx:
        return (a >= mn) & (a <= mx)
    return (a >= mn) | (a <= mx)


def _clean(angles: Any, mags: Any) -> Tuple[np.ndarray, np.ndarray]:
    if angles is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64).ravel()
    if ang.size == 0:
        return ang, np.empty(0, dtype=np.float64)
    if mags is not None and len(mags) == len(ang):
        mag = np.asarray(mags, dtype=np.float64).ravel()
    else:
        mag = np.ones_like(ang)
    ok = np.isfinite(ang) & np.isfinite(mag)
    if not ok.all():
        ang = ang[ok]
        mag = mag[ok]
    return np.mod(ang, TWO_PI), mag


def frame_circular_stats(
    angles: Any,
    mags: Any,
    invert_y: bool = True,
    weight_by_mag: bool = True,
) -> Tuple[float, float]:
    """Return ``(mean_angle_deg, R)`` for one frame; ``(nan, nan)`` if there is no vector."""
    ang, mag = _clean(angles, mags)
    if ang.size == 0:
        return float("nan"), float("nan")
    if invert_y:
        ang = np.mod(-ang, TWO_PI)
    w = mag if weight_by_mag else np.ones_like(ang)
    sw = float(w.sum())
    if sw <= 0.0:
        return float("nan"), float("nan")
    c = float((w * np.cos(ang)).sum() / sw)
    s = float((w * np.sin(ang)).sum() / sw)
    return float(np.degrees(np.arctan2(s, c)) % 360.0), float(np.hypot(c, s))


def frame_phototaxis_index(
    angles: Any,
    mags: Any,
    toward_min: float = 180.0,
    toward_max: float = 360.0,
    away_min: float = 0.0,
    away_max: float = 180.0,
    weight_by_mag: bool = True,
    invert_y: bool = True,
) -> float:
    """Return the Phototaxis Index ``(toward - away) / (toward + away)`` in ``[-1, +1]``
    for one frame of raw vectors, or ``nan`` if both sectors are empty."""
    ang, mag = _clean(angles, mags)
    if ang.size == 0:
        return float("nan")
    screen_deg = np.degrees(np.mod(-ang, TWO_PI) if invert_y else ang) % 360.0
    w = mag if weight_by_mag else np.ones_like(ang)
    t_mask = in_angle_range(screen_deg, toward_min, toward_max)
    a_mask = in_angle_range(screen_deg, away_min, away_max)
    t_sum = float(w[t_mask].sum())
    a_sum = float(w[a_mask].sum())
    denom = t_sum + a_sum
    if denom <= 1e-12:
        return float("nan")
    return float((t_sum - a_sum) / denom)


def hist_phototaxis_series(
    ang_counts: np.ndarray,
    ang_mag_sums: Optional[np.ndarray] = None,
    toward_min: float = 180.0,
    toward_max: float = 360.0,
    away_min: float = 0.0,
    away_max: float = 180.0,
    weight_by_mag: bool = True,
) -> np.ndarray:
    """Compute Phototaxis Index series ``(T,)`` from ``(T, N_BINS)`` angular histograms."""
    src = (
        ang_mag_sums
        if (weight_by_mag and ang_mag_sums is not None)
        else ang_counts
    )
    if src is None or src.ndim != 2 or src.shape[0] == 0:
        return np.array([], dtype=np.float64)
    mat = src.astype(np.float64, copy=False)
    n_bins = mat.shape[1]
    bin_deg = 360.0 / float(n_bins)
    screen_centers = (360.0 - (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_deg) % 360.0
    t_mask = in_angle_range(screen_centers, toward_min, toward_max)
    a_mask = in_angle_range(screen_centers, away_min, away_max)
    t_sum = mat[:, t_mask].sum(axis=1) if np.any(t_mask) else np.zeros(mat.shape[0], dtype=np.float64)
    a_sum = mat[:, a_mask].sum(axis=1) if np.any(a_mask) else np.zeros(mat.shape[0], dtype=np.float64)
    denom = t_sum + a_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(denom > 1e-9, (t_sum - a_sum) / denom, np.nan)
    return pi


def format_phototaxis_readout(
    pi_val: Optional[float],
    light_pos: Any = PI_DEFAULT_LIGHT_POS,
) -> Tuple[str, Tuple[int, int, int]]:
    """Format a live Phototaxis Index value and RGB color for DearPyGui."""
    if pi_val is None or not np.isfinite(pi_val):
        return "Phototaxis: --", (150, 150, 150)
    t_arrow, a_arrow = pi_light_arrows(light_pos)
    val = float(pi_val)
    if val > 0.05:
        return f"Phototaxis: {val:+.2f} (Toward {t_arrow})", (100, 255, 140)
    if val < -0.05:
        return f"Phototaxis: {val:+.2f} (Away {a_arrow})", (255, 140, 100)
    return f"Phototaxis: {val:+.2f} (Neutral)", (210, 210, 210)


def frame_histograms(
    angles: Any,
    mags: Any,
    n_bins: int = N_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(counts, mag_sums)`` of shape ``(n_bins,)`` for one frame, image convention."""
    ang, mag = _clean(angles, mags)
    edges = np.linspace(0.0, TWO_PI, n_bins + 1)
    counts, _ = np.histogram(ang, bins=edges)
    msum, _ = np.histogram(ang, bins=edges, weights=mag)
    return np.minimum(counts, 65535).astype(np.uint16), msum.astype(np.float32)


def build_arrays(
    per_frame: Sequence[Tuple[Any, Any]],
) -> Tuple[List[float], List[float], np.ndarray, np.ndarray]:
    """``per_frame``: ``[(angles, mags), ...]`` -> ``(mean_deg[T], R[T], counts[T, N_BINS], mag_sums[T, N_BINS])``."""
    n = len(per_frame)
    means: List[float] = []
    rs: List[float] = []
    counts = np.zeros((n, N_BINS), dtype=np.uint16)
    msums = np.zeros((n, N_BINS), dtype=np.float32)
    for i, (a, m) in enumerate(per_frame):
        mean_deg, r = frame_circular_stats(a, m)
        means.append(mean_deg)
        rs.append(r)
        counts[i], msums[i] = frame_histograms(a, m)
    return means, rs, counts, msums
