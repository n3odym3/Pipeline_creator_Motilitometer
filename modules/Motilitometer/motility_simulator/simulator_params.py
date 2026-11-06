"""Parameter table of the Motility Simulator.

Single source of truth shared by the engine (defaults / keys) and the UI (widgets,
tooltips, ranges). Angles are in degrees, counter-clockwise as seen on screen
(0 = right, 90 = up, 180 = left, 270 = down).

``structural`` parameters change the population itself (sprites, speeds, positions...)
and only take effect when the simulation is (re)started. Non-structural parameters can
be tuned live while the simulation is running.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

BG_MODES = ["Solid color", "Image (stretched)", "Live frame (overlay)"]
SPRITE_MODES = ["Circle", "PNG image", "Ring (outline)"]
BOUNDARY_MODES = ["Bounce (closed system)", "Disappear & respawn"]
INTENTION_MODES = ["Random (per individual)", "Common direction"]
CHIRALITY_MODES = ["Random", "Clockwise", "Counter-clockwise"]
EXPORT_FORMATS = ["AVI lossless (FFV1)", "AVI MJPG", "MP4 (mp4v)"]


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str  # int | float | bool | combo | color | path | str
    default: Any
    section: str
    tip: str = ""
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    items: Optional[Tuple[str, ...]] = None
    structural: bool = False
    fmt: str = "%.2f"
    filetypes: str = "image"  # only for kind == "path"


_S_VIDEO = "Video & Calibration"
_S_BG = "Background"
_S_ORG = "Organisms"
_S_MOT = "Motion"
_S_PHOTO = "Phototaxis"
_S_DRIFT = "Drift / Sedimentation"
_S_FOCUS = "Focus / Blur"
_S_SHOCK = "Photoshock"

PARAMS: List[Param] = [
    # ── Video & calibration ─────────────────────────────────────────────
    Param("width", "Width (px)", "int", 800, _S_VIDEO, "Image width in pixels.", 16, 8192, structural=True),
    Param("height", "Height (px)", "int", 600, _S_VIDEO, "Image height in pixels.", 16, 8192, structural=True),
    Param("fps", "Frame rate (fps)", "float", 30.0, _S_VIDEO,
          "Acquisition frame rate. Use the same value in the Motilitometer Tracker.", 0.1, 1000.0,
          structural=True, fmt="%.1f"),
    Param("calibration", "Scale (µm/px)", "float", 0.7575, _S_VIDEO,
          "Spatial calibration. Use the same value in the Motilitometer Tracker so that speeds in µm/s match.",
          0.0001, 10000.0, structural=True, fmt="%.4f"),
    Param("duration_s", "Duration (s)", "float", 0.0, _S_VIDEO,
          "Length of the live run and of the exported video (0 = unlimited live run; export then uses 10 s).", 0.0, 36000.0, fmt="%.1f"),
    Param("seed", "Random seed", "int", 0, _S_VIDEO,
          "0 = different each run. Any other value gives a perfectly reproducible video.", 0, 2**31 - 1,
          structural=True),
    Param("realtime", "Real-time pacing (live)", "bool", True, _S_VIDEO,
          "Live mode: emit frames at the chosen fps. Uncheck to emit as fast as possible "
          "(the tracker may drop frames; use video export for exact validation)."),
    Param("sensor_noise", "Sensor noise (gray levels)", "float", 0.0, _S_VIDEO,
          "Standard deviation of additive camera noise (re-drawn each frame).", 0.0, 50.0, structural=True,
          fmt="%.1f"),

    # ── Background ──────────────────────────────────────────────────────
    Param("bg_mode", "Background", "combo", BG_MODES[0], _S_BG, "Solid color or an image stretched to the resolution.",
          items=tuple(BG_MODES), structural=True),
    Param("bg_color", "Background color", "color", [200, 200, 200], _S_BG, "Solid background color.", structural=True),
    Param("bg_image", "Background image", "path", "", _S_BG,
          "Image stretched (not cropped) to fit the chosen resolution.", structural=True),
    Param("blur_bg", "Apply global focus blur to background", "bool", False, _S_BG,
          "Also blur the background with the global focus blur.", structural=True),

    # ── Organisms ───────────────────────────────────────────────────────
    Param("sync_count", "Sync count from CMD", "bool", False, _S_ORG,
          "When receiving CMD from Counter Tracker, automatically synchronize the number of simulated individuals."),
    Param("n_individuals", "Number of individuals", "int", 100, _S_ORG, "Total number of simulated organisms.", 0, 100000,
          structural=True),
    Param("non_motile_pct", "Non-motile (%)", "float", 0.0, _S_ORG,
          "Percentage of individuals that do not swim (moved only by drift / Brownian motion / collisions).",
          0.0, 100.0, structural=True, fmt="%.1f"),
    Param("speed_mean", "Speed mean (µm/s)", "float", 50.0, _S_ORG, "Mean swimming speed of motile individuals.", 0.0,
          100000.0, structural=True, fmt="%.1f"),
    Param("speed_std", "Speed std (µm/s)", "float", 10.0, _S_ORG, "Standard deviation of swimming speed.", 0.0, 100000.0,
          structural=True, fmt="%.1f"),
    Param("size_mean", "Diameter mean (µm)", "float", 10.0, _S_ORG, "Mean organism diameter (largest side for PNG).", 0.1,
          10000.0, structural=True, fmt="%.2f"),
    Param("size_std", "Diameter std (µm)", "float", 1.0, _S_ORG, "Standard deviation of the diameter.", 0.0, 10000.0,
          structural=True, fmt="%.2f"),
    Param("sprite_mode", "Appearance", "combo", SPRITE_MODES[0], _S_ORG, "Textured circle or a PNG image.",
          items=tuple(SPRITE_MODES), structural=True),
    Param("color", "Organism color", "color", [40, 120, 50], _S_ORG, "Base color of the circles.", structural=True),
    Param("sprite_png", "PNG image", "path", "", _S_ORG,
          "Image of the organism (pointing right). Uses its alpha channel; without alpha, the color of the "
          "top-left pixel is treated as transparent background.", structural=True),
    Param("opacity", "Opacity / contrast (%)", "float", 100.0, _S_ORG, "Global opacity of the organisms.", 0.0, 100.0,
          structural=True, fmt="%.0f"),
    Param("tex_noise", "Texture noise (%)", "float", 30.0, _S_ORG,
          "Amplitude of the brightness noise inside the circles (gives optical flow something to track).", 0.0, 100.0,
          structural=True, fmt="%.0f"),
    Param("tex_grain", "Texture grain (px)", "float", 1.0, _S_ORG, "Spatial scale of the texture noise (0 = pixel-wise).",
          0.0, 20.0, structural=True, fmt="%.1f"),
    Param("rotate_with_heading", "Rotate with heading", "bool", False, _S_ORG,
          "Rotate the sprite/texture along the swimming direction (otherwise fixed random orientation).",
          structural=True),

    # ── Motion ──────────────────────────────────────────────────────────
    Param("boundary", "At image border", "combo", BOUNDARY_MODES[0], _S_MOT,
          "Bounce: closed system. Disappear: individuals leaving the image are replaced by new ones entering "
          "from the border (constant population).", items=tuple(BOUNDARY_MODES)),
    Param("collisions", "Collisions", "bool", False, _S_MOT,
          "Individuals bounce on each other. When off they pass through each other."),
    Param("adhesion", "Adhesion (aggregation)", "bool", False, _S_MOT,
          "Individuals that touch stop dead and stay stuck (aggregation). Takes priority over collisions."),
    Param("adhesion_prob", "Adhesion probability (%)", "float", 100.0, _S_MOT,
          "Chance that a contact leads to sticking.", 0.0, 100.0, fmt="%.0f"),
    Param("intention_mode", "Individual intention", "combo", INTENTION_MODES[0], _S_MOT,
          "Preferred swimming direction of each individual: random, or around a common direction.",
          items=tuple(INTENTION_MODES), structural=True),
    Param("intention_angle", "Common direction (°)", "float", 0.0, _S_MOT, "0=right, 90=up, 180=left, 270=down.", 0.0,
          360.0, structural=True, fmt="%.0f"),
    Param("intention_spread", "Direction spread (° std)", "float", 20.0, _S_MOT,
          "Spread of individual intentions around the common direction.", 0.0, 180.0, structural=True, fmt="%.0f"),
    Param("intention_strength", "Intention strength (%)", "float", 100.0, _S_MOT,
          "How strongly individuals steer toward their intention. 0 = no preferred direction "
          "(pure random walk, or pure circles in helical mode).", 0.0, 100.0, fmt="%.0f"),
    Param("turn_gain", "Directional persistence", "float", 0.2, _S_MOT,
          "Fraction of the heading error corrected each frame toward the intention (0 = never, 1 = instantly).",
          0.0, 1.0, fmt="%.2f"),
    Param("turn_noise", "Angular noise (°/frame std)", "float", 5.0, _S_MOT,
          "Random heading deviation between two frames. Low = almost straight swimming.", 0.0, 180.0, fmt="%.1f"),
    Param("helical", "Helical swimming", "bool", False, _S_MOT,
          "Superimpose a rotating component on the swimming direction. With zero intention strength "
          "the organism swims in circles."),
    Param("helix_period", "Helix period (s)", "float", 1.0, _S_MOT, "Duration of one helical turn.", 0.05, 600.0,
          fmt="%.2f"),
    Param("helix_amp", "Helical component", "float", 0.5, _S_MOT,
          "Weight of the rotating component relative to the straight one (higher = wider helix, slower net progress).",
          0.01, 10.0, fmt="%.2f"),
    Param("chirality", "Helix handedness", "combo", CHIRALITY_MODES[0], _S_MOT, "Rotation sense of the helices.",
          items=tuple(CHIRALITY_MODES), structural=True),
    Param("brownian", "Brownian jitter (µm/√s)", "float", 0.0, _S_MOT,
          "Random positional diffusion applied to every free individual.", 0.0, 1000.0, fmt="%.2f"),

    # ── Phototaxis ──────────────────────────────────────────────────────
    Param("photo_angle", "Phototaxis direction (°)", "float", 0.0, _S_PHOTO,
          "Direction toward which the phototaxis vector points (0=right, 90=up, 180=left, 270=down). "
          "Positive or negative phototaxis = choose the angle.", 0.0, 360.0, fmt="%.0f"),
    Param("photo_strength", "Phototaxis strength (%)", "float", 0.0, _S_PHOTO,
          "Weight of the phototaxis vector against each individual's own intention "
          "(100 = everyone obeys, 50 = half influenced, 0 = disabled).", 0.0, 100.0, fmt="%.0f"),

    # ── Drift ───────────────────────────────────────────────────────────
    Param("drift_on", "Enable drift", "bool", False, _S_DRIFT, "Global sedimentation/flow vector applied to everyone."),
    Param("drift_angle", "Drift direction (°)", "float", 270.0, _S_DRIFT, "0=right, 90=up, 180=left, 270=down.", 0.0,
          360.0, fmt="%.0f"),
    Param("drift_speed", "Drift speed (µm/s)", "float", 5.0, _S_DRIFT, "Drift speed.", 0.0, 100000.0, fmt="%.2f"),
    Param("drift_motile_pct", "Drift applied to motile (%)", "float", 100.0, _S_DRIFT,
          "Non-motile individuals always feel 100% of the drift; motile ones feel this fraction.", 0.0, 100.0,
          fmt="%.0f"),

    # ── Focus ───────────────────────────────────────────────────────────
    Param("focus_blur", "Global blur (σ px)", "float", 0.0, _S_FOCUS, "Gaussian blur applied to the whole population.",
          0.0, 50.0, structural=True, fmt="%.2f"),
    Param("oof_pct", "Out-of-focus fraction (%)", "float", 0.0, _S_FOCUS,
          "Share of the population lying outside the focal plane.", 0.0, 100.0, structural=True, fmt="%.0f"),
    Param("oof_blur", "Out-of-focus blur (σ px)", "float", 3.0, _S_FOCUS,
          "Extra Gaussian blur of the out-of-focus individuals (added in quadrature to the global blur).", 0.0, 50.0,
          structural=True, fmt="%.2f"),

    # ── Photoshock ──────────────────────────────────────────────────────
    Param("shock_intensity", "Photoshock intensity (%)", "float", 100.0, _S_SHOCK,
          "Slow-down of motile individuals during the shock (100 = full stop).", 0.0, 100.0, fmt="%.0f"),
    Param("shock_duration", "Photoshock duration (s)", "float", 2.0, _S_SHOCK,
          "Duration of the slow-down. Individuals then turn around for the same duration, "
          "after which they resume their pre-shock direction.", 0.0, 3600.0, fmt="%.2f"),
    Param("shock_times", "Scheduled shocks (s)", "str", "", _S_SHOCK,
          "Comma separated times (e.g. '5, 12.5') at which a photoshock is triggered automatically "
          "(live and export).", structural=True),
]

SECTIONS: List[str] = [_S_VIDEO, _S_BG, _S_ORG, _S_MOT, _S_PHOTO, _S_DRIFT, _S_FOCUS, _S_SHOCK]
PARAM_BY_KEY: Dict[str, Param] = {p.key: p for p in PARAMS}


def defaults() -> Dict[str, Any]:
    """Return a fresh dict with every parameter set to its default value."""
    out: Dict[str, Any] = {}
    for p in PARAMS:
        out[p.key] = list(p.default) if isinstance(p.default, list) else p.default
    return out



# Only these need a restart (they change the canvas / time base / random stream). Everything else is live.
RESTART_KEYS = {"width", "height", "fps", "calibration", "seed"}
