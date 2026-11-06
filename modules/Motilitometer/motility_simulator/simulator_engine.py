"""Headless synthetic motility engine (numpy + OpenCV, no GUI dependency).

The engine simulates a population of swimming / drifting organisms with fully known
parameters, renders them as BGR uint8 frames and records the exact ground truth.

Conventions
-----------
* Angles are *mathematical* (counter-clockwise as seen on screen, 0 = right).
  Image displacements therefore use ``dy = -sin(angle)``.
* Internal units are pixels and frames; user units are µm and seconds
  (``conv = 1 / (calibration * fps)`` turns µm/s into px/frame).

Performance notes
-----------------
* Physics is fully vectorised over the population (numpy); pair search uses a
  ``scipy.spatial.cKDTree`` when available (blockwise numpy fallback otherwise).
* Every sprite (texture + blur) is rendered *once*; a frame only costs one small
  ``cv2.warpAffine`` (sub-pixel placement and optional rotation) plus an in-place
  alpha blend per individual, on a reused float32 canvas.
"""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # optional, much faster neighbour search for large populations
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from loguru import logger

from .simulator_params import (
    BG_MODES, BOUNDARY_MODES, CHIRALITY_MODES, INTENTION_MODES, RESTART_KEYS, SPRITE_MODES, defaults,
)

TWO_PI = 2.0 * math.pi
MAX_UNIQUE_SPRITES = 1500  # bigger populations re-use sprites (memory bound)

GT_COLUMNS = [
    "frame", "time_s", "id", "x_px", "y_px", "x_um", "y_um", "vx_px_per_frame", "vy_px_per_frame",
    "speed_um_s", "heading_deg", "radius_px", "diameter_um", "motile", "stuck", "out_of_focus",
    "blur_sigma_px", "shock_phase",
]
GT_FORMATS = ["%d", "%.6f", "%d", "%.4f", "%.4f", "%.4f", "%.4f", "%.5f", "%.5f", "%.4f", "%.3f", "%.4f", "%.4f",
              "%d", "%d", "%d", "%.3f", "%d"]


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (a + math.pi) % TWO_PI - math.pi


def _rgb_to_bgr(color: Any) -> np.ndarray:
    c = np.asarray(color, dtype=np.float32).reshape(-1)[:3]  # RGB, 0..255
    return np.clip(c[::-1], 0, 255).astype(np.float32)


def load_rgba(path: str) -> np.ndarray:
    """Load an image as BGRA uint8. Without alpha, the top-left pixel colour is made transparent."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img
    corner = img[0, 0].astype(np.float32)
    dist = np.linalg.norm(img.astype(np.float32) - corner, axis=2)
    alpha = np.clip(dist / 40.0, 0.0, 1.0)
    return np.dstack([img, (alpha * 255).astype(np.uint8)])


class SimulationEngine:
    """Population simulation + renderer. Not thread-safe except through its public methods."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        full = defaults()
        if cfg:
            full.update({k: v for k, v in cfg.items() if k in full})
        self.cfg = full
        self._lock = threading.RLock()

        self.W = int(full["width"])
        self.H = int(full["height"])
        self.N = max(0, int(full["n_individuals"]))
        self.fps = max(0.1, float(full["fps"]))
        self.calib = max(1e-6, float(full["calibration"]))
        self.dt = 1.0 / self.fps
        self.conv = 1.0 / (self.calib * self.fps)  # µm/s -> px/frame
        seed = int(full["seed"])
        self.rng = np.random.default_rng(seed if seed > 0 else None)
        self._tex_seed = int(self.rng.integers(1 << 30))  # per-sprite texture seeds stay stable across rebuilds
        self.frame = 0
        self._dirty: set = set()
        self.last_error = ""
        self._shock: Optional[Dict[str, Any]] = None
        self._shock_phase = 0
        self._shock_frames: set = set()

        self._init_population()
        self._build_background()
        self._build_sprites()
        self.set_dynamic(full)
        self._parse_shocks()

        self._canvas = np.empty((self.H, self.W, 3), dtype=np.float32)
        self._affine_M = np.zeros((2, 3), dtype=np.float64)
        self._warp_flags = cv2.INTER_LINEAR

    def _parse_shocks(self) -> None:
        sched = []
        for tok in str(self.cfg.get("shock_times", "") or "").replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    sched.append(int(round(float(tok) * self.fps)))
                except ValueError:
                    pass
        self._shock_frames = set(sched)

    # ── construction ──────────────────────────────────────────────────────
    def _new_genes(self, n: int) -> Dict[str, np.ndarray]:
        """Random draws attached to each individual; parameters are derived from them, so changing a
        parameter live moves every individual smoothly instead of re-drawing the population."""
        r = self.rng
        return {
            "z_size": r.standard_normal(n), "u_oof": r.random(n), "z_speed": r.standard_normal(n),
            "u_motile": r.random(n), "u_int": r.uniform(-math.pi, math.pi, n), "z_int": r.standard_normal(n),
            "u_chir": r.random(n), "u_period": r.uniform(0.8, 1.2, n),
        }

    def _init_population(self) -> None:
        rng, N = self.rng, self.N
        self.genes = self._new_genes(N)
        self.pos = np.empty((N, 2), dtype=np.float64)
        self.pos[:, 0] = rng.uniform(0, self.W, N)
        self.pos[:, 1] = rng.uniform(0, self.H, N)
        self.psi = rng.uniform(0, TWO_PI, N)
        self.vis_theta = rng.uniform(-math.pi, math.pi, N)
        self.stuck = np.zeros(N, dtype=bool)
        self.kick = np.zeros((N, 2))
        self.vel = np.zeros((N, 2))
        self.ids = np.arange(N, dtype=np.int64)
        self._next_id = N
        self.theta = np.zeros(N)
        self._derive_all()
        self._clamp_inside()

        c = self.cfg
        self.photo_s = float(c["photo_strength"]) / 100.0
        self.photo_a = math.radians(float(c["photo_angle"]))
        self.int_k = float(c["intention_strength"]) / 100.0
        self.theta = self._target()[0].copy()

        if (c["collisions"] or c["adhesion"]) and N > 1:
            self._relax()

    # derived per-individual properties --------------------------------------
    def _derive_all(self) -> None:
        self._derive_bank()
        self._derive_motion()
        self._derive_intention()
        self._derive_helix()

    def _derive_bank(self) -> None:
        c, N, g = self.cfg, self.N, self.genes
        u = min(N, MAX_UNIQUE_SPRITES)
        self.unique = u
        self.sprite_idx = np.arange(N) % max(u, 1)
        diam_um = np.maximum(float(c["size_mean"]) + float(c["size_std"]) * g["z_size"][:u], 2.0 * self.calib)
        self._bank_radius = np.maximum(diam_um / 2.0 / self.calib, 1.0)
        n_oof = int(round(u * float(c["oof_pct"]) / 100.0))
        rank = np.argsort(np.argsort(g["u_oof"][:u]))
        self._bank_oof = rank < n_oof
        gb, ob = float(c["focus_blur"]), float(c["oof_blur"])
        self._bank_sigma = np.sqrt(gb * gb + (ob * ob) * self._bank_oof)
        self.radius = self._bank_radius[self.sprite_idx]
        self.sigma = self._bank_sigma[self.sprite_idx]
        self.oof = self._bank_oof[self.sprite_idx]

    def _derive_motion(self) -> None:
        c, N, g = self.cfg, self.N, self.genes
        n_non = int(round(N * float(c["non_motile_pct"]) / 100.0))
        self.motile = np.argsort(np.argsort(g["u_motile"])) >= n_non
        self.speed = self._speed_for(slice(None))

    def _speed_for(self, idx: Any) -> np.ndarray:
        c = self.cfg
        s = np.clip(float(c["speed_mean"]) + float(c["speed_std"]) * self.genes["z_speed"][idx], 0.0, None)
        return s * self.conv * self.motile[idx]

    def _phi_for(self, idx: Any) -> np.ndarray:
        c, g = self.cfg, self.genes
        if c["intention_mode"] == INTENTION_MODES[1]:
            return np.radians(float(c["intention_angle"]) + float(c["intention_spread"]) * g["z_int"][idx])
        return g["u_int"][idx].copy()

    def _derive_intention(self) -> None:
        self.phi = self._phi_for(slice(None))

    def _derive_helix(self) -> None:
        c, g = self.cfg, self.genes
        sign = {CHIRALITY_MODES[1]: -1.0, CHIRALITY_MODES[2]: 1.0}.get(c["chirality"])
        self.chir = np.full(self.N, sign) if sign is not None else np.where(g["u_chir"] < 0.5, -1.0, 1.0)
        period = max(0.05, float(c["helix_period"])) * g["u_period"]
        self.omega = self.chir * TWO_PI / (period * self.fps)

    # live resize ------------------------------------------------------------
    _STATE_ARRAYS = ("pos", "psi", "vis_theta", "stuck", "kick", "vel", "ids", "theta")

    def _resize(self, new_n: int) -> None:
        old = self.N
        new_n = max(0, int(new_n))
        if new_n == old:
            return
        if new_n < old:
            for k in self.genes:
                self.genes[k] = self.genes[k][:new_n]
            for name in self._STATE_ARRAYS:
                setattr(self, name, getattr(self, name)[:new_n])
        else:
            n = new_n - old
            add = self._new_genes(n)
            for k in self.genes:
                self.genes[k] = np.concatenate([self.genes[k], add[k]])
            rng = self.rng
            extra = {
                "pos": np.column_stack([rng.uniform(0, self.W, n), rng.uniform(0, self.H, n)]),
                "psi": rng.uniform(0, TWO_PI, n), "vis_theta": rng.uniform(-math.pi, math.pi, n),
                "stuck": np.zeros(n, dtype=bool), "kick": np.zeros((n, 2)), "vel": np.zeros((n, 2)),
                "ids": np.arange(self._next_id, self._next_id + n, dtype=np.int64), "theta": np.zeros(n),
            }
            self._next_id += n
            for name in self._STATE_ARRAYS:
                setattr(self, name, np.concatenate([getattr(self, name), extra[name]]))
        self.N = new_n
        self._derive_all()
        self._build_sprites()
        if new_n > old:
            self.theta[old:] = self._target()[0][old:]
            self._clamp_inside()

    def _clamp_inside(self) -> None:
        r = self.radius
        self.pos[:, 0] = np.clip(self.pos[:, 0], r, np.maximum(self.W - r, r))
        self.pos[:, 1] = np.clip(self.pos[:, 1], r, np.maximum(self.H - r, r))

    def _relax(self, iterations: int = 30) -> None:
        """Remove initial overlaps so adhesion does not trigger at t=0."""
        for _ in range(iterations):
            i, j, d, dist = self._pairs()
            if i.size == 0:
                return
            n = d / np.maximum(dist, 1e-9)[:, None]
            zero = dist < 1e-9
            if zero.any():
                ang = self.rng.uniform(0, TWO_PI, int(zero.sum()))
                n[zero] = np.column_stack([np.cos(ang), np.sin(ang)])
            ov = (self.radius[i] + self.radius[j] - dist) * 0.5 + 0.05
            for ax in (0, 1):
                self.pos[:, ax] += np.bincount(j, n[:, ax] * ov, self.N) - np.bincount(i, n[:, ax] * ov, self.N)
            self._clamp_inside()

    def _build_background(self) -> None:
        c, W, H = self.cfg, self.W, self.H
        sg = float(c["focus_blur"]) if c["blur_bg"] else 0.0
        bg = None
        if c["bg_mode"] == BG_MODES[1] and c["bg_image"]:
            img = cv2.imread(str(c["bg_image"]), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read background image: {c['bg_image']}")
            interp = cv2.INTER_AREA if (img.shape[1] > W or img.shape[0] > H) else cv2.INTER_LINEAR
            bg = cv2.resize(img, (W, H), interpolation=interp).astype(np.float32)
        elif len(BG_MODES) > 2 and c["bg_mode"] == BG_MODES[2]:
            bg = np.zeros((H, W, 3), dtype=np.float32)
        if bg is None:
            bg = np.empty((H, W, 3), dtype=np.float32)
            bg[:] = _rgb_to_bgr(c["bg_color"])
        if sg > 0.05:
            bg = cv2.GaussianBlur(bg, (0, 0), sg)
        self.bg = np.ascontiguousarray(bg)

        self._noise_bank = None
        sn = float(c["sensor_noise"])
        if sn > 0:
            self._noise_bank = [self.rng.normal(0, sn, (H, W, 1)).astype(np.float32) for _ in range(4)]

    def _build_sprites(self) -> None:
        c, rng = self.cfg, self.rng
        base = _rgb_to_bgr(c["color"])
        opac = float(c["opacity"]) / 100.0
        amp = float(c["tex_noise"]) / 100.0
        grain = float(c["tex_grain"])
        self.rotate = bool(c["rotate_with_heading"])
        use_png = c["sprite_mode"] == SPRITE_MODES[1] and bool(c["sprite_png"])
        png = load_rgba(str(c["sprite_png"])) if use_png else None
        png_cache: Dict[Tuple[int, int], np.ndarray] = {}
        dist_cache: Dict[int, np.ndarray] = {}

        self._sprites: List[np.ndarray] = []
        self.last_error = ""
        self._cen = np.empty(self.unique, dtype=np.float64)
        self._sizes = np.empty(self.unique, dtype=np.int64)
        is_ring = len(SPRITE_MODES) > 2 and (c.get("sprite_mode") == SPRITE_MODES[2])
        for k in range(self.unique):
            krng = np.random.default_rng(self._tex_seed + k)
            r, sg = float(self._bank_radius[k]), float(self._bank_sigma[k])
            pad = int(math.ceil(3.0 * sg)) + 2
            if png is None:
                half = int(math.ceil(r)) + pad
                S = 2 * half + 1
                dmap = dist_cache.get(S)
                if dmap is None:
                    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
                    dmap = np.hypot(xx - half, yy - half)
                    dist_cache[S] = dmap
                if is_ring:
                    thickness = max(1.5, r * 0.3)
                    alpha = np.clip(1.0 - np.abs(dmap - (r - thickness / 2.0)) / (thickness / 2.0), 0.0, 1.0)
                    col = np.broadcast_to(base, (S, S, 3))
                else:
                    alpha = np.clip(r - dmap + 0.5, 0.0, 1.0)
                if amp > 0:
                    noise = krng.standard_normal((S, S)).astype(np.float32)
                    if grain > 0.05:
                        noise = cv2.GaussianBlur(noise, (0, 0), grain)
                    noise /= (float(noise.std()) + 1e-6)
                    lum = np.clip(1.0 + amp * noise, 0.0, 2.0)
                    col = np.clip(base[None, None, :] * lum[..., None], 0, 255)
                else:
                    col = np.broadcast_to(base, (S, S, 3))
                a = alpha * opac
                spr = np.empty((S, S, 4), dtype=np.float32)
                spr[..., :3] = col * a[..., None]
                spr[..., 3] = a
            else:
                scale = 2.0 * r / max(png.shape[0], png.shape[1])
                nw, nh = max(1, int(round(png.shape[1] * scale))), max(1, int(round(png.shape[0] * scale)))
                rs = png_cache.get((nw, nh))
                if rs is None:
                    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                    rs = cv2.resize(png, (nw, nh), interpolation=interp).astype(np.float32)
                    rs[..., :3] *= (rs[..., 3:4] / 255.0)       # premultiply
                    rs[..., 3] /= 255.0
                    png_cache[(nw, nh)] = rs
                half = int(math.ceil(max(nw, nh) * 0.7072)) + pad + 1
                S = 2 * half + 1
                spr = np.zeros((S, S, 4), dtype=np.float32)
                ty, tx = half - nh // 2, half - nw // 2
                spr[ty:ty + nh, tx:tx + nw] = rs * opac
                if not self.rotate:
                    ang = math.degrees(krng.uniform(0, TWO_PI))
                    M = cv2.getRotationMatrix2D((float(half), float(half)), ang, 1.0)
                    spr = cv2.warpAffine(spr, M, (S, S), flags=cv2.INTER_LINEAR)
            if sg > 0.05:
                spr = cv2.GaussianBlur(spr, (0, 0), sg)
            self._sprites.append(np.ascontiguousarray(spr))
            self._cen[k] = float(half)
            self._sizes[k] = spr.shape[0]

    # ── live-tunable parameters ───────────────────────────────────────────
    def set_dynamic(self, cfg: Dict[str, Any]) -> None:
        """Apply the non-structural parameters (safe to call while running)."""
        with self._lock:
            for k, v in cfg.items():
                if k in self.cfg:
                    self.cfg[k] = v
            c = self.cfg
            self.boundary_bounce = c["boundary"] == BOUNDARY_MODES[0]
            self.collisions = bool(c["collisions"])
            self.adhesion = bool(c["adhesion"])
            if not self.adhesion:
                self.stuck[:] = False  # releasing adhesion frees aggregated individuals
            self.adh_prob = float(c["adhesion_prob"]) / 100.0
            self.int_k = float(c["intention_strength"]) / 100.0
            self.gain = float(c["turn_gain"])
            self.turn_noise = math.radians(float(c["turn_noise"]))
            self.helical = bool(c["helical"])
            self.helix_amp = max(1e-3, float(c["helix_amp"]))
            self.brown_px = float(c["brownian"]) * math.sqrt(self.dt) / self.calib
            self.photo_s = float(c["photo_strength"]) / 100.0
            self.photo_a = math.radians(float(c["photo_angle"]))
            self.drift_on = bool(c["drift_on"])
            a = math.radians(float(c["drift_angle"]))
            sp = float(c["drift_speed"]) * self.conv
            self.drift_vec = np.array([math.cos(a) * sp, -math.sin(a) * sp])
            frac = float(c["drift_motile_pct"]) / 100.0
            self.drift_w = np.where(self.motile, frac, 1.0)
            self.shock_I = float(c["shock_intensity"]) / 100.0
            self.shock_D = float(c["shock_duration"])

    # ── live update of every parameter except the restart-only ones ───────
    _GROUPS = {
        "n": {"n_individuals"},
        "app": {"size_mean", "size_std", "oof_pct", "oof_blur", "focus_blur", "sprite_mode", "color", "sprite_png",
                "opacity", "tex_noise", "tex_grain", "rotate_with_heading"},
        "bg": {"bg_mode", "bg_color", "bg_image", "blur_bg", "focus_blur", "sensor_noise"},
        "mot": {"non_motile_pct", "speed_mean", "speed_std"},
        "int": {"intention_mode", "intention_angle", "intention_spread"},
        "hel": {"chirality", "helix_period"},
        "shock": {"shock_times"},
    }

    def update_params(self, changes: Dict[str, Any]) -> None:
        # Live calibration & fps recalculation if passed
        recalc_conv = False
        if "calibration" in changes and float(changes["calibration"]) > 0:
            self.cfg["calibration"] = float(changes["calibration"])
            self.calib = float(self.cfg["calibration"])
            recalc_conv = True
        if "fps" in changes and float(changes["fps"]) > 0:
            self.cfg["fps"] = float(changes["fps"])
            self.fps = float(self.cfg["fps"])
            self.dt = 1.0 / self.fps
            recalc_conv = True
        if recalc_conv:
            self.conv = 1.0 / (self.calib * self.fps)
            self._dirty.add("mot")
        """Queue parameter changes; heavy rebuilds are applied by the simulation thread (apply_pending)."""
        with self._lock:
            for k, v in changes.items():
                if k in RESTART_KEYS or k not in self.cfg:
                    continue
                self.cfg[k] = v
                for grp, keys in self._GROUPS.items():
                    if k in keys:
                        self._dirty.add(grp)
            self.set_dynamic({})

    def apply_pending(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            d, self._dirty = self._dirty, set()
            try:
                if "n" in d:
                    self._resize(int(self.cfg["n_individuals"]))
                    d -= {"mot", "int", "hel", "app"}
                if "mot" in d:
                    self._derive_motion()
                if "int" in d:
                    self._derive_intention()
                if "hel" in d:
                    self._derive_helix()
                if "app" in d:
                    self._derive_bank()
                    self._build_sprites()
                    self._clamp_inside()
                if "bg" in d:
                    self._build_background()
                if "shock" in d:
                    self._parse_shocks()
                self.set_dynamic({})
            except Exception as e:  # keep the simulation alive on bad input (e.g. unreadable image)
                self.last_error = str(e)
                logger.error(f"Motility simulator live update failed: {e}")

    # ── photoshock ────────────────────────────────────────────────────────
    def trigger_photoshock(self, intensity_pct: Optional[float] = None, duration_s: Optional[float] = None) -> None:
        with self._lock:
            self._start_shock(intensity_pct, duration_s)

    def _start_shock(self, intensity_pct: Optional[float] = None, duration_s: Optional[float] = None) -> None:
        inten = self.shock_I if intensity_pct is None else float(np.clip(intensity_pct / 100.0, 0, 1))
        dur = self.shock_D if duration_s is None else float(duration_s)
        D = int(round(dur * self.fps))
        if D <= 0:
            return
        # slow-down for D frames, then U-turn for D frames, then the original intention resumes
        self._shock = {"t": 0, "D": D, "I": inten}

    @property
    def shock_active(self) -> bool:
        return self._shock is not None

    # ── physics ───────────────────────────────────────────────────────────
    def _target(self) -> Tuple[np.ndarray, np.ndarray]:
        """Effective intention (angle, magnitude in [0,1]): individual vs phototaxis vector."""
        s, k = self.photo_s, self.int_k
        tx = (1.0 - s) * k * np.cos(self.phi) + s * math.cos(self.photo_a)
        ty = (1.0 - s) * k * np.sin(self.phi) + s * math.sin(self.photo_a)
        mag = np.hypot(tx, ty)
        tang = np.where(mag > 1e-9, np.arctan2(ty, tx), self.theta)
        return tang, np.minimum(mag, 1.0)

    def step(self) -> None:
        with self._lock:
            self._step()

    def _step(self) -> None:
        self.apply_pending()
        N = self.N
        if N == 0:
            self.frame += 1
            return
        rng = self.rng
        if self.frame in self._shock_frames:
            self._start_shock()

        free = ~self.stuck
        mot = self.motile & free
        tang, mag = self._target()

        speed_mult, phase = 1.0, 0
        sh = self._shock
        if sh is not None:
            t, D = sh["t"], sh["D"]
            if t < D:
                speed_mult, phase = 1.0 - sh["I"], 1
            elif t < 2 * D:
                tang = tang + math.pi  # U-turn; the pre-shock direction resumes afterwards
                phase = 2
            sh["t"] = t + 1
            if sh["t"] >= 2 * D:
                self._shock = None
        self._shock_phase = phase

        # heading: relaxation toward intention + random deviation
        dth = self.gain * mag * _wrap(tang - self.theta) + rng.normal(0.0, self.turn_noise, N)
        self.theta += np.where(mot, dth, 0.0)

        if self.helical:
            self.psi += self.omega
            den = mag + self.helix_amp
            vx = (mag * np.cos(self.theta) + self.helix_amp * np.cos(self.theta + self.psi)) / den
            vy = (mag * np.sin(self.theta) + self.helix_amp * np.sin(self.theta + self.psi)) / den
        else:
            vx, vy = np.cos(self.theta), np.sin(self.theta)

        sp = self.speed * speed_mult * mot
        dx, dy = vx * sp, -vy * sp
        moving = sp > 0
        self.vis_theta = np.where(moving, np.arctan2(vy, vx), self.vis_theta)

        if self.drift_on:
            dx += self.drift_vec[0] * self.drift_w
            dy += self.drift_vec[1] * self.drift_w
        if self.brown_px > 0:
            dx += rng.normal(0.0, self.brown_px, N)
            dy += rng.normal(0.0, self.brown_px, N)
        dx += self.kick[:, 0]
        dy += self.kick[:, 1]
        self.kick *= 0.85
        dx *= free
        dy *= free

        self.pos[:, 0] += dx
        self.pos[:, 1] += dy

        if self.boundary_bounce:
            self._bounce()
        else:
            self._respawn_outside()

        if self.collisions or self.adhesion:
            self._interact(dx, dy)

        self.vel[:, 0] = dx
        self.vel[:, 1] = dy
        self.frame += 1

    def _bounce(self) -> None:
        r = self.radius
        mot = self.motile
        for ax, size in ((0, self.W), (1, self.H)):
            p = self.pos[:, ax]
            hi = np.maximum(size - r, r)
            lo_hit, hi_hit = p < r, p > hi
            hit = lo_hit | hi_hit
            if not hit.any():
                continue
            refl = np.where(lo_hit, 2 * r - p, np.where(hi_hit, 2 * hi - p, p))
            clamped = np.clip(p, r, hi)
            newp = np.where(mot, refl, clamped)  # non-motile settle on the wall (sedimentation)
            self.pos[:, ax] = np.clip(newp, r, hi)
            hm = hit & mot
            if hm.any():
                self.theta[hm] = (math.pi - self.theta[hm]) if ax == 0 else -self.theta[hm]
                self.psi[hm] *= -1.0
                self.omega[hm] *= -1.0
            self.kick[hit, ax] *= -1.0

    def _respawn_outside(self) -> None:
        r, W, H = self.radius, self.W, self.H
        x, y = self.pos[:, 0], self.pos[:, 1]
        out = ((x < -r) | (x > W + r) | (y < -r) | (y > H + r)) & ~self.stuck
        idx = np.nonzero(out)[0]
        if idx.size:
            self._respawn(idx)

    def _respawn(self, idx: np.ndarray) -> None:
        """Replace individuals that left the image by new ones entering through the border."""
        rng, n = self.rng, idx.size
        c = self.cfg
        mot = self.motile[idx]
        g = self.genes
        g["z_speed"][idx] = rng.standard_normal(n)
        g["u_int"][idx] = rng.uniform(-math.pi, math.pi, n)
        g["z_int"][idx] = rng.standard_normal(n)
        self.phi[idx] = self._phi_for(idx)
        spd = self._speed_for(idx)
        self.speed[idx] = spd
        self.psi[idx] = rng.uniform(0, TWO_PI, n)
        self.kick[idx] = 0.0
        self.stuck[idx] = False
        self.ids[idx] = np.arange(self._next_id, self._next_id + n)
        self._next_id += n

        s, k = self.photo_s, self.int_k
        tx = (1 - s) * k * np.cos(self.phi[idx]) + s * math.cos(self.photo_a)
        ty = (1 - s) * k * np.sin(self.phi[idx]) + s * math.sin(self.photo_a)
        ang = np.where(np.hypot(tx, ty) > 1e-9, np.arctan2(ty, tx), self.phi[idx])
        self.theta[idx] = ang

        vx = np.cos(ang) * spd + (self.drift_vec[0] * self.drift_w[idx] if self.drift_on else 0.0)
        vy = -np.sin(ang) * spd + (self.drift_vec[1] * self.drift_w[idx] if self.drift_on else 0.0)
        # inward flux through each edge: left, right, top, bottom
        w = np.column_stack([np.maximum(vx, 0) * self.H, np.maximum(-vx, 0) * self.H,
                             np.maximum(vy, 0) * self.W, np.maximum(-vy, 0) * self.W])
        tot = w.sum(axis=1)
        r = self.radius[idx]
        u = rng.random(n)
        edge = np.where(tot > 1e-12, (np.cumsum(w, axis=1) < (u * tot)[:, None]).sum(axis=1), -1)
        edge = np.minimum(edge, 3)
        along = rng.random(n)
        e = edge
        px = np.where(e == 0, -r, np.where(e == 1, self.W + r, along * self.W))
        py = np.where(e == 2, -r, np.where(e == 3, self.H + r, along * self.H))
        free_pos = e < 0  # no net motion at all: reappear anywhere
        if free_pos.any():
            px = np.where(free_pos, rng.random(n) * self.W, px)
            py = np.where(free_pos, rng.random(n) * self.H, py)
        self.pos[idx, 0] = px
        self.pos[idx, 1] = py

    def _pairs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Indices (i<j) of touching pairs plus their separation vector and distance."""
        N, pos, r = self.N, self.pos, self.radius
        empty = np.empty(0, dtype=np.int64)
        if N < 2:
            return empty, empty, np.empty((0, 2)), np.empty(0)
        reach = 2.0 * float(r.max())
        if cKDTree is not None:
            pr = cKDTree(pos).query_pairs(reach, output_type="ndarray")
            if pr.shape[0] == 0:
                return empty, empty, np.empty((0, 2)), np.empty(0)
            i, j = pr[:, 0], pr[:, 1]
        else:
            block = max(1, int(4e6 // N))
            parts_i, parts_j = [], []
            for s in range(0, N, block):
                d2 = ((pos[s:s + block, None, :] - pos[None, :, :]) ** 2).sum(-1)
                ii, jj = np.nonzero(d2 < reach * reach)
                ii = ii + s
                m = jj > ii
                parts_i.append(ii[m])
                parts_j.append(jj[m])
            i, j = np.concatenate(parts_i), np.concatenate(parts_j)
            if i.size == 0:
                return empty, empty, np.empty((0, 2)), np.empty(0)
        d = pos[j] - pos[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        ok = dist < (r[i] + r[j])
        return i[ok], j[ok], d[ok], dist[ok]

    def _interact(self, dx: np.ndarray, dy: np.ndarray) -> None:
        N = self.N
        i, j, d, dist = self._pairs()
        if i.size == 0:
            return
        if self.adhesion:
            sel = np.ones(i.size, dtype=bool) if self.adh_prob >= 1.0 else self.rng.random(i.size) < self.adh_prob
            if sel.any():
                for arr in (i[sel], j[sel]):
                    self.stuck[arr] = True
                    self.kick[arr] = 0.0
                dx[self.stuck] = 0.0
                dy[self.stuck] = 0.0
            keep = ~sel
            i, j, d, dist = i[keep], j[keep], d[keep], dist[keep]
        if not self.collisions or i.size == 0:
            return

        si, sj = self.stuck[i], self.stuck[j]
        keep = ~(si & sj)
        if not keep.all():
            i, j, d, dist, si, sj = i[keep], j[keep], d[keep], dist[keep], si[keep], sj[keep]
            if i.size == 0:
                return
        r = self.radius
        n = d / np.maximum(dist, 1e-9)[:, None]
        zero = dist < 1e-9
        if zero.any():
            n[zero] = (1.0, 0.0)

        # positional correction (stuck individuals never move)
        wi = np.where(si, 0.0, np.where(sj, 1.0, 0.5))
        wj = 1.0 - wi
        ov = r[i] + r[j] - dist + 1e-3
        for ax in (0, 1):
            self.pos[:, ax] += np.bincount(j, n[:, ax] * ov * wj, N) - np.bincount(i, n[:, ax] * ov * wi, N)

        # elastic exchange along the contact normal (mass ~ area)
        vi = np.column_stack([dx[i], dy[i]])
        vj = np.column_stack([dx[j], dy[j]])
        rel = ((vi - vj) * n).sum(axis=1)
        appr = rel > 0
        if not appr.any():
            return
        mi = np.where(si, 1e9, r[i] ** 2)
        mj = np.where(sj, 1e9, r[j] ** 2)
        J = np.where(appr, 2.0 * rel / (mi + mj), 0.0)
        dvx = np.bincount(i, -J * mj * n[:, 0], N) + np.bincount(j, J * mi * n[:, 0], N)
        dvy = np.bincount(i, -J * mj * n[:, 1], N) + np.bincount(j, J * mi * n[:, 1], N)
        hit = ((dvx != 0) | (dvy != 0)) & ~self.stuck
        if not hit.any():
            return

        # motile: deflect the swimming direction; non-motile: damped momentum kick
        hm = hit & self.motile
        sp0 = np.hypot(dx, dy)
        turn = hm & (sp0 > 1e-9)
        if turn.any():
            a0 = np.arctan2(-dy[turn], dx[turn])
            a1 = np.arctan2(-(dy[turn] + dvy[turn]), dx[turn] + dvx[turn])
            self.theta[turn] += _wrap(a1 - a0)
        hn = hit & ~self.motile
        self.kick[hn, 0] += dvx[hn]
        self.kick[hn, 1] += dvy[hn]
        if self.boundary_bounce:
            self._clamp_inside()

    # ── rendering ─────────────────────────────────────────────────────────
    def render(self, bg_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Render the current state as a new BGR uint8 frame."""
        with self._lock:
            return self._render(bg_frame=bg_frame)

    def _render(self, bg_frame: Optional[np.ndarray] = None) -> np.ndarray:
        self.apply_pending()
        W, H = self.W, self.H
        f = self._canvas
        if bg_frame is not None:
            f_h, f_w = int(bg_frame.shape[0]), int(bg_frame.shape[1])
            if f_h != H or f_w != W:
                self.H, self.W = f_h, f_w
                W, H = self.W, self.H
                self._canvas = np.empty((H, W, 3), dtype=np.float32)
                self._clamp_inside()
            f = self._canvas
            if bg_frame.ndim == 2:
                np.copyto(f, cv2.cvtColor(bg_frame, cv2.COLOR_GRAY2BGR), casting="unsafe")
            else:
                np.copyto(f, bg_frame, casting="unsafe")
        else:
            np.copyto(f, self.bg)
        if bg_frame is None and self._noise_bank is not None:
            nz = self._noise_bank[int(self.rng.integers(4))]
            if self.rng.random() < 0.5:
                nz = nz[:, ::-1]
            if self.rng.random() < 0.5:
                nz = nz[::-1]
            f += nz

        N = self.N
        if N:
            sidx = self.sprite_idx
            cen = self._cen[sidx]
            x0f = self.pos[:, 0] - cen
            y0f = self.pos[:, 1] - cen
            ix, iy = np.floor(x0f), np.floor(y0f)
            fx, fy = (x0f - ix).tolist(), (y0f - iy).tolist()
            ixl, iyl = ix.astype(np.int64).tolist(), iy.astype(np.int64).tolist()
            sizes = self._sizes[sidx].tolist()
            sidxl = sidx.tolist()
            cenl = cen.tolist()
            sprites = self._sprites
            warp = cv2.warpAffine
            flags = self._warp_flags
            M = self._affine_M
            if self.rotate:
                cosl, sinl = np.cos(self.vis_theta).tolist(), np.sin(self.vis_theta).tolist()
            else:
                M[0, 0] = 1.0
                M[0, 1] = 0.0
                M[1, 0] = 0.0
                M[1, 1] = 1.0
            for k in range(N):
                S = sizes[k]
                x0, y0 = ixl[k], iyl[k]
                xa, ya = max(x0, 0), max(y0, 0)
                xb, yb = min(x0 + S, W), min(y0 + S, H)
                if xa >= xb or ya >= yb:
                    continue
                if self.rotate:
                    a_, b_, c_ = cosl[k], sinl[k], cenl[k]
                    M[0, 0] = a_
                    M[0, 1] = b_
                    M[0, 2] = (1.0 - a_) * c_ - b_ * c_ + fx[k]
                    M[1, 0] = -b_
                    M[1, 1] = a_
                    M[1, 2] = b_ * c_ + (1.0 - a_) * c_ + fy[k]
                else:
                    M[0, 2] = fx[k]
                    M[1, 2] = fy[k]
                patch = warp(sprites[sidxl[k]], M, (S, S), flags=flags)
                p = patch[ya - y0:yb - y0, xa - x0:xb - x0]
                roi = f[ya:yb, xa:xb]
                roi -= roi * p[:, :, 3:4]
                roi += p[:, :, :3]
        return cv2.convertScaleAbs(f)

    # ── ground truth ──────────────────────────────────────────────────────
    def ground_truth(self) -> np.ndarray:
        """(M, len(GT_COLUMNS)) float array for the individuals visible in the current frame."""
        with self._lock:
            r = self.radius
            x, y = self.pos[:, 0], self.pos[:, 1]
            vis = (x + r > 0) & (x - r < self.W) & (y + r > 0) & (y - r < self.H)
            idx = np.nonzero(vis)[0]
            vx, vy = self.vel[idx, 0], self.vel[idx, 1]
            sp_um = np.hypot(vx, vy) / self.conv
            head = np.degrees(np.arctan2(-vy, vx))
            m = idx.size
            rows = np.column_stack([
                np.full(m, self.frame), np.full(m, self.frame * self.dt),
                self.ids[idx], x[idx], y[idx], x[idx] * self.calib, y[idx] * self.calib, vx, vy, sp_um, head,
                r[idx], 2 * r[idx] * self.calib, self.motile[idx], self.stuck[idx], self.oof[idx],
                self.sigma[idx], np.full(m, self._shock_phase),
            ])
            return rows

    def live_metrics(self) -> Dict[str, float]:
        """Returns live ground truth statistics for the individuals visible in the current frame."""
        with self._lock:
            r = self.radius
            x, y = self.pos[:, 0], self.pos[:, 1]
            vis = (x + r > 0) & (x - r < self.W) & (y + r > 0) & (y - r < self.H)
            idx = np.nonzero(vis)[0]
            n_vis = int(idx.size)
            if n_vis == 0:
                return {"count": 0, "pct_motile": 0.0, "mean_speed_um": 0.0}
            vx, vy = self.vel[idx, 0], self.vel[idx, 1]
            sp_um = np.hypot(vx, vy) / self.conv
            is_mot = self.motile[idx] & (~self.stuck[idx])
            pct_mot = float(np.count_nonzero(is_mot) / n_vis * 100.0)
            mean_spd = float(np.mean(sp_um))
            return {
                "count": n_vis,
                "pct_motile": pct_mot,
                "mean_speed_um": mean_spd,
            }

    def params_dict(self) -> Dict[str, Any]:
        d = dict(self.cfg)
        d["derived"] = {"um_per_s_to_px_per_frame": self.conv, "n_sprites_unique": self.unique,
                        "motile_count": int(self.motile.sum()), "scipy_pair_search": cKDTree is not None}
        d["conventions"] = ("angles: degrees counter-clockwise on screen (0=right, 90=up); image y axis points "
                            "down; vx/vy in px/frame in image coordinates")
        return d


def save_ground_truth(path: str, rows: List[np.ndarray]) -> None:
    data = np.concatenate(rows, axis=0) if rows else np.empty((0, len(GT_COLUMNS)))
    np.savetxt(path, data, delimiter=",", header=",".join(GT_COLUMNS), comments="", fmt=GT_FORMATS)


def save_params(path: str, params: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, default=str)
