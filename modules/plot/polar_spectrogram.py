from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

try:
    from scipy.ndimage import gaussian_filter, uniform_filter1d
except ImportError:
    gaussian_filter = None
    uniform_filter1d = None

from core.processing_base import ProcessingBase

_ANGLE_TICKS: List[int] = [0, 90, 180, 270, 360]
_ANGLE_TICK_LABELS: List[str] = [
    "0° (→ Right)",
    "90° (↑ Up)",
    "180° (← Left)",
    "270° (↓ Down)",
    "360° (→ Right)",
]
_TWO_PI: float = 2.0 * np.pi


class PolarSpectrogramProcessor(ProcessingBase):
    """
    Multiprocessing worker for rendering Polar / Angular Spectrograms.
    Receives temporal sequences of vector orientation distributions,
    assembles the 2D time-angle spectrogram matrix, applies temporal averaging
    and 2D Gaussian/circular smoothing, and renders heatmaps and trajectories
    off the main GUI thread.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        *,
        buffer_size: int = 5,
        drop_policy: str = "drop_oldest",
        daemon: bool = True,
    ) -> None:
        self._fig: Optional[Figure] = None
        self._canvas: Optional[FigureCanvas] = None
        self._canvas_key: Optional[Tuple[int, int, int]] = None
        self._bins_cache_key: Optional[int] = None
        self._bin_centers: Optional[np.ndarray] = None
        self._bin_centers_deg: Optional[np.ndarray] = None
        self._hues_cache_key: Optional[Tuple[int, bool]] = None
        self._cached_hues: Optional[np.ndarray] = None
        super().__init__(
            params=params,
            buffer_size=buffer_size,
            drop_policy=drop_policy,
            daemon=daemon,
        )

    def _get_or_create_axes(
        self,
        canvas_width: int,
        canvas_height: int,
        dpi: int,
        fig_bg: str,
        ax_bg: str,
    ) -> Tuple[Figure, FigureCanvas, Any]:
        """Return persistent Figure, FigureCanvas, and Axes, recreating Figure only on dimension/DPI changes."""
        key = (canvas_width, canvas_height, dpi)
        if self._fig is None or self._canvas is None or self._canvas_key != key:
            self._fig = Figure(
                figsize=(canvas_width / dpi, canvas_height / dpi),
                dpi=dpi,
                facecolor=fig_bg,
            )
            self._canvas = FigureCanvas(self._fig)
            self._canvas_key = key
        else:
            self._fig.clear()
            self._fig.set_facecolor(fig_bg)
        ax = self._fig.add_subplot(111, facecolor=ax_bg)
        return self._fig, self._canvas, ax

    @staticmethod
    def clean_vector_inputs(
        angles: Any,
        magnitudes: Any = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sanitize raw angle and magnitude sequences into finite [0, 2π) float32 arrays."""
        if angles is None or len(angles) == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty

        angles_arr = np.asarray(angles, dtype=np.float32).ravel()
        if magnitudes is not None and len(magnitudes) == len(angles_arr):
            mags_arr = np.asarray(magnitudes, dtype=np.float32).ravel()
            valid_mask = np.isfinite(angles_arr) & np.isfinite(mags_arr)
            if bool(valid_mask.all()):
                angles_clean = angles_arr
                mags_clean = mags_arr
            else:
                angles_clean = angles_arr[valid_mask]
                mags_clean = mags_arr[valid_mask]
        else:
            valid_mask = np.isfinite(angles_arr)
            angles_clean = angles_arr if bool(valid_mask.all()) else angles_arr[valid_mask]
            mags_clean = np.ones_like(angles_clean, dtype=np.float32)

        if angles_clean.size > 0:
            min_ang = float(np.min(angles_clean))
            max_ang = float(np.max(angles_clean))
            if max_ang > _TWO_PI + 0.1 and max_ang <= 360.0:
                angles_clean = np.radians(angles_clean)
                angles_clean = np.mod(angles_clean, _TWO_PI)
            elif min_ang < 0.0 or max_ang >= _TWO_PI:
                angles_clean = np.mod(angles_clean, _TWO_PI)

        return angles_clean, mags_clean

    @staticmethod
    def compress_frame_vectors(
        angles_clean: np.ndarray,
        mags_clean: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Compress large per-frame vector arrays (>360 vectors) into at most 360
        1-degree angular bins (mean angle, vector count, magnitude sum) to avoid
        storing or trigonometrically evaluating hundreds of thousands of vectors per frame.
        """
        if angles_clean.size <= 360:
            return angles_clean, None, mags_clean

        idx = np.clip((angles_clean * (360.0 / _TWO_PI)).astype(np.intp), 0, 359)
        cnt = np.bincount(idx, minlength=360).astype(np.float32)
        nz = cnt > 0
        cnt_nz = cnt[nz]
        ang_sum = np.bincount(idx, weights=angles_clean, minlength=360)[nz]
        mag_sum = np.bincount(idx, weights=mags_clean, minlength=360).astype(np.float32)[nz]
        ang_mean = np.mod((ang_sum / cnt_nz).astype(np.float32), _TWO_PI)
        return ang_mean, cnt_nz, mag_sum

    @staticmethod
    def bin_single_frame(
        angles_clean: np.ndarray,
        mags_clean: np.ndarray,
        num_bins: int,
        weight_by_magnitude: bool,
        invert_y: bool,
        count_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float, float, int]:
        """Compute 1D angular histogram counts and circular statistics for a single frame."""
        if angles_clean.size == 0:
            return np.zeros(num_bins, dtype=np.float32), 0.0, 0.0, 0.0, 0

        plot_angles = np.mod(-angles_clean, _TWO_PI) if invert_y else angles_clean
        cos_a = np.cos(plot_angles)
        sin_a = np.sin(plot_angles)

        if count_weights is None:
            n_vectors = int(angles_clean.size)
            sum_mag = float(np.sum(mags_clean))
            mean_speed = sum_mag / n_vectors
            if weight_by_magnitude:
                if sum_mag > 0.0:
                    c_bar = float(np.dot(mags_clean, cos_a)) / sum_mag
                    s_bar = float(np.dot(mags_clean, sin_a)) / sum_mag
                    mean_rad = float(np.arctan2(s_bar, c_bar)) % _TWO_PI
                    r_val = float(np.hypot(c_bar, s_bar))
                else:
                    mean_rad = 0.0
                    r_val = 0.0
                weights: Optional[np.ndarray] = mags_clean
            else:
                c_bar = float(np.mean(cos_a))
                s_bar = float(np.mean(sin_a))
                mean_rad = float(np.arctan2(s_bar, c_bar)) % _TWO_PI
                r_val = float(np.hypot(c_bar, s_bar))
                weights = None
        else:
            total_cnt = float(np.sum(count_weights))
            n_vectors = int(round(total_cnt))
            if n_vectors <= 0:
                return np.zeros(num_bins, dtype=np.float32), 0.0, 0.0, 0.0, 0
            sum_mag = float(np.sum(mags_clean))
            mean_speed = sum_mag / total_cnt
            if weight_by_magnitude:
                if sum_mag > 0.0:
                    c_bar = float(np.dot(mags_clean, cos_a)) / sum_mag
                    s_bar = float(np.dot(mags_clean, sin_a)) / sum_mag
                    mean_rad = float(np.arctan2(s_bar, c_bar)) % _TWO_PI
                    r_val = float(np.hypot(c_bar, s_bar))
                else:
                    mean_rad = 0.0
                    r_val = 0.0
                weights = mags_clean
            else:
                c_bar = float(np.dot(count_weights, cos_a)) / total_cnt
                s_bar = float(np.dot(count_weights, sin_a)) / total_cnt
                mean_rad = float(np.arctan2(s_bar, c_bar)) % _TWO_PI
                r_val = float(np.hypot(c_bar, s_bar))
                weights = count_weights

        mean_angle_deg = float(np.degrees(mean_rad))
        bin_idx = np.clip((plot_angles * (num_bins / _TWO_PI)).astype(np.intp), 0, num_bins - 1)
        counts = np.bincount(bin_idx, weights=weights, minlength=num_bins).astype(np.float32)
        return counts, mean_angle_deg, r_val, mean_speed, n_vectors

    def _process_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], Tuple[Dict[str, Any], ...]],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process accumulated frame vector data and render the polar spectrogram.

        Args:
            data: Pre-binned payload dict or list of raw per-frame dicts.
            params: Dictionary of rendering and analysis settings.

        Returns:
            Tuple of (img_bgr, stats_dict).
        """
        if not data or not isinstance(data, (list, tuple, dict)):
            return self._render_empty(params)

        try:
            num_bins = max(8, min(360, int(params.get("num_bins", 72))))
            weight_by_mag = bool(params.get("weight_by_magnitude", True))
            invert_y = bool(params.get("invert_y", True))
            orientation = str(params.get("orientation", "Horizontal (Time ->)"))
            color_mode = str(params.get("color_mode", "Colormap"))
            colormap_name = str(params.get("colormap", "turbo"))
            norm_mode = str(params.get("normalization", "Frame Peak (Max = 100%)"))
            color_scale_mode = str(params.get("color_scale_mode", "Auto"))
            color_vmin = float(params.get("color_vmin", 0.0))
            color_vmax = float(params.get("color_vmax", 50.0))
            show_mean_curve = bool(params.get("show_mean_curve", True))
            plot_title = str(params.get("plot_title", ""))
            canvas_width = max(200, int(params.get("canvas_width", 1100)))
            canvas_height = max(200, int(params.get("canvas_height", 650)))
            dpi = max(50, int(params.get("dpi", 100)))
            is_light = bool(params.get("is_light", False))
            fps = max(0.1, float(params.get("fps", 30.0)))
            delta_t = float(params.get("delta_t", 1.0 / fps))

            temporal_window = max(1, int(params.get("temporal_window", 1)))
            smooth_time = max(0.0, float(params.get("smooth_time", 1.0)))
            smooth_angle = max(0.0, float(params.get("smooth_angle", 1.0)))
            interpolation = str(params.get("interpolation", "bicubic"))
            mean_curve_smooth = max(1, int(params.get("mean_curve_smooth", 3)))

            if self._bins_cache_key != num_bins or self._bin_centers is None or self._bin_centers_deg is None:
                theta_bins = np.linspace(0.0, _TWO_PI, num_bins + 1, dtype=np.float32)
                bin_width = _TWO_PI / num_bins
                self._bin_centers = theta_bins[:-1] + bin_width * 0.5
                self._bin_centers_deg = np.degrees(self._bin_centers)
                self._bins_cache_key = num_bins
            bin_centers = self._bin_centers
            bin_centers_deg = self._bin_centers_deg

            if isinstance(data, dict) and "matrix" in data:
                raw_matrix = np.asarray(data["matrix"], dtype=np.float32)
                num_frames = int(raw_matrix.shape[0])
                if num_frames == 0:
                    return self._render_empty(params)
                mean_angles_deg = np.asarray(data.get("mean_angles_deg", np.zeros(num_frames)), dtype=np.float32)
                directionality_r = np.asarray(data.get("directionality_r", np.zeros(num_frames)), dtype=np.float32)
                mean_speeds = np.asarray(data.get("mean_speeds", np.zeros(num_frames)), dtype=np.float32)
                total_vectors_arr = np.asarray(data.get("total_vectors_arr", np.zeros(num_frames, dtype=np.int32)), dtype=np.int32)
                timestamps = np.asarray(data.get("timestamps", np.arange(num_frames, dtype=np.float32) * delta_t), dtype=np.float32)
                frame_indices = np.asarray(data.get("frame_indices", np.arange(num_frames, dtype=np.int32)), dtype=np.int32)
            else:
                num_frames = len(data)
                raw_matrix = np.zeros((num_frames, num_bins), dtype=np.float32)
                mean_angles_deg = np.zeros(num_frames, dtype=np.float32)
                directionality_r = np.zeros(num_frames, dtype=np.float32)
                mean_speeds = np.zeros(num_frames, dtype=np.float32)
                total_vectors_arr = np.zeros(num_frames, dtype=np.int32)
                timestamps = np.zeros(num_frames, dtype=np.float32)
                frame_indices = np.zeros(num_frames, dtype=np.int32)

                for i, frame in enumerate(data):
                    frame_indices[i] = int(frame.get("frame_idx", i))
                    timestamps[i] = float(frame.get("timestamp", i * delta_t))
                    angles_clean, mags_clean = self.clean_vector_inputs(
                        frame.get("angles"), frame.get("magnitudes")
                    )
                    if angles_clean.size == 0:
                        continue
                    row, m_ang, r_val, spd, n_vec = self.bin_single_frame(
                        angles_clean, mags_clean, num_bins, weight_by_mag, invert_y
                    )
                    raw_matrix[i, :] = row
                    mean_angles_deg[i] = m_ang
                    directionality_r[i] = r_val
                    mean_speeds[i] = spd
                    total_vectors_arr[i] = n_vec

            matrix = raw_matrix

            # 1. Temporal rolling window average
            if temporal_window > 1 and num_frames >= 2:
                w = min(temporal_window, num_frames)
                if uniform_filter1d is not None:
                    matrix = uniform_filter1d(matrix, size=w, axis=0, mode="nearest")
                else:
                    kernel = np.ones(w, dtype=np.float32) / float(w)
                    matrix = np.apply_along_axis(
                        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=matrix
                    )

            # 2. 2D Gaussian smoothing with periodic circular wrapping on the angular axis
            if (smooth_time > 0.0 or smooth_angle > 0.0) and matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
                if gaussian_filter is not None:
                    matrix = gaussian_filter(
                        matrix,
                        sigma=(smooth_time, smooth_angle),
                        mode=["nearest", "wrap"],
                    )

            # 3. Intensity normalization
            if norm_mode == "Frame Peak (Max = 100%)":
                row_maxes = np.max(matrix, axis=1, keepdims=True)
                row_maxes = np.where(row_maxes == 0.0, 1.0, row_maxes)
                matrix = (matrix / row_maxes) * 100.0
                cbar_label = "% Frame Peak"
            elif norm_mode == "Frame Density (%)":
                row_sums = np.sum(matrix, axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
                matrix = (matrix / row_sums) * 100.0
                cbar_label = "Density (%)"
            elif norm_mode == "Global Max":
                g_max = float(np.max(matrix))
                matrix = (matrix / g_max) * 100.0 if g_max > 0.0 else matrix.copy()
                cbar_label = "% Global Max"
            elif norm_mode == "Log Scale":
                matrix = np.log10(1.0 + np.maximum(0.0, matrix))
                cbar_label = "Log10(1 + Counts)"
            else:
                if matrix is raw_matrix:
                    matrix = raw_matrix.copy()
                cbar_label = "Weighted Magnitude" if weight_by_mag else "Vector Count"

            if is_light:
                fig_bg, ax_bg = "#f4f5f7", "#ffffff"
                text_color, tick_color, grid_color = "#1f2937", "#4b5563", "#d1d5db"
                mean_line_color = "#d97706"
            else:
                fig_bg, ax_bg = "#1e1e1e", "#141414"
                text_color, tick_color, grid_color = "#ffffff", "#cccccc", "#444444"
                mean_line_color = "#00f0ff"

            fig, canvas, ax = self._get_or_create_axes(canvas_width, canvas_height, dpi, fig_bg, ax_bg)

            t_start = float(timestamps[0])
            t_end = float(timestamps[-1])
            if t_end <= t_start:
                t_end = t_start + max(0.001, delta_t)

            is_horizontal = "Horizontal" in orientation

            # Circular moving average for the mean resultant angle trajectory
            if show_mean_curve and num_frames > 0:
                valid_mask = total_vectors_arr > 0
                if mean_curve_smooth > 1 and num_frames >= 3:
                    w_curve = min(mean_curve_smooth, num_frames)
                    if w_curve % 2 == 0:
                        w_curve += 1
                    ang_rad = np.radians(mean_angles_deg)
                    cos_vals = np.where(valid_mask, np.cos(ang_rad), 0.0)
                    sin_vals = np.where(valid_mask, np.sin(ang_rad), 0.0)
                    w_weights = valid_mask.astype(np.float32)
                    kernel_c = np.ones(w_curve, dtype=np.float32)
                    sum_c = np.convolve(cos_vals, kernel_c, mode="same")
                    sum_s = np.convolve(sin_vals, kernel_c, mode="same")
                    sum_w = np.convolve(w_weights, kernel_c, mode="same")
                    sum_w = np.where(sum_w == 0.0, 1.0, sum_w)
                    smoothed_rad = np.arctan2(sum_s / sum_w, sum_c / sum_w) % _TWO_PI
                    curve_angles = np.degrees(smoothed_rad)
                else:
                    curve_angles = mean_angles_deg.copy()

                curve_angles[~valid_mask] = np.nan
                jumps = np.abs(np.diff(curve_angles)) > 180.0
                if np.any(jumps):
                    curve_angles[1:][jumps] = np.nan
            else:
                curve_angles = np.empty(0, dtype=np.float32)

            disp_matrix = matrix.T if is_horizontal else matrix
            extent = [t_start, t_end, 0.0, 360.0] if is_horizontal else [0.0, 360.0, t_start, t_end]

            if color_mode == "Angle (HSV Flow)":
                if color_scale_mode == "Manual":
                    c_min = color_vmin
                    c_max = color_vmax if color_vmax > color_vmin else (color_vmin + 1.0)
                    disp_norm = np.clip((disp_matrix - c_min) / (c_max - c_min), 0.0, 1.0)
                else:
                    max_val = float(np.max(disp_matrix))
                    disp_norm = np.clip(disp_matrix / (max_val if max_val > 0.0 else 1.0), 0.0, 1.0)

                hues_key = (num_bins, invert_y)
                if self._hues_cache_key != hues_key or self._cached_hues is None:
                    orig_angs = np.mod(-bin_centers, _TWO_PI) if invert_y else bin_centers
                    self._cached_hues = (orig_angs / _TWO_PI) % 1.0
                    self._hues_cache_key = hues_key
                hues = self._cached_hues

                hsv_img = np.empty((*disp_matrix.shape, 3), dtype=np.float32)
                hsv_img[..., 0] = hues[:, np.newaxis] if is_horizontal else hues[np.newaxis, :]
                hsv_img[..., 1] = 1.0
                hsv_img[..., 2] = disp_norm

                rgb_img = mcolors.hsv_to_rgb(hsv_img)
                ax.imshow(rgb_img, origin="lower", extent=extent, aspect="auto", interpolation=interpolation)
            else:
                imshow_kwargs: Dict[str, Any] = {
                    "origin": "lower",
                    "extent": extent,
                    "aspect": "auto",
                    "cmap": colormap_name,
                    "interpolation": interpolation,
                }
                if color_scale_mode == "Manual":
                    imshow_kwargs["vmin"] = color_vmin
                    imshow_kwargs["vmax"] = color_vmax if color_vmax > color_vmin else (color_vmin + 1.0)

                im = ax.imshow(disp_matrix, **imshow_kwargs)
                cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
                cbar.set_label(cbar_label, color=text_color, fontsize=8)
                cbar.ax.tick_params(colors=tick_color, labelsize=8)

            if is_horizontal:
                ax.set_ylim(360.0, 0.0) if invert_y else ax.set_ylim(0.0, 360.0)
                ax.set_yticks(_ANGLE_TICKS)
                ax.set_yticklabels(_ANGLE_TICK_LABELS, color=tick_color, fontsize=8)
                ax.set_ylabel("Angle Sector (°)", color=text_color, fontsize=9, fontweight="medium")
                ax.set_xlim(t_start, t_end)
                ax.set_xlabel("Time (s)", color=text_color, fontsize=9, fontweight="medium")
                if show_mean_curve and curve_angles.size > 0:
                    ax.plot(timestamps, curve_angles, color=mean_line_color, linewidth=1.9, alpha=0.95, label="Mean Direction")
            else:
                ax.set_xlim(0.0, 360.0)
                ax.set_xticks(_ANGLE_TICKS)
                ax.set_xticklabels(_ANGLE_TICK_LABELS, color=tick_color, fontsize=8)
                ax.set_xlabel("Angle Sector (°)", color=text_color, fontsize=9, fontweight="medium")
                ax.set_ylim(t_start, t_end)
                ax.set_ylabel("Time (s)", color=text_color, fontsize=9, fontweight="medium")
                if show_mean_curve and curve_angles.size > 0:
                    ax.plot(curve_angles, timestamps, color=mean_line_color, linewidth=1.9, alpha=0.95, label="Mean Direction")

            ax.tick_params(colors=tick_color, labelsize=8)
            ax.grid(True, color=grid_color, linestyle=":", alpha=0.5)

            t_span = t_end - t_start
            tot_vecs = int(np.sum(total_vectors_arr))
            last_mean = float(mean_angles_deg[-1]) if num_frames > 0 else 0.0
            last_r = float(directionality_r[-1]) if num_frames > 0 else 0.0
            last_updown = "UP ↑" if (0.0 <= last_mean <= 180.0) else "DOWN ↓"

            smooth_info: List[str] = []
            if temporal_window > 1:
                smooth_info.append(f"Avg:{temporal_window}f")
            if smooth_time > 0.0 or smooth_angle > 0.0:
                smooth_info.append(f"σt:{smooth_time:.1f}, σθ:{smooth_angle:.1f}")
            smooth_str = f" [{', '.join(smooth_info)}]" if smooth_info else ""

            default_title = (
                f"Polar Spectrogram{smooth_str} | {num_frames} frames ({t_span:.2f}s) | "
                f"Current Mean: {last_mean:.1f}° ({last_updown}, R={last_r:.2f}) | {tot_vecs} vectors"
            )
            ax.set_title(plot_title or default_title, color=text_color, fontsize=10, pad=10)

            fig.tight_layout()
            canvas.draw()
            img_rgba = np.asarray(canvas.buffer_rgba())
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

            is_prebinned = isinstance(data, dict) and "matrix" in data
            stats: Dict[str, Any] = {
                "total_frames": num_frames,
                "current_frame": int(frame_indices[-1]) if num_frames > 0 else 0,
                "elapsed_time": t_span,
                "latest_mean_angle_deg": last_mean,
                "latest_directionality_r": last_r,
                "latest_mean_speed": float(mean_speeds[-1]) if num_frames > 0 else 0.0,
                "latest_total_vectors": int(total_vectors_arr[-1]) if num_frames > 0 else 0,
                "matrix": matrix,
                "bin_centers_deg": bin_centers_deg,
                "temporal_window": temporal_window,
                "smooth_time": smooth_time,
                "smooth_angle": smooth_angle,
                "interpolation": interpolation,
            }
            if not is_prebinned:
                stats["raw_matrix"] = raw_matrix
                stats["timestamps"] = timestamps
                stats["frame_indices"] = frame_indices
                stats["mean_angles_deg"] = mean_angles_deg
                stats["directionality_r"] = directionality_r
                stats["mean_speeds"] = mean_speeds
                stats["total_vectors_arr"] = total_vectors_arr

            return img_bgr, stats

        except Exception as e:
            logger.error(f"PolarSpectrogramProcessor render error: {e}")
            return self._render_empty(params)

    def _render_empty(self, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Render an empty initialized spectrogram grid with axes and instructions."""
        try:
            is_light = bool(params.get("is_light", False))
            fig_bg = "#f4f5f7" if is_light else "#1e1e1e"
            ax_bg = "#ffffff" if is_light else "#141414"
            text_color = "#1f2937" if is_light else "#ffffff"
            tick_color = "#4b5563" if is_light else "#cccccc"
            grid_color = "#d1d5db" if is_light else "#333333"

            num_bins = max(8, min(360, int(params.get("num_bins", 72))))
            canvas_width = max(200, int(params.get("canvas_width", 1100)))
            canvas_height = max(200, int(params.get("canvas_height", 650)))
            dpi = max(50, int(params.get("dpi", 100)))
            orientation = str(params.get("orientation", "Horizontal (Time ->)"))
            invert_y = bool(params.get("invert_y", True))
            is_horizontal = "Horizontal" in orientation

            fig, canvas, ax = self._get_or_create_axes(canvas_width, canvas_height, dpi, fig_bg, ax_bg)

            if is_horizontal:
                ax.set_xlim(0.0, 10.0)
                ax.set_ylim(360.0, 0.0) if invert_y else ax.set_ylim(0.0, 360.0)
                ax.set_yticks(_ANGLE_TICKS)
                ax.set_yticklabels(_ANGLE_TICK_LABELS, color=tick_color, fontsize=8)
                ax.set_xlabel("Time (s)", color=text_color, fontsize=9)
                ax.set_ylabel("Angle Sector (°)", color=text_color, fontsize=9)
            else:
                ax.set_xlim(0.0, 360.0)
                ax.set_ylim(0.0, 10.0)
                ax.set_xticks(_ANGLE_TICKS)
                ax.set_xticklabels(_ANGLE_TICK_LABELS, color=tick_color, fontsize=8)
                ax.set_xlabel("Angle Sector (°)", color=text_color, fontsize=9)
                ax.set_ylabel("Time (s)", color=text_color, fontsize=9)

            ax.tick_params(colors=tick_color, labelsize=8)
            ax.grid(True, color=grid_color, linestyle=":", alpha=0.5)
            ax.set_title("Polar Spectrogram [Awaiting Vector Stream]", color=text_color, fontsize=10, pad=10)

            ax.text(
                0.5,
                0.5,
                "Waiting for vector orientation data...\n(Connect Optical Flow, Counter Tracker, or Vector Source)",
                color=text_color,
                fontsize=11,
                ha="center",
                va="center",
                transform=ax.transAxes,
                alpha=0.6,
                bbox=dict(boxstyle="round,pad=0.8", facecolor=ax_bg, edgecolor=grid_color, alpha=0.85),
            )

            fig.tight_layout()
            canvas.draw()
            img_rgba = np.asarray(canvas.buffer_rgba())
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

            empty_stats: Dict[str, Any] = {
                "total_frames": 0,
                "current_frame": 0,
                "elapsed_time": 0.0,
                "latest_mean_angle_deg": 0.0,
                "latest_directionality_r": 0.0,
                "latest_mean_speed": 0.0,
                "latest_total_vectors": 0,
                "matrix": np.zeros((0, num_bins), dtype=np.float32),
                "bin_centers_deg": np.zeros(num_bins, dtype=np.float32),
                "temporal_window": 1,
                "smooth_time": 0.0,
                "smooth_angle": 0.0,
                "interpolation": "bicubic",
            }

            return img_bgr, empty_stats

        except Exception as e:
            logger.error(f"PolarSpectrogramProcessor empty render error: {e}")
            w = max(200, int(params.get("canvas_width", 1100)))
            h = max(200, int(params.get("canvas_height", 650)))
            return np.zeros((h, w, 3), dtype=np.uint8), {}
