from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from core.processing_base import ProcessingBase

_TWO_PI: float = 2.0 * np.pi


class PolarPlotProcessor(ProcessingBase):
    """
    Multiprocessing worker for background Matplotlib polar plot rendering.
    Pools vector orientation data, computes circular statistics,
    and renders the Rose diagram / polar plot off the main GUI thread.
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
        self._ax: Optional[Any] = None
        self._canvas_key: Optional[Tuple[int, int, int]] = None
        self._bins_cache_key: Optional[int] = None
        self._bin_centers: Optional[np.ndarray] = None
        self._colors_cache_key: Optional[Tuple[int, str, str, bool]] = None
        self._cached_bar_colors: Optional[Any] = None
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
        """Return persistent Figure, FigureCanvas, and PolarAxes, recreating only on dimension/DPI changes."""
        key = (canvas_width, canvas_height, dpi)
        if self._fig is None or self._canvas is None or self._ax is None or self._canvas_key != key:
            self._fig = Figure(
                figsize=(canvas_width / dpi, canvas_height / dpi),
                dpi=dpi,
                facecolor=fig_bg,
            )
            self._canvas = FigureCanvas(self._fig)
            self._ax = self._fig.add_subplot(111, projection="polar", facecolor=ax_bg)
            self._canvas_key = key
        else:
            self._ax.clear()
            self._fig.set_facecolor(fig_bg)
            self._ax.set_facecolor(ax_bg)
        return self._fig, self._canvas, self._ax

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

    def _process_data(
        self,
        data: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process vector orientations and render the polar plot.

        Args:
            data: List of frame dicts [{"angles": ndarray, "magnitudes": ndarray}, ...].
            params: Rendering configuration dictionary.

        Returns:
            Tuple of (img_bgr, stats_dict).
        """
        if not data:
            return self._render_empty(params)

        try:
            pooled_angles: List[np.ndarray] = []
            pooled_mags: List[np.ndarray] = []

            for item in data:
                ang, mag = self.clean_vector_inputs(item.get("angles"), item.get("magnitudes"))
                if ang.size > 0:
                    pooled_angles.append(ang)
                    pooled_mags.append(mag)

            if not pooled_angles:
                return self._render_empty(params)

            if len(pooled_angles) == 1:
                all_angles = pooled_angles[0]
                all_mags = pooled_mags[0]
            else:
                all_angles = np.concatenate(pooled_angles)
                all_mags = np.concatenate(pooled_mags)

            total_vectors = int(all_angles.size)
            if total_vectors == 0:
                return self._render_empty(params)

            invert_y = bool(params.get("invert_y", True))
            plot_angles = np.mod(-all_angles, _TWO_PI) if invert_y else all_angles

            weight_by_mag = bool(params.get("weight_by_magnitude", True))
            cos_a = np.cos(plot_angles)
            sin_a = np.sin(plot_angles)
            sum_mag = float(np.sum(all_mags))
            mean_speed = sum_mag / total_vectors

            if weight_by_mag:
                if sum_mag > 0.0:
                    c_bar = float(np.dot(all_mags, cos_a)) / sum_mag
                    s_bar = float(np.dot(all_mags, sin_a)) / sum_mag
                    mean_angle_rad = float(np.arctan2(s_bar, c_bar)) % _TWO_PI
                    directionality_r = float(np.hypot(c_bar, s_bar))
                else:
                    mean_angle_rad = 0.0
                    directionality_r = 0.0
            else:
                c_bar = float(np.mean(cos_a))
                s_bar = float(np.mean(sin_a))
                mean_angle_rad = float(np.arctan2(s_bar, c_bar)) % _TWO_PI
                directionality_r = float(np.hypot(c_bar, s_bar))

            mean_angle_deg = float(np.degrees(mean_angle_rad))

            stats: Dict[str, Any] = {
                "mean_angle_deg": mean_angle_deg,
                "mean_angle_rad": mean_angle_rad,
                "directionality_r": directionality_r,
                "mean_speed": mean_speed,
                "total_vectors": total_vectors,
                "accumulated_frames": len(data),
            }

            is_light = bool(params.get("is_light", False))
            if is_light:
                fig_bg, ax_bg = "#f4f5f7", "#ffffff"
                text_color, tick_color, grid_color = "#1f2937", "#4b5563", "#d1d5db"
                bar_edge, arrow_color, arrow_edge = "#374151", "#0284c7", "#0f172a"
            else:
                fig_bg, ax_bg = "#1e1e1e", "#141414"
                text_color, tick_color, grid_color = "#ffffff", "#cccccc", "#444444"
                bar_edge, arrow_color, arrow_edge = "#ffffff", "#00ffff", "#ffffff"

            canvas_width = max(200, int(params.get("canvas_width", 1000)))
            canvas_height = max(200, int(params.get("canvas_height", 1000)))
            dpi = max(50, int(params.get("dpi", 100)))

            _, canvas, ax = self._get_or_create_axes(canvas_width, canvas_height, dpi, fig_bg, ax_bg)

            zero_location = str(params.get("zero_location", "E"))
            clockwise = bool(params.get("clockwise", False))
            ax.set_theta_zero_location(zero_location)
            ax.set_theta_direction(-1 if clockwise else 1)
            ax.grid(True, color=grid_color, linestyle="--", alpha=0.7)
            ax.tick_params(colors=tick_color, labelsize=8)

            num_bins = max(8, min(360, int(params.get("num_bins", 24))))
            width = _TWO_PI / num_bins
            if self._bins_cache_key != num_bins or self._bin_centers is None:
                theta_bins = np.linspace(0.0, _TWO_PI, num_bins + 1, dtype=np.float32)
                self._bin_centers = theta_bins[:-1] + width * 0.5
                self._bins_cache_key = num_bins
            bin_centers = self._bin_centers

            bin_idx = np.clip((plot_angles * (num_bins / _TWO_PI)).astype(np.intp), 0, num_bins - 1)
            counts = np.bincount(
                bin_idx,
                weights=all_mags if weight_by_mag else None,
                minlength=num_bins,
            ).astype(np.float32)

            total_counts = float(np.sum(counts))
            if bool(params.get("normalize", False)) and total_counts > 0.0:
                counts = (counts / total_counts) * 100.0

            color_mode = str(params.get("color_mode", "Angle (HSV Flow)"))
            colormap_name = str(params.get("colormap", "turbo"))
            max_count_raw = float(np.max(counts)) if counts.size > 0 else 0.0
            max_count = max_count_raw if max_count_raw > 0.0 else 1.0

            if color_mode in ("Angle (HSV Flow)", "Angle (Colormap)"):
                color_key = (num_bins, color_mode, colormap_name, invert_y)
                if self._colors_cache_key != color_key or self._cached_bar_colors is None:
                    if color_mode == "Angle (HSV Flow)":
                        orig_image_angs = np.mod(-bin_centers, _TWO_PI) if invert_y else bin_centers
                        hues = (orig_image_angs / _TWO_PI) % 1.0
                        hsv_arr = np.column_stack((hues, np.ones_like(hues), np.ones_like(hues)))
                        self._cached_bar_colors = mcolors.hsv_to_rgb(hsv_arr)
                    else:
                        cmap = plt.get_cmap(colormap_name)
                        self._cached_bar_colors = cmap((bin_centers / _TWO_PI) % 1.0)
                    self._colors_cache_key = color_key
                colors: Any = self._cached_bar_colors
            else:
                cmap = plt.get_cmap(colormap_name)
                colors = cmap(counts / max_count)

            ax.bar(
                bin_centers,
                counts,
                width=width,
                bottom=0.0,
                color=colors,
                edgecolor=bar_edge,
                alpha=0.88,
                linewidth=0.7,
            )

            if bool(params.get("show_mean_vector", True)) and max_count_raw > 0.0:
                arrow_r = max_count * max(0.18, directionality_r)
                ax.annotate(
                    "",
                    xy=(mean_angle_rad, arrow_r),
                    xytext=(0, 0),
                    arrowprops=dict(
                        facecolor=arrow_color,
                        edgecolor=arrow_edge,
                        width=2.5,
                        headwidth=9,
                        headlength=9,
                        alpha=0.95,
                    ),
                )
                ax.plot([mean_angle_rad], [arrow_r], "o", color=arrow_color, markersize=4)

            plot_title = str(params.get("plot_title", ""))
            title = plot_title or f"Global Vector Direction (N={total_vectors})"
            ax.set_title(title, color=text_color, fontsize=11, pad=12)

            canvas.draw()
            img_rgba = np.asarray(canvas.buffer_rgba())
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

            return img_bgr, stats

        except Exception as e:
            logger.error(f"PolarPlotProcessor render error: {e}")
            return self._render_empty(params)

    def _render_empty(self, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Render an empty polar plot."""
        empty_stats: Dict[str, Any] = {
            "mean_angle_deg": 0.0,
            "mean_angle_rad": 0.0,
            "directionality_r": 0.0,
            "mean_speed": 0.0,
            "total_vectors": 0,
            "accumulated_frames": 0,
        }
        try:
            is_light = bool(params.get("is_light", False))
            fig_bg = "#f4f5f7" if is_light else "#1e1e1e"
            ax_bg = "#ffffff" if is_light else "#141414"
            text_color = "#1f2937" if is_light else "#ffffff"
            tick_color = "#4b5563" if is_light else "#cccccc"
            grid_color = "#d1d5db" if is_light else "#444444"

            canvas_width = max(200, int(params.get("canvas_width", 1000)))
            canvas_height = max(200, int(params.get("canvas_height", 1000)))
            dpi = max(50, int(params.get("dpi", 100)))

            _, canvas, ax = self._get_or_create_axes(canvas_width, canvas_height, dpi, fig_bg, ax_bg)

            zero_location = str(params.get("zero_location", "E"))
            clockwise = bool(params.get("clockwise", False))
            ax.set_theta_zero_location(zero_location)
            ax.set_theta_direction(-1 if clockwise else 1)
            ax.grid(True, color=grid_color, linestyle="--", alpha=0.7)
            ax.tick_params(colors=tick_color, labelsize=8)

            title = str(params.get("plot_title", "")) or "Global Vector Direction"
            ax.set_title(title, color=text_color, fontsize=11, pad=12)

            canvas.draw()
            img_rgba = np.asarray(canvas.buffer_rgba())
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

            return img_bgr, empty_stats
        except Exception as e:
            logger.error(f"PolarPlotProcessor render empty error: {e}")
            cw = max(200, int(params.get("canvas_width", 1000)))
            ch = max(200, int(params.get("canvas_height", 1000)))
            return np.zeros((ch, cw, 3), dtype=np.uint8), empty_stats
