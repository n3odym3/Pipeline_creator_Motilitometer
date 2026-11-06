from __future__ import annotations

import numpy as np
import cv2
from typing import Any, Dict, Optional, Tuple
from loguru import logger
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for background worker
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from core.processing_base import ProcessingBase


class PolarPlotProcessor(ProcessingBase):
    """
    Multiprocessing worker for background Matplotlib polar plot rendering.
    Takes pooled vector orientation data, computes circular statistics,
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
        super().__init__(
            params=params,
            buffer_size=buffer_size,
            drop_policy=drop_policy,
            daemon=daemon,
        )

    def _process_data(self, data: Any, params: Dict[str, Any]) -> Any:
        """
        Processes vector orientations and renders the polar plot.
        data: list of dicts [{"angles": ndarray, "magnitudes": ndarray}, ...]
        params: rendering configuration dictionary
        Returns: (img_bgr, stats_dict)
        """
        if not data:
            return self._render_empty(params)

        try:
            pooled_angles = []
            pooled_mags = []

            for item in data:
                ang = item.get("angles")
                mag = item.get("magnitudes")
                if ang is not None and len(ang) > 0:
                    pooled_angles.append(ang)
                    if mag is not None and len(mag) == len(ang):
                        pooled_mags.append(mag)
                    else:
                        pooled_mags.append(np.ones_like(ang))

            if not pooled_angles:
                return self._render_empty(params)

            all_angles = np.concatenate(pooled_angles)
            all_mags = np.concatenate(pooled_mags)

            # Clean non-finites
            valid_mask = np.isfinite(all_angles) & np.isfinite(all_mags)
            all_angles = all_angles[valid_mask]
            all_mags = all_mags[valid_mask]

            total_vectors = len(all_angles)
            if total_vectors == 0:
                return self._render_empty(params)

            # Keep angles in [0, 2*pi)
            all_angles = np.mod(all_angles, 2 * np.pi)

            # Invert Y if requested (image -> screen coordinates)
            invert_y = params.get("invert_y", True)
            if invert_y:
                plot_angles = np.mod(-all_angles, 2 * np.pi)
            else:
                plot_angles = all_angles

            # Circular statistics
            weight_by_mag = params.get("weight_by_magnitude", True)
            weights = all_mags if (weight_by_mag and len(all_mags) == len(plot_angles)) else np.ones_like(plot_angles)
            sum_w = np.sum(weights)

            if sum_w > 0:
                c_bar = np.sum(weights * np.cos(plot_angles)) / sum_w
                s_bar = np.sum(weights * np.sin(plot_angles)) / sum_w
                mean_angle_rad = float(np.arctan2(s_bar, c_bar)) % (2 * np.pi)
                directionality_r = float(np.sqrt(c_bar**2 + s_bar**2))
            else:
                mean_angle_rad = 0.0
                directionality_r = 0.0

            mean_angle_deg = float(np.degrees(mean_angle_rad))
            mean_speed = float(np.mean(all_mags)) if len(all_mags) > 0 else 0.0

            stats = {
                "mean_angle_deg": mean_angle_deg,
                "mean_angle_rad": mean_angle_rad,
                "directionality_r": directionality_r,
                "mean_speed": mean_speed,
                "total_vectors": total_vectors,
                "accumulated_frames": len(data),
            }

            # Theme colors
            is_light = params.get("is_light", False)
            if is_light:
                fig_bg = "#f4f5f7"
                ax_bg = "#ffffff"
                text_color = "#1f2937"
                tick_color = "#4b5563"
                grid_color = "#d1d5db"
                bar_edge = "#374151"
                arrow_color = "#0284c7"
                arrow_edge = "#0f172a"
            else:
                fig_bg = "#1e1e1e"
                ax_bg = "#141414"
                text_color = "#ffffff"
                tick_color = "#cccccc"
                grid_color = "#444444"
                bar_edge = "#ffffff"
                arrow_color = "#00ffff"
                arrow_edge = "#ffffff"

            # Canvas setup
            canvas_width = int(params.get("canvas_width", 1000))
            canvas_height = int(params.get("canvas_height", 1000))
            dpi = int(params.get("dpi", 100))

            fig = Figure(
                figsize=(canvas_width / dpi, canvas_height / dpi),
                dpi=dpi,
                facecolor=fig_bg
            )
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111, projection='polar', facecolor=ax_bg)

            zero_location = params.get("zero_location", "E")
            clockwise = params.get("clockwise", False)
            ax.set_theta_zero_location(zero_location)
            ax.set_theta_direction(-1 if clockwise else 1)
            ax.grid(True, color=grid_color, linestyle='--', alpha=0.7)
            ax.tick_params(colors=tick_color, labelsize=8)

            # Histogram computation
            num_bins = int(params.get("num_bins", 24))
            theta_bins = np.linspace(0, 2 * np.pi, num_bins + 1)
            width = 2 * np.pi / num_bins
            bin_centers = theta_bins[:-1] + width / 2.0

            if weight_by_mag:
                counts, _ = np.histogram(plot_angles, bins=theta_bins, weights=all_mags)
            else:
                counts, _ = np.histogram(plot_angles, bins=theta_bins)

            if params.get("normalize", False) and np.sum(counts) > 0:
                counts = (counts / np.sum(counts)) * 100.0

            # Colors
            color_mode = params.get("color_mode", "Angle (HSV Flow)")
            colormap_name = params.get("colormap", "turbo")
            max_count = np.max(counts) if len(counts) > 0 and np.max(counts) > 0 else 1.0

            if color_mode == "Angle (HSV Flow)":
                if invert_y:
                    orig_image_angs = np.mod(-bin_centers, 2 * np.pi)
                else:
                    orig_image_angs = bin_centers
                hues = (orig_image_angs / (2.0 * np.pi)) % 1.0
                colors = [mcolors.hsv_to_rgb([h, 1.0, 1.0]) for h in hues]
            elif color_mode == "Angle (Colormap)":
                cmap = plt.get_cmap(colormap_name)
                colors = [cmap((bc / (2.0 * np.pi)) % 1.0) for bc in bin_centers]
            else:
                cmap = plt.get_cmap(colormap_name)
                norm_counts = counts / max_count
                colors = [cmap(nc) for nc in norm_counts]

            # Render Rose diagram bars
            ax.bar(
                bin_centers,
                counts,
                width=width,
                bottom=0.0,
                color=colors,
                edgecolor=bar_edge,
                alpha=0.88,
                linewidth=0.7
            )

            # Overlay Mean Resultant Vector Arrow
            show_mean = params.get("show_mean_vector", True)
            if show_mean and max_count > 0:
                arrow_r = max_count * max(0.18, directionality_r)
                ax.annotate(
                    '',
                    xy=(mean_angle_rad, arrow_r),
                    xytext=(0, 0),
                    arrowprops=dict(
                        facecolor=arrow_color,
                        edgecolor=arrow_edge,
                        width=2.5,
                        headwidth=9,
                        headlength=9,
                        alpha=0.95
                    )
                )
                ax.plot([mean_angle_rad], [arrow_r], 'o', color=arrow_color, markersize=4)

            plot_title = params.get("plot_title", "")
            title = plot_title or f"Global Vector Direction (N={total_vectors})"
            ax.set_title(title, color=text_color, fontsize=11, pad=12)

            canvas.draw()
            rgba_buffer = canvas.buffer_rgba()
            img_rgba = np.asarray(rgba_buffer)
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

            return img_bgr, stats

        except Exception as e:
            logger.error(f"PolarPlotProcessor render error: {e}")
            return self._render_empty(params)

    def _render_empty(self, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Renders an empty polar plot."""
        try:
            is_light = params.get("is_light", False)
            fig_bg = "#f4f5f7" if is_light else "#1e1e1e"
            ax_bg = "#ffffff" if is_light else "#141414"
            text_color = "#1f2937" if is_light else "#ffffff"
            tick_color = "#4b5563" if is_light else "#cccccc"
            grid_color = "#d1d5db" if is_light else "#444444"

            canvas_width = int(params.get("canvas_width", 1000))
            canvas_height = int(params.get("canvas_height", 1000))
            dpi = int(params.get("dpi", 100))

            fig = Figure(
                figsize=(canvas_width / dpi, canvas_height / dpi),
                dpi=dpi,
                facecolor=fig_bg
            )
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111, projection='polar', facecolor=ax_bg)

            zero_location = params.get("zero_location", "E")
            clockwise = params.get("clockwise", False)
            ax.set_theta_zero_location(zero_location)
            ax.set_theta_direction(-1 if clockwise else 1)
            ax.grid(True, color=grid_color, linestyle='--', alpha=0.7)
            ax.tick_params(colors=tick_color, labelsize=8)

            title = params.get("plot_title", "") or "Global Vector Direction"
            ax.set_title(title, color=text_color, fontsize=11, pad=12)

            canvas.draw()
            rgba_buffer = canvas.buffer_rgba()
            img_rgba = np.asarray(rgba_buffer)
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

            empty_stats = {
                "mean_angle_deg": 0.0,
                "mean_angle_rad": 0.0,
                "directionality_r": 0.0,
                "mean_speed": 0.0,
                "total_vectors": 0,
                "accumulated_frames": 0,
            }
            return img_bgr, empty_stats
        except Exception as e:
            logger.error(f"PolarPlotProcessor render empty error: {e}")
            cw = int(params.get("canvas_width", 1000))
            ch = int(params.get("canvas_height", 1000))
            return np.zeros((ch, cw, 3), dtype=np.uint8), {
                "mean_angle_deg": 0.0,
                "mean_angle_rad": 0.0,
                "directionality_r": 0.0,
                "mean_speed": 0.0,
                "total_vectors": 0,
                "accumulated_frames": 0,
            }
