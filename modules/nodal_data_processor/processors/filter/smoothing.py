from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from loguru import logger
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class SmoothingProcessor(DataProcessorBase):
    """
    Smooth data using Rolling Mean, EMA, Savitzky-Golay, Rolling Median, or Gaussian filter.
    """

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)

        # General settings
        self.mode: str = "Rolling Mean"
        self.axis: str = "Y"  # "Y", "X", "Both"

        # Rolling Mean & Rolling Median settings
        self.window: int = 5
        self.center: bool = True

        # EMA settings
        self.ema_type: str = "Span"  # "Span" or "Alpha"
        self.span: float = 10.0
        self.alpha: float = 0.2

        # Savitzky-Golay settings
        self.savgol_window: int = 7
        self.savgol_poly: int = 2
        self.savgol_mode: str = "interp"  # "interp", "mirror", "nearest", "constant", "wrap"

        # Gaussian filter settings
        self.gaussian_sigma: float = 2.0

        # UI widget tags
        self.mode_widget: Optional[Union[int, str]] = None
        self.axis_widget: Optional[Union[int, str]] = None
        self.window_widget: Optional[Union[int, str]] = None
        self.center_widget: Optional[Union[int, str]] = None
        self.ema_type_widget: Optional[Union[int, str]] = None
        self.span_widget: Optional[Union[int, str]] = None
        self.alpha_widget: Optional[Union[int, str]] = None
        self.savgol_win_widget: Optional[Union[int, str]] = None
        self.savgol_poly_widget: Optional[Union[int, str]] = None
        self.savgol_mode_widget: Optional[Union[int, str]] = None
        self.gaussian_sigma_widget: Optional[Union[int, str]] = None
        self.tooltip_text: Optional[Union[int, str]] = None

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """
        Process the input data and apply the configured smoothing filter.
        """
        if not self.validate_data(x_list, y_list):
            return x_list.copy(), y_list.copy(), labels

        new_x = x_list.copy()
        new_y = y_list.copy()

        try:
            if self.axis in ("Y", "Both"):
                new_y = self._apply_filter(y_list)
            if self.axis in ("X", "Both"):
                new_x = self._apply_filter(x_list)
        except Exception as e:
            logger.error(f"Error in SmoothingProcessor ({self.mode}): {e}")
            return x_list.copy(), y_list.copy(), labels

        return new_x, new_y, labels

    def _apply_filter(self, data: List[float]) -> List[float]:
        """Apply the selected smoothing algorithm to a 1D float list."""
        n = len(data)
        if n <= 1:
            return list(data)

        arr = np.asarray(data, dtype=float)

        if self.mode == "Rolling Mean":
            return self._smooth_rolling_mean(arr)
        elif self.mode == "Exponential Moving Average":
            return self._smooth_ema(arr)
        elif self.mode == "Savitzky-Golay":
            return self._smooth_savgol(arr)
        elif self.mode == "Rolling Median":
            return self._smooth_rolling_median(arr)
        elif self.mode == "Gaussian Filter":
            return self._smooth_gaussian(arr)
        elif self.mode == "Cumulative Moving Average":
            return self._smooth_cma(arr)
        else:
            logger.warning(f"Unknown smoothing mode '{self.mode}', returning raw data.")
            return list(data)

    def _smooth_rolling_mean(self, arr: np.ndarray) -> List[float]:
        """Rolling mean with boundary handling preserving length."""
        w = max(1, int(self.window))
        if w >= len(arr):
            # Window encompasses all or more points: return overall mean
            return [float(np.mean(arr))] * len(arr)
        if w == 1:
            return arr.tolist()

        try:
            import pandas as pd
            s = pd.Series(arr)
            res = s.rolling(window=w, min_periods=1, center=self.center).mean().to_numpy()
            return res.tolist()
        except ImportError:
            # Fallback using numpy convolve if pandas is not available
            half = w // 2 if self.center else 0
            pad_width = (half, w - 1 - half) if self.center else (w - 1, 0)
            padded = np.pad(arr, pad_width, mode="edge")
            kernel = np.ones(w) / w
            smoothed = np.convolve(padded, kernel, mode="valid")
            return smoothed.tolist()

    def _smooth_rolling_median(self, arr: np.ndarray) -> List[float]:
        """Rolling median with boundary handling preserving length."""
        w = max(1, int(self.window))
        if w == 1:
            return arr.tolist()
        if w >= len(arr):
            return [float(np.median(arr))] * len(arr)

        try:
            import pandas as pd
            s = pd.Series(arr)
            res = s.rolling(window=w, min_periods=1, center=self.center).median().to_numpy()
            return res.tolist()
        except ImportError:
            from scipy.signal import medfilt
            # Ensure odd window for medfilt
            eff_w = w if w % 2 != 0 else w + 1
            if eff_w > len(arr):
                eff_w = len(arr) if len(arr) % 2 != 0 else len(arr) - 1
            if eff_w < 3:
                return arr.tolist()
            return medfilt(arr, kernel_size=eff_w).tolist()

    def _smooth_ema(self, arr: np.ndarray) -> List[float]:
        """Exponential moving average."""
        if self.ema_type == "Alpha":
            alpha = max(0.001, min(1.0, float(self.alpha)))
        else:
            span = max(1.0, float(self.span))
            alpha = 2.0 / (span + 1.0)

        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out.tolist()

    def _smooth_savgol(self, arr: np.ndarray) -> List[float]:
        """
        Savitzky-Golay filter with adaptive safety checks to prevent crashes on incompatible data lengths.
        """
        n = len(arr)
        if n < 3:
            return arr.tolist()

        try:
            from scipy.signal import savgol_filter
        except ImportError:
            logger.error("scipy is required for Savitzky-Golay filter.")
            return arr.tolist()

        # Target window length
        w = max(3, int(self.savgol_window))
        # SavGol requires odd window
        if w % 2 == 0:
            w += 1

        # Safety adaptation: window cannot exceed data length
        if w > n:
            w = n if n % 2 != 0 else n - 1

        if w < 3:
            return arr.tolist()

        # Safety adaptation: polynomial order must be strictly less than window length
        poly = max(1, int(self.savgol_poly))
        if poly >= w:
            poly = w - 1

        valid_modes = ["interp", "mirror", "nearest", "constant", "wrap"]
        mode = self.savgol_mode if self.savgol_mode in valid_modes else "interp"

        try:
            filtered = savgol_filter(arr, window_length=w, polyorder=poly, mode=mode)
            return filtered.tolist()
        except Exception as e:
            logger.warning(f"Savitzky-Golay fallback triggered (w={w}, poly={poly}, N={n}): {e}")
            return arr.tolist()

    def _smooth_gaussian(self, arr: np.ndarray) -> List[float]:
        """Gaussian 1D filter."""
        sigma = max(0.1, float(self.gaussian_sigma))
        try:
            from scipy.ndimage import gaussian_filter1d
            res = gaussian_filter1d(arr, sigma=sigma, mode="nearest")
            return res.tolist()
        except ImportError:
            logger.error("scipy is required for Gaussian filter.")
            return arr.tolist()

    def _smooth_cma(self, arr: np.ndarray) -> List[float]:
        """Cumulative Moving Average."""
        counts = np.arange(1, len(arr) + 1, dtype=float)
        res = np.cumsum(arr, dtype=float) / counts
        return res.tolist()

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls with dynamic visibility based on the selected mode."""
        # Mode selector
        self.mode_widget = dpg.add_combo(
            items=[
                "Rolling Mean",
                "Exponential Moving Average",
                "Savitzky-Golay",
                "Rolling Median",
                "Gaussian Filter",
                "Cumulative Moving Average",
            ],
            label="Mode",
            default_value=self.mode,
            callback=self._on_mode_change,
            parent=parent,
            width=150,
        )
        self.register_param("mode", self.mode_widget)
        with dpg.tooltip(self.mode_widget):
            self.tooltip_text = dpg.add_text("", wrap=260)

        # Axis selector
        self.axis_widget = dpg.add_combo(
            items=["Y", "X", "Both"],
            label="Axis",
            default_value=self.axis,
            callback=lambda s, v: self.update_setting("axis", v),
            parent=parent,
            width=150,
        )
        self.register_param("axis", self.axis_widget)
        with dpg.tooltip(self.axis_widget):
            dpg.add_text("Target axis to apply smoothing to (usually Y).", wrap=260)

        # Rolling Mean / Median: Window
        self.window_widget = dpg.add_input_int(
            label="Window Size",
            default_value=self.window,
            callback=lambda s, v: self.update_setting("window", max(1, v)),
            parent=parent,
            width=150,
            step=1,
        )
        self.register_param("window", self.window_widget)
        with dpg.tooltip(self.window_widget):
            dpg.add_text("Number of points included in the sliding window.", wrap=260)

        # Rolling Mean / Median: Center
        self.center_widget = dpg.add_checkbox(
            label="Center Window",
            default_value=self.center,
            callback=lambda s, v: self.update_setting("center", v),
            parent=parent,
        )
        self.register_param("center", self.center_widget)
        with dpg.tooltip(self.center_widget):
            dpg.add_text("If checked, centers the window at current point (eliminates phase delay).", wrap=260)

        # EMA: Parameter Type (Span vs Alpha)
        self.ema_type_widget = dpg.add_combo(
            items=["Span", "Alpha"],
            label="Param Type",
            default_value=self.ema_type,
            callback=self._on_ema_type_change,
            parent=parent,
            width=150,
        )
        self.register_param("ema_type", self.ema_type_widget)
        with dpg.tooltip(self.ema_type_widget):
            dpg.add_text("Specify EMA smoothness by Span (window-equivalent) or direct Alpha factor.", wrap=260)

        # EMA: Span
        self.span_widget = dpg.add_input_float(
            label="Span",
            default_value=self.span,
            callback=lambda s, v: self.update_setting("span", max(1.0, float(v))),
            parent=parent,
            width=150,
            step=1.0,
        )
        self.register_param("span", self.span_widget)
        with dpg.tooltip(self.span_widget):
            dpg.add_text("Equivalent window length for EMA (alpha = 2 / (span + 1)). Higher = smoother.", wrap=260)

        # EMA: Alpha
        self.alpha_widget = dpg.add_input_float(
            label="Alpha",
            default_value=self.alpha,
            callback=lambda s, v: self.update_setting("alpha", max(0.001, min(1.0, float(v)))),
            parent=parent,
            width=150,
            step=0.05,
            format="%.3f",
        )
        self.register_param("alpha", self.alpha_widget)
        with dpg.tooltip(self.alpha_widget):
            dpg.add_text("Weight of current point (0 < alpha <= 1). Lower = smoother.", wrap=260)

        # Savitzky-Golay: Window Length
        self.savgol_win_widget = dpg.add_input_int(
            label="Window Length",
            default_value=self.savgol_window,
            callback=lambda s, v: self.update_setting("savgol_window", max(3, v)),
            parent=parent,
            width=150,
            step=2,
        )
        self.register_param("savgol_window", self.savgol_win_widget)
        with dpg.tooltip(self.savgol_win_widget):
            dpg.add_text("Length of filter window (must be an odd integer >= 3). Automatically adapted if data is shorter.", wrap=260)

        # Savitzky-Golay: Poly Order
        self.savgol_poly_widget = dpg.add_input_int(
            label="Poly Order",
            default_value=self.savgol_poly,
            callback=lambda s, v: self.update_setting("savgol_poly", max(1, v)),
            parent=parent,
            width=150,
            step=1,
        )
        self.register_param("savgol_poly", self.savgol_poly_widget)
        with dpg.tooltip(self.savgol_poly_widget):
            dpg.add_text("Order of the fitting polynomial (must be < Window Length). Automatically adapted if too high.", wrap=260)

        # Savitzky-Golay: Mode
        self.savgol_mode_widget = dpg.add_combo(
            items=["interp", "mirror", "nearest", "constant", "wrap"],
            label="Edge Mode",
            default_value=self.savgol_mode,
            callback=lambda s, v: self.update_setting("savgol_mode", v),
            parent=parent,
            width=150,
        )
        self.register_param("savgol_mode", self.savgol_mode_widget)
        with dpg.tooltip(self.savgol_mode_widget):
            dpg.add_text("Extension mode for boundary points.", wrap=260)

        # Gaussian: Sigma
        self.gaussian_sigma_widget = dpg.add_input_float(
            label="Sigma",
            default_value=self.gaussian_sigma,
            callback=lambda s, v: self.update_setting("gaussian_sigma", max(0.1, float(v))),
            parent=parent,
            width=150,
            step=0.5,
            format="%.2f",
        )
        self.register_param("gaussian_sigma", self.gaussian_sigma_widget)
        with dpg.tooltip(self.gaussian_sigma_widget):
            dpg.add_text("Standard deviation of the Gaussian kernel. Higher = smoother.", wrap=260)

        self._update_ui_visibility()

    def _on_mode_change(self, sender: Union[int, str], value: str, *args: Any, **kwargs: Any) -> None:
        """Callback when the smoothing mode combo box changes."""
        self.update_setting("mode", value)
        self._update_ui_visibility()

    def _on_ema_type_change(self, sender: Union[int, str], value: str, *args: Any, **kwargs: Any) -> None:
        """Callback when EMA parameter type changes."""
        self.update_setting("ema_type", value)
        self._update_ui_visibility()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Override to update dynamic UI visibility after loading parameters."""
        super().set_parameters(params)
        self._update_ui_visibility()

    def _update_ui_visibility(self) -> None:
        """Update visibility of parameter widgets depending on current mode."""
        is_rolling_mean = (self.mode == "Rolling Mean")
        is_rolling_median = (self.mode == "Rolling Median")
        is_ema = (self.mode == "Exponential Moving Average")
        is_savgol = (self.mode == "Savitzky-Golay")
        is_gaussian = (self.mode == "Gaussian Filter")

        # Rolling Mean / Median controls
        show_window = is_rolling_mean or is_rolling_median
        if self.window_widget and dpg.does_item_exist(self.window_widget):
            dpg.configure_item(self.window_widget, show=show_window)
        if self.center_widget and dpg.does_item_exist(self.center_widget):
            dpg.configure_item(self.center_widget, show=show_window)

        # EMA controls
        if self.ema_type_widget and dpg.does_item_exist(self.ema_type_widget):
            dpg.configure_item(self.ema_type_widget, show=is_ema)
        if self.span_widget and dpg.does_item_exist(self.span_widget):
            dpg.configure_item(self.span_widget, show=(is_ema and self.ema_type == "Span"))
        if self.alpha_widget and dpg.does_item_exist(self.alpha_widget):
            dpg.configure_item(self.alpha_widget, show=(is_ema and self.ema_type == "Alpha"))

        # Savitzky-Golay controls
        if self.savgol_win_widget and dpg.does_item_exist(self.savgol_win_widget):
            dpg.configure_item(self.savgol_win_widget, show=is_savgol)
        if self.savgol_poly_widget and dpg.does_item_exist(self.savgol_poly_widget):
            dpg.configure_item(self.savgol_poly_widget, show=is_savgol)
        if self.savgol_mode_widget and dpg.does_item_exist(self.savgol_mode_widget):
            dpg.configure_item(self.savgol_mode_widget, show=is_savgol)

        # Gaussian controls
        if self.gaussian_sigma_widget and dpg.does_item_exist(self.gaussian_sigma_widget):
            dpg.configure_item(self.gaussian_sigma_widget, show=is_gaussian)

        # Tooltip update
        if self.tooltip_text and dpg.does_item_exist(self.tooltip_text):
            tooltips = {
                "Rolling Mean": "Rolling Mean:\nCalculates the moving average across a sliding window. Window centering avoids time shift.",
                "Exponential Moving Average": "Exponential Moving Average (EMA):\nApplies exponentially decreasing weights over time. Fast and memory-efficient.",
                "Savitzky-Golay": "Savitzky-Golay Filter:\nFits local polynomials to preserve signal peak heights and widths while smoothing high-frequency noise. Includes safety adaptation against data size mismatches.",
                "Rolling Median": "Rolling Median:\nSliding median filter. Highly robust against single-point outliers, spikes, and impulse noise without blurring sharp steps.",
                "Gaussian Filter": "Gaussian Filter 1D:\nConvolves signal with a Gaussian bell curve of standard deviation Sigma. Yields perfectly smooth curves without rectangular ringing.",
                "Cumulative Moving Average": "Cumulative Moving Average (CMA):\nAverages all values observed from the first point up to the current index.",
            }
            dpg.set_value(self.tooltip_text, tooltips.get(self.mode, ""))
