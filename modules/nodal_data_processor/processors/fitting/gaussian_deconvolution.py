from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from loguru import logger
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase
from core.input_output_types import IOTypes


class GaussianDeconvolution(DataProcessorBase):
    """
    Deconvolve multi-peak signals into a sum of Gaussian components.
    Evaluates total fit, coverage percentage, R², and emits individual Gaussian curves.
    """

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)

        # Output pin ports on the node
        self.outputs: Dict[str, Any] = {
            "Out": IOTypes.DATALIST,
            "Total Fit": IOTypes.DATALIST,
        }
        # Auxiliary outputs for plot superposition (up to 10 Gaussians)
        self.auxiliary_outputs: List[str] = ["Total_Fit"] + [f"G{i}" for i in range(1, 11)]

        # Configuration parameters
        self.mode: str = "Auto (Best Fit)"  # "Auto (Best Fit)", "Fixed Count", "Manual Centers"
        self.num_gaussians: int = 3
        self.max_gaussians: int = 5
        self.manual_centers_str: str = "450, 650"
        self.lock_centers: bool = False
        self.center_tolerance: float = 25.0
        self.fit_baseline: bool = True
        self.emit_total_fit: bool = True
        self.emit_components: bool = True
        self.replace: bool = False

        # Internal fit metrics
        self.last_r2: float = 0.0
        self.last_coverage: float = 0.0
        self.last_components: List[Dict[str, Any]] = []

        # UI widget tags
        self.mode_widget: Optional[Union[int, str]] = None
        self.num_g_widget: Optional[Union[int, str]] = None
        self.max_g_widget: Optional[Union[int, str]] = None
        self.centers_widget: Optional[Union[int, str]] = None
        self.lock_centers_widget: Optional[Union[int, str]] = None
        self.tol_widget: Optional[Union[int, str]] = None
        self.baseline_widget: Optional[Union[int, str]] = None
        self.emit_total_widget: Optional[Union[int, str]] = None
        self.emit_comp_widget: Optional[Union[int, str]] = None
        self.replace_widget: Optional[Union[int, str]] = None
        self.result_text_widget: Optional[Union[int, str]] = None
        self.tooltip_text: Optional[Union[int, str]] = None

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fit multiple Gaussians to input data and compute coverage and components.
        """
        default_out = {
            "Out": (x_list.copy(), y_list.copy(), labels),
            "Total Fit": (x_list.copy(), y_list.copy(), labels),
        }

        if not self.validate_data(x_list, y_list):
            return default_out

        n = len(x_list)
        if n < 4:
            logger.warning("GaussianDeconvolution: not enough points to fit Gaussians (needs >= 4).")
            return default_out

        try:
            x_arr = np.asarray(x_list, dtype=float)
            y_arr = np.asarray(y_list, dtype=float)

            # Drop non-finite points
            valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if np.sum(valid_mask) < 4:
                return default_out

            x_clean = x_arr[valid_mask]
            y_clean = y_arr[valid_mask]

            # Fit Multi-Gaussian model
            components, total_fit_clean, r2, cov_pct, baseline = self._fit_multi_gaussian(x_clean, y_clean)

            self.last_r2 = r2
            self.last_coverage = cov_pct
            self.last_components = components

            # Project fit across the original x_list coordinates
            total_fit_full = self._evaluate_model(x_arr, components, baseline)

            # Update UI result display
            self._update_results_text(r2, cov_pct, components)

            # Build result dictionary
            out_y = total_fit_full.tolist() if self.replace else y_list.copy()
            res: Dict[str, Any] = {
                "Out": (x_list.copy(), out_y, labels),
                "Total Fit": (x_list.copy(), total_fit_full.tolist(), labels),
                "R2": float(r2),
                "Coverage_Pct": float(cov_pct),
                "Num_Gaussians": len(components),
                "Baseline": float(baseline),
            }

            # Emit auxiliary Total Fit for plotting
            if self.emit_total_fit:
                res["Total_Fit"] = {
                    "name": f"_Total_Fit (R²={r2:.3f}, {cov_pct:.1f}%)",
                    "x": x_list.copy(),
                    "y": total_fit_full.tolist(),
                    "action": "select",
                }

            # Emit auxiliary individual components for plotting
            if self.emit_components:
                for idx, c in enumerate(components):
                    comp_key = f"G{idx + 1}"
                    comp_curve = self._evaluate_single_gaussian(x_arr, c["amplitude"], c["mu"], c["sigma"], baseline)
                    res[comp_key] = {
                        "name": f"_G{idx + 1} ({c['mu']:.1f}, {c['pct']:.1f}%)",
                        "x": x_list.copy(),
                        "y": comp_curve.tolist(),
                        "action": "select",
                    }

            return res

        except Exception as e:
            logger.error(f"Error in GaussianDeconvolution: {e}")
            return default_out

    def _fit_multi_gaussian(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, float, float, float]:
        """Fit multiple Gaussians to clean data."""
        from scipy.optimize import curve_fit

        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        x_range = max(1e-5, x_max - x_min)
        y_min = float(np.min(y_arr))
        y_max = float(np.max(y_arr))
        y_range = max(1e-5, y_max - y_min)
        max_amp = max(0.1 * y_max, 2.0 * y_range)
        base_init = y_min if self.fit_baseline else 0.0

        def model_func(x: np.ndarray, *params: float) -> np.ndarray:
            y_fit = np.zeros_like(x, dtype=float)
            num_g = (len(params) - 1) // 3 if self.fit_baseline else len(params) // 3
            for i in range(num_g):
                amp = params[i * 3]
                mu = params[i * 3 + 1]
                sig = max(1e-5, params[i * 3 + 2])
                y_fit += amp * np.exp(-((x - mu) ** 2) / (2.0 * sig ** 2))
            if self.fit_baseline:
                y_fit += params[-1]
            return y_fit

        # Case 1: Manual Centers
        if self.mode == "Manual Centers":
            centers = self._parse_manual_centers(x_min, x_max)
            k = len(centers)
            p0 = []
            lb = []
            ub = []
            tol = max(0.1, float(self.center_tolerance))

            for mu_c in centers:
                idx_c = int(np.argmin(np.abs(x_arr - mu_c)))
                amp_c = max(0.01, float(y_arr[idx_c]) - base_init)
                sig_c = max(x_range / (4.0 * k), 5.0)
                p0.extend([amp_c, mu_c, sig_c])

                if self.lock_centers:
                    lb.extend([0.0, mu_c - 1e-3, 0.5])
                    ub.extend([max_amp, mu_c + 1e-3, x_range])
                else:
                    lb.extend([0.0, max(x_min, mu_c - tol), 0.5])
                    ub.extend([max_amp, min(x_max, mu_c + tol), x_range])

            if self.fit_baseline:
                p0.append(base_init)
                lb.append(min(0.0, y_min))
                ub.append(y_max)

            try:
                popt, _ = curve_fit(model_func, x_arr, y_arr, p0=p0, bounds=(lb, ub), maxfev=3500)
            except Exception as e:
                logger.debug(f"Manual centers curve_fit fallback: {e}")
                popt = np.array(p0)

        # Case 2: Auto Best Fit or Fixed Count
        else:
            target_n = self.num_gaussians if self.mode == "Fixed Count" else self.max_gaussians
            target_n = max(1, min(10, int(target_n)))

            # Seed first Gaussian
            peak_mu = self._detect_first_peak(x_arr, y_arr)
            first_amp = max(0.01, y_max - base_init)
            first_sig = max(x_range / 6.0, 5.0)

            current_params = [first_amp, peak_mu, first_sig]
            if self.fit_baseline:
                current_params.append(base_init)

            best_popt = current_params
            best_r2 = -1e9

            for k in range(1, target_n + 1):
                if k == 1:
                    p0 = current_params
                    lb = [0.0, max(x_min, peak_mu - x_range / 3.0), 0.5]
                    ub = [max_amp, min(x_max, peak_mu + x_range / 3.0), x_range]
                    if self.fit_baseline:
                        lb.append(min(0.0, y_min))
                        ub.append(y_max)
                else:
                    # Calculate residual of (k-1) model
                    y_prev = model_func(x_arr, *current_params)
                    res = y_arr - y_prev
                    max_res_idx = int(np.argmax(res))
                    seed_mu = float(x_arr[max_res_idx])
                    seed_amp = float(max(0.01, res[max_res_idx]))
                    seed_sig = max(x_range / (3.0 * k), 5.0)

                    gauss_p = current_params[:-1] if self.fit_baseline else current_params
                    base_p = [current_params[-1]] if self.fit_baseline else []
                    p0 = gauss_p + [seed_amp, seed_mu, seed_sig] + base_p

                    lb = []
                    ub = []
                    num_g = (len(p0) - 1) // 3 if self.fit_baseline else len(p0) // 3
                    for g_i in range(num_g):
                        g_mu = p0[g_i * 3 + 1]
                        lb.extend([0.0, max(x_min, g_mu - x_range / 2.0), 0.5])
                        ub.extend([max_amp, min(x_max, g_mu + x_range / 2.0), x_range])
                    if self.fit_baseline:
                        lb.append(min(0.0, y_min))
                        ub.append(y_max)

                try:
                    popt, _ = curve_fit(model_func, x_arr, y_arr, p0=p0, bounds=(lb, ub), maxfev=3500)
                    current_params = popt.tolist()
                except Exception as e:
                    logger.debug(f"Deconvolution step k={k} curve_fit notice: {e}")
                    current_params = p0
                    popt = np.array(p0)

                # Compute R² for this k
                y_fit_k = model_func(x_arr, *current_params)
                ss_res = np.sum((y_arr - y_fit_k) ** 2)
                ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
                r2_k = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

                if self.mode == "Auto (Best Fit)":
                    # Check if improvement is marginal (< 0.005 gain)
                    if k > 1 and (r2_k - best_r2 < 0.005 or r2_k > 0.995):
                        # Keep current params if it improved, then stop
                        if r2_k > best_r2:
                            best_popt = current_params
                            best_r2 = r2_k
                        break
                    best_r2 = r2_k
                    best_popt = current_params
                else:
                    best_popt = current_params

            popt = np.array(best_popt)

        # Extract individual components and overall metrics
        num_gauss = (len(popt) - 1) // 3 if self.fit_baseline else len(popt) // 3
        baseline = float(popt[-1]) if self.fit_baseline else 0.0

        all_components = []
        for idx in range(num_gauss):
            amp = float(popt[idx * 3])
            mu = float(popt[idx * 3 + 1])
            sig = float(abs(popt[idx * 3 + 2]))
            fwhm = 2.35482 * sig
            area = amp * sig * np.sqrt(2.0 * np.pi)

            all_components.append({
                "amplitude": amp,
                "mu": mu,
                "sigma": sig,
                "fwhm": fwhm,
                "area": area,
            })

        # Calculate Total Fit curve and R²
        total_fit = model_func(x_arr, *popt)
        ss_res = np.sum((y_arr - total_fit) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        # Calculate true physical intersection coverage (%)
        y_net = np.maximum(0.0, y_arr - baseline)
        y_fit_net = np.maximum(0.0, total_fit - baseline)
        sig_area = float(np.trapezoid(y_net, x_arr)) if len(x_arr) > 1 else float(np.sum(y_net))
        cov_area = float(np.trapezoid(np.minimum(y_net, y_fit_net), x_arr)) if len(x_arr) > 1 else 0.0
        cov_pct = min(100.0, max(0.0, (cov_area / sig_area) * 100.0)) if sig_area > 0 else 100.0

        # Calculate individual covered area and relative percentage
        for c in all_components:
            comp_y = c["amplitude"] * np.exp(-((x_arr - c["mu"]) ** 2) / (2.0 * max(1e-5, c["sigma"]) ** 2))
            comp_cov = float(np.trapezoid(np.minimum(y_net, comp_y), x_arr)) if len(x_arr) > 1 else float(np.sum(comp_y))
            c["covered_area"] = comp_cov

        sum_cov = sum(c["covered_area"] for c in all_components)
        for c in all_components:
            c["pct"] = (c["covered_area"] / sum_cov * 100.0) if sum_cov > 0 else (100.0 / len(all_components))

        # Sort components descending by area contribution
        components_sorted = sorted(all_components, key=lambda x: x["covered_area"], reverse=True)

        return components_sorted, total_fit, float(r2), float(cov_pct), float(baseline)

    def _evaluate_model(self, x_arr: np.ndarray, components: List[Dict[str, Any]], baseline: float) -> np.ndarray:
        """Evaluate sum of Gaussians + baseline across x_arr."""
        y_fit = np.full_like(x_arr, baseline, dtype=float)
        for c in components:
            amp = c["amplitude"]
            mu = c["mu"]
            sig = max(1e-5, c["sigma"])
            y_fit += amp * np.exp(-((x_arr - mu) ** 2) / (2.0 * sig ** 2))
        return y_fit

    def _evaluate_single_gaussian(
        self,
        x_arr: np.ndarray,
        amp: float,
        mu: float,
        sig: float,
        baseline: float
    ) -> np.ndarray:
        """Evaluate a single Gaussian curve."""
        s = max(1e-5, sig)
        return amp * np.exp(-((x_arr - mu) ** 2) / (2.0 * s ** 2)) + baseline

    def _detect_first_peak(self, x_arr: np.ndarray, y_arr: np.ndarray) -> float:
        """Detect central peak position."""
        try:
            from scipy.signal import find_peaks
            y_range = np.ptp(y_arr)
            prominence = max(0.01, y_range * 0.05) if y_range > 0 else None
            peaks_idx, _ = find_peaks(y_arr, prominence=prominence)
            if len(peaks_idx) > 0:
                best_idx = peaks_idx[int(np.argmax(y_arr[peaks_idx]))]
                return float(x_arr[best_idx])
        except Exception:
            pass
        return float(x_arr[int(np.argmax(y_arr))])

    def _parse_manual_centers(self, x_min: float, x_max: float) -> List[float]:
        """Parse manual center string like '435, 480, 685'."""
        parts = self.manual_centers_str.replace(";", ",").replace(" ", ",").split(",")
        centers = []
        for p in parts:
            p_strip = p.strip()
            if not p_strip:
                continue
            try:
                val = float(p_strip)
                centers.append(min(x_max, max(x_min, val)))
            except ValueError:
                pass
        if not centers:
            centers = [(x_min + x_max) / 2.0]
        return sorted(centers)[:10]

    def _update_results_text(self, r2: float, cov: float, components: List[Dict[str, Any]]) -> None:
        """Update live text display on node widget."""
        if not hasattr(self, 'result_text_widget') or not self.result_text_widget:
            return
        if not dpg.does_item_exist(self.result_text_widget):
            return

        lines = [f"R² = {r2:.4f}  |  Coverage = {cov:.1f}%", f"Fitted {len(components)} Gaussian(s):"]
        for i, c in enumerate(components):
            lines.append(f"G{i+1}: μ={c['mu']:.1f}, FWHM={c['fwhm']:.1f} ({c['pct']:.1f}%)")
        dpg.set_value(self.result_text_widget, "\n".join(lines))

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls with dynamic visibility."""
        # Mode selector
        self.mode_widget = dpg.add_combo(
            items=["Auto (Best Fit)", "Fixed Count", "Manual Centers"],
            label="Mode",
            default_value=self.mode,
            callback=self._on_mode_change,
            parent=parent,
            width=150,
        )
        self.register_param("mode", self.mode_widget)
        with dpg.tooltip(self.mode_widget):
            self.tooltip_text = dpg.add_text("", wrap=260)

        # Max Gaussians (for Auto)
        self.max_g_widget = dpg.add_input_int(
            label="Max Gaussians",
            default_value=self.max_gaussians,
            callback=lambda s, v: self.update_setting("max_gaussians", max(1, min(10, v))),
            parent=parent,
            width=150,
            step=1,
        )
        self.register_param("max_gaussians", self.max_g_widget)
        with dpg.tooltip(self.max_g_widget):
            dpg.add_text("Maximum number of Gaussians to search for before stopping.", wrap=260)

        # Fixed count
        self.num_g_widget = dpg.add_input_int(
            label="Gaussian Count",
            default_value=self.num_gaussians,
            callback=lambda s, v: self.update_setting("num_gaussians", max(1, min(10, v))),
            parent=parent,
            width=150,
            step=1,
        )
        self.register_param("num_gaussians", self.num_g_widget)
        with dpg.tooltip(self.num_g_widget):
            dpg.add_text("Exact number of Gaussians to fit.", wrap=260)

        # Manual centers string
        self.centers_widget = dpg.add_input_text(
            label="Centers (X)",
            default_value=self.manual_centers_str,
            callback=lambda s, v: self.update_setting("manual_centers_str", v),
            parent=parent,
            width=150,
        )
        self.register_param("manual_centers_str", self.centers_widget)
        with dpg.tooltip(self.centers_widget):
            dpg.add_text("Comma-separated list of target X center positions (e.g. '435, 480, 685').", wrap=260)

        # Lock centers checkbox
        self.lock_centers_widget = dpg.add_checkbox(
            label="Lock Centers (Fix μ)",
            default_value=self.lock_centers,
            callback=self._on_lock_centers_change,
            parent=parent,
        )
        self.register_param("lock_centers", self.lock_centers_widget)
        with dpg.tooltip(self.lock_centers_widget):
            dpg.add_text("If checked, centers (μ) are strictly locked to manual positions. Only amplitudes and widths are fitted.", wrap=260)

        # Center tolerance
        self.tol_widget = dpg.add_input_float(
            label="Center Tol (±X)",
            default_value=self.center_tolerance,
            callback=lambda s, v: self.update_setting("center_tolerance", max(0.1, float(v))),
            parent=parent,
            width=150,
            step=5.0,
            format="%.1f",
        )
        self.register_param("center_tolerance", self.tol_widget)
        with dpg.tooltip(self.tol_widget):
            dpg.add_text("Allowed variation window around each manual center (μ ± Tol).", wrap=260)

        # Baseline checkbox
        self.baseline_widget = dpg.add_checkbox(
            label="Fit Baseline",
            default_value=self.fit_baseline,
            callback=lambda s, v: self.update_setting("fit_baseline", v),
            parent=parent,
        )
        self.register_param("fit_baseline", self.baseline_widget)
        with dpg.tooltip(self.baseline_widget):
            dpg.add_text("Include a constant background baseline offset in the fit.", wrap=260)

        # Emit Total Fit checkbox
        self.emit_total_widget = dpg.add_checkbox(
            label="Emit Total Fit",
            default_value=self.emit_total_fit,
            callback=lambda s, v: self.update_setting("emit_total_fit", v),
            parent=parent,
        )
        self.register_param("emit_total_fit", self.emit_total_widget)
        with dpg.tooltip(self.emit_total_widget):
            dpg.add_text("Emit the overall sum of Gaussians as an auxiliary curve for automatic plot superposition.", wrap=260)

        # Emit Components checkbox
        self.emit_comp_widget = dpg.add_checkbox(
            label="Emit Components",
            default_value=self.emit_components,
            callback=lambda s, v: self.update_setting("emit_components", v),
            parent=parent,
        )
        self.register_param("emit_components", self.emit_comp_widget)
        with dpg.tooltip(self.emit_comp_widget):
            dpg.add_text("Emit each individual Gaussian component (G1, G2...) as separate curves for plot superposition.", wrap=260)

        # Replace Output checkbox
        self.replace_widget = dpg.add_checkbox(
            label="Replace 'Out'",
            default_value=self.replace,
            callback=lambda s, v: self.update_setting("replace", v),
            parent=parent,
        )
        self.register_param("replace", self.replace_widget)
        with dpg.tooltip(self.replace_widget):
            dpg.add_text("If checked, replaces primary output 'Out' with the Total Fit curve instead of passing original data.", wrap=260)

        # Live text display for fit metrics
        dpg.add_separator(parent=parent)
        self.result_text_widget = dpg.add_text(
            "Ready for fitting...",
            parent=parent,
            wrap=250,
            color=(120, 220, 120),
        )

        self._update_ui_visibility()

    def _on_mode_change(self, sender: Union[int, str], value: str, *args: Any, **kwargs: Any) -> None:
        self.update_setting("mode", value)
        self._update_ui_visibility()

    def _on_lock_centers_change(self, sender: Union[int, str], value: bool, *args: Any, **kwargs: Any) -> None:
        self.update_setting("lock_centers", value)
        self._update_ui_visibility()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        self._update_ui_visibility()

    def _update_ui_visibility(self) -> None:
        """Show/hide widgets depending on selected mode."""
        is_auto = (self.mode == "Auto (Best Fit)")
        is_fixed = (self.mode == "Fixed Count")
        is_manual = (self.mode == "Manual Centers")

        if self.max_g_widget and dpg.does_item_exist(self.max_g_widget):
            dpg.configure_item(self.max_g_widget, show=is_auto)

        if self.num_g_widget and dpg.does_item_exist(self.num_g_widget):
            dpg.configure_item(self.num_g_widget, show=is_fixed)

        if self.centers_widget and dpg.does_item_exist(self.centers_widget):
            dpg.configure_item(self.centers_widget, show=is_manual)

        if self.lock_centers_widget and dpg.does_item_exist(self.lock_centers_widget):
            dpg.configure_item(self.lock_centers_widget, show=is_manual)

        if self.tol_widget and dpg.does_item_exist(self.tol_widget):
            dpg.configure_item(self.tol_widget, show=(is_manual and not self.lock_centers))

        if self.tooltip_text and dpg.does_item_exist(self.tooltip_text):
            if is_auto:
                dpg.set_value(
                    self.tooltip_text,
                    "Auto Mode:\nIteratively searches for optimal Gaussians up to Max Gaussians and stops when improvement levels off."
                )
            elif is_fixed:
                dpg.set_value(
                    self.tooltip_text,
                    "Fixed Count Mode:\nFits exactly the specified number of Gaussians using progressive residual seeding."
                )
            else:
                dpg.set_value(
                    self.tooltip_text,
                    "Manual Centers Mode:\nInitializes and fits Gaussians at user-specified target X positions."
                )
