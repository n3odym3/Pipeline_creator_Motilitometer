from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from modules.nodal_data_processor.data_processor_base import DataProcessorBase
from core.input_output_types import IOTypes

class LinearRegression(DataProcessorBase):
    """
    Perform a linear regression (y = ax + b) on the data.
    Outputs the original data and an auxiliary sample containing the fit and R².
    """

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        # Default outputs
        self.outputs: Dict[str, Any] = {
            "Out": IOTypes.DATALIST,
            "Fit": IOTypes.DATALIST
        }
        self.auxiliary_outputs: List[str] = ["Regression_Fit"]
        self.replace: bool = False
        self.equation: bool = False
        self.start_index: int = 0
        self.end_index: int = -1

    def process(
        self, 
        x_list: List[float], 
        y_list: List[float], 
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate linear regression."""
        if not self.validate_data(x_list, y_list):
            return {"Out": (x_list, y_list, labels)}

        try:
            # Clean data (remove NaNs and infinities)
            x_arr = np.array(x_list)
            y_arr = np.array(y_list)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            
            x_clean = x_arr[mask]
            y_clean = y_arr[mask]
 
            N = len(x_clean)
            if N < 2:
                return {"Out": (x_list, y_list, labels)}

            # Determine fit indices (allowing negative wrap-around)
            start = self.start_index if self.start_index >= 0 else N + self.start_index
            end = self.end_index if self.end_index >= 0 else N + self.end_index

            # Clamp boundaries
            start = max(0, min(start, N - 1))
            end = max(0, min(end, N - 1))

            # Validate range
            if start > end or (end - start + 1) < 2:
                from loguru import logger
                logger.warning(f"Linear Regression: Invalid fit range [{start}, {end}]. Falling back to full range [0, {N-1}].")
                start = 0
                end = N - 1

            # Extract subset for fitting
            x_fit_sub = x_clean[start : end + 1]
            y_fit_sub = y_clean[start : end + 1]

            # Linear regression: y = ax + b on subset
            slope, intercept = np.polyfit(x_fit_sub, y_fit_sub, 1)
            
            # Predict over full range
            y_fit = (slope * x_arr + intercept).tolist()
            
            # Calculate R-squared specifically on the fitted subset
            y_fit_sub_pred = slope * x_fit_sub + intercept
            y_mean_sub = np.mean(y_fit_sub)
            ss_res = np.sum((y_fit_sub - y_fit_sub_pred)**2)
            ss_tot = np.sum((y_fit_sub - y_mean_sub)**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            # Format Equation string
            sign = "+" if intercept >= 0 else "-"
            eq_str = f"y = {slope:.4f}x {sign} {abs(intercept):.4f} (R²={r_squared:.4f})"
            fit_name = eq_str if self.equation else "_Fit"

            # Return original data on main output, and fit result in metadata for auto-plotting
            if self.replace:
                return {
                    "Out": (x_list, y_fit, labels),
                    "Regression_Fit": {
                        "name": fit_name,
                        "x": x_list,
                        "y": y_fit,
                        "action": "unselect"
                    },
                    "R2": float(r_squared),
                    "Slope": float(slope),
                    "Intercept": float(intercept)
                }
            else:
                return {
                    "Out": (x_list, y_list, labels),
                    "Fit": (x_list, y_fit, labels),
                    "Regression_Fit": {
                        "name": fit_name,
                        "x": x_list,
                        "y": y_fit,
                        "action": "select"
                    },
                    "R2": float(r_squared),
                    "Slope": float(slope),
                    "Intercept": float(intercept)
                }

        except Exception as e:
            from loguru import logger
            logger.error(f"Linear Regression Error: {e}")
            return {"Out": (x_list, y_list, labels)}

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls."""
        import dearpygui.dearpygui as dpg
        dpg.add_text("Linear Fit: y = ax + b", parent=parent)

        replace_check = dpg.add_checkbox(
            label="Replace Input Series",
            default_value=self.replace,
            callback=lambda s, v: self.update_setting("replace", v),
            parent=parent
        )
        self.register_param("replace", replace_check)
        with dpg.tooltip(replace_check):
            dpg.add_text("If checked, outputs only the fit, replacing the input series in the plot.", wrap=250)

        equation_check = dpg.add_checkbox(
            label="Use Equation as Name",
            default_value=self.equation,
            callback=lambda s, v: self.update_setting("equation", v),
            parent=parent
        )
        self.register_param("equation", equation_check)
        with dpg.tooltip(equation_check):
            dpg.add_text("If checked, uses the equation as the fit series name. Otherwise, appends '_Fit'.", wrap=250)

        start_input = dpg.add_input_int(
            label="Start Index",
            default_value=self.start_index,
            callback=lambda s, v: self.update_setting("start_index", v),
            parent=parent,
            width=100
        )
        self.register_param("start_index", start_input)
        with dpg.tooltip(start_input):
            dpg.add_text("Starting index for fitting (0-based). Negative values count from the end.", wrap=250)

        end_input = dpg.add_input_int(
            label="End Index",
            default_value=self.end_index,
            callback=lambda s, v: self.update_setting("end_index", v),
            parent=parent,
            width=100
        )
        self.register_param("end_index", end_input)
        with dpg.tooltip(end_input):
            dpg.add_text("Ending index for fitting (inclusive). Negative values count from the end. Default is -1 (end of series).", wrap=250)

