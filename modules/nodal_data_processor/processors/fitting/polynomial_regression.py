from typing import List, Tuple, Optional, Union, Dict, Any
import dearpygui.dearpygui as dpg
import numpy as np
from modules.nodal_data_processor.data_processor_base import DataProcessorBase
from core.input_output_types import IOTypes


class PolyRegression(DataProcessorBase):
    """Calculate polynomial regression and return fitted data and equation."""
    
    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        self.poly_order: int = 1
        self.replace: bool = False
        self.equation: bool = False
        self.outputs: Dict[str, Any] = {
            "Out": IOTypes.DATALIST,
            "Fit": IOTypes.DATALIST,
            "Equation": IOTypes.TEXT
        }
        self.auxiliary_outputs: List[str] = ["Regression_Fit"]
    
    def process(
        self, 
        x_list: List[float], 
        y_list: List[float], 
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform polynomial regression."""
        if not self.validate_data(x_list, y_list):
            return {
                "Out": (x_list, y_list, labels),
                "Equation": "Error: Invalid data"
            }
        
        try:
            # Drop NaN or infinite values
            x_arr = np.array(x_list)
            y_arr = np.array(y_list)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_clean = x_arr[mask]
            y_clean = y_arr[mask]
            
            if len(x_clean) <= self.poly_order:
                return {
                    "Out": (x_list, y_list, labels),
                    "Equation": "Error: Not enough points"
                }

            # Perform fit
            coefs = np.polyfit(x_clean, y_clean, self.poly_order)
            poly_func = np.poly1d(coefs)
            
            # Generate fitted Y
            y_fit = poly_func(x_arr).tolist()
            
            # Format equation string
            eq_parts = []
            for i, c in enumerate(coefs):
                p = self.poly_order - i
                if p > 1:
                    eq_parts.append(f"{c:.4f}x^{p}")
                elif p == 1:
                    eq_parts.append(f"{c:.4f}x")
                else:
                    eq_parts.append(f"{c:.4f}")
            
            equation = "y = " + " + ".join(eq_parts).replace("+ -", "- ")
            fit_name = equation if self.equation else "_Fit"
            
            if self.replace:
                return {
                    "Out": (x_list, y_fit, labels),
                    "Regression_Fit": {
                        "name": fit_name,
                        "x": x_list,
                        "y": y_fit,
                        "action": "unselect"
                    },
                    "Equation": equation
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
                    "Equation": equation
                }
            
        except Exception as e:
            return {
                "Out": (x_list, y_list, labels),
                "Equation": f"Error: {str(e)}"
            }

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls for Polynomial Order."""
        
        order_widget = dpg.add_input_int(
            label="Poly Order",
            default_value=self.poly_order,
            min_value=1,
            max_value=10,
            callback=lambda s, v: self.update_setting("poly_order", max(1, v)),
            parent=parent,
            width=100
        )
        self.register_param("poly_order", order_widget)
        with dpg.tooltip(order_widget):
            dpg.add_text("Order of the fitted polynomial (1 = linear, 2 = quadratic, 3 = cubic, etc.).", wrap=250)

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


