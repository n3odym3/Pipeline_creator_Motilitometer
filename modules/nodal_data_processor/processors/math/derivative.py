from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from loguru import logger
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class Derivative(DataProcessorBase):
    """
    Calculate the first or second numerical derivative (dy/dx or d²y/dx²) of the data.
    """

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)

        self.order: str = "First (dy/dx)"  # "First (dy/dx)" or "Second (d²y/dx²)"
        self.dx_source: str = "From X axis"  # "From X axis", "Per sample (dx=1.0)", "Custom dx"
        self.custom_dx: float = 1.0

        # UI widget tags
        self.order_widget: Optional[Union[int, str]] = None
        self.dx_widget: Optional[Union[int, str]] = None
        self.custom_dx_widget: Optional[Union[int, str]] = None
        self.tooltip_text: Optional[Union[int, str]] = None

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """
        Compute numerical derivative of Y with respect to X or chosen dx.
        Preserves data length using central differences (np.gradient).
        """
        if not self.validate_data(x_list, y_list):
            return x_list.copy(), y_list.copy(), labels

        n = len(y_list)
        if n < 2:
            return x_list.copy(), y_list.copy(), labels

        try:
            x_arr = np.asarray(x_list, dtype=float)
            y_arr = np.asarray(y_list, dtype=float)

            # Determine dx / spacing
            use_x_coords = False
            step = 1.0

            if self.dx_source == "From X axis":
                # Check for duplicate consecutive values or zero variance
                diffs = np.diff(x_arr)
                if np.any(diffs == 0.0) or np.all(diffs == 0.0):
                    logger.warning("Derivative: consecutive identical X values detected, falling back to dx=1.0.")
                    step = 1.0
                else:
                    use_x_coords = True
            elif self.dx_source == "Custom dx":
                step = self.custom_dx if self.custom_dx != 0.0 else 1.0
            else:  # "Per sample (dx=1.0)"
                step = 1.0

            # Compute first derivative
            if use_x_coords:
                dy = np.gradient(y_arr, x_arr)
            else:
                dy = np.gradient(y_arr, step)

            dy = np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)

            # Compute second derivative if selected
            if "Second" in self.order:
                if n < 3:
                    logger.warning("Derivative: dataset too small for second derivative (needs >= 3 points).")
                    return x_list.copy(), dy.tolist(), labels

                if use_x_coords:
                    d2y = np.gradient(dy, x_arr)
                else:
                    d2y = np.gradient(dy, step)

                d2y = np.nan_to_num(d2y, nan=0.0, posinf=0.0, neginf=0.0)
                out_y = d2y.tolist()
            else:
                out_y = dy.tolist()

            return x_list.copy(), out_y, labels

        except Exception as e:
            logger.error(f"Error computing derivative: {e}")
            return x_list.copy(), y_list.copy(), labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls for derivative order and dx step source."""
        # Order selection: First vs Second
        self.order_widget = dpg.add_combo(
            items=["First (dy/dx)", "Second (d²y/dx²)"],
            label="Order",
            default_value=self.order,
            callback=self._on_order_change,
            parent=parent,
            width=140,
        )
        self.register_param("order", self.order_widget)
        with dpg.tooltip(self.order_widget):
            self.tooltip_text = dpg.add_text("", wrap=260)

        # Step (dx) Source
        self.dx_widget = dpg.add_combo(
            items=["From X axis", "Per sample (dx=1.0)", "Custom dx"],
            label="dx Source",
            default_value=self.dx_source,
            callback=self._on_dx_source_change,
            parent=parent,
            width=140,
        )
        self.register_param("dx_source", self.dx_widget)
        with dpg.tooltip(self.dx_widget):
            dpg.add_text(
                "Source of the dx increment:\n"
                "- From X axis: uses real X coordinate intervals (dy/dx).\n"
                "- Per sample: treats sample interval as 1.0 (dy/di).\n"
                "- Custom dx: fixed user-defined step.",
                wrap=260,
            )

        # Custom dx input
        self.custom_dx_widget = dpg.add_input_float(
            label="Custom dx",
            default_value=self.custom_dx,
            callback=lambda s, v: self.update_setting("custom_dx", float(v)),
            parent=parent,
            width=140,
            step=0.1,
            format="%.4f",
        )
        self.register_param("custom_dx", self.custom_dx_widget)
        with dpg.tooltip(self.custom_dx_widget):
            dpg.add_text("Fixed interval dx between consecutive points.", wrap=260)

        self._update_ui_visibility()

    def _on_order_change(self, sender: Union[int, str], value: str, *args: Any, **kwargs: Any) -> None:
        self.update_setting("order", value)
        self._update_ui_visibility()

    def _on_dx_source_change(self, sender: Union[int, str], value: str, *args: Any, **kwargs: Any) -> None:
        self.update_setting("dx_source", value)
        self._update_ui_visibility()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        self._update_ui_visibility()

    def _update_ui_visibility(self) -> None:
        """Update visibility of custom dx input and tooltip text."""
        show_custom_dx = (self.dx_source == "Custom dx")
        if self.custom_dx_widget and dpg.does_item_exist(self.custom_dx_widget):
            dpg.configure_item(self.custom_dx_widget, show=show_custom_dx)

        if self.tooltip_text and dpg.does_item_exist(self.tooltip_text):
            if "Second" in self.order:
                dpg.set_value(
                    self.tooltip_text,
                    "Second Derivative (d²y/dx²):\n"
                    "Calculates curvature and acceleration of Y. Zero-crossings locate inflection points."
                )
            else:
                dpg.set_value(
                    self.tooltip_text,
                    "First Derivative (dy/dx):\n"
                    "Calculates instantaneous rate of change / slope of Y. Zero-crossings locate local peaks and valleys."
                )
