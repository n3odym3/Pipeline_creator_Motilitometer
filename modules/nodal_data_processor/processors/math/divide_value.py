from typing import List, Tuple, Optional, Union, Dict, Any
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class DivideValue(DataProcessorBase):
    """Divide X or Y values by a constant value."""
    
    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        self.axis: str = "X"
        self.divisor: float = 1.0
        self.tooltip_text: Union[int, str] = -1
    
    def process(
        self, 
        x_list: List[float], 
        y_list: List[float], 
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Divide the selected axis by the divisor."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels
        
        # Avoid division by zero
        safe_divisor = self.divisor if self.divisor != 0 else 1.0
        
        if self.axis == "X":
            new_x = [x / safe_divisor for x in x_list]
            return new_x, y_list, labels
        else:
            new_y = [y / safe_divisor for y in y_list]
            return x_list, new_y, labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls for Axis and Divisor."""
        def update_tooltip(app_data: str) -> None:
            if not hasattr(self, 'tooltip_text') or not dpg.does_item_exist(self.tooltip_text):
                return
            if app_data == "X":
                dpg.set_value(
                    self.tooltip_text, 
                    "X Axis:\nDivides the X coordinates of all data points by the divisor."
                )
            elif app_data == "Y":
                dpg.set_value(
                    self.tooltip_text, 
                    "Y Axis:\nDivides the Y coordinates of all data points by the divisor."
                )

        def combo_callback(sender: Union[int, str], app_data: str, user_data: Any, *args: Any, **kwargs: Any) -> None:
            setattr(self, "axis", app_data)
            update_tooltip(app_data)

        # Axis selection
        axis_widget = dpg.add_combo(
            label="Axis",
            items=["X", "Y"],
            default_value=self.axis,
            callback=combo_callback,
            parent=parent,
            width=100
        )
        self.register_param("axis", axis_widget)
        with dpg.tooltip(axis_widget):
            self.tooltip_text = dpg.add_text("", wrap=250)
            
        update_tooltip(self.axis)
        
        # Divisor input
        divisor_widget = dpg.add_input_float(
            label="Divisor",
            default_value=self.divisor,
            callback=lambda s, v: setattr(self, "divisor", v),
            parent=parent,
            width=100,
            step=-1
        )
        self.register_param("divisor", divisor_widget)
        with dpg.tooltip(divisor_widget):
            dpg.add_text("Value to divide all points on the selected axis by.", wrap=250)
        
        if self.divisor == 0:
            dpg.add_text("Warning: Divisor is 0", color=(255, 100, 100), parent=parent)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        if hasattr(self, 'tooltip_text') and dpg.does_item_exist(self.tooltip_text):
            if self.axis == "X":
                dpg.set_value(
                    self.tooltip_text, 
                    "X Axis:\nDivides the X coordinates of all data points by the divisor."
                )
            elif self.axis == "Y":
                dpg.set_value(
                    self.tooltip_text, 
                    "Y Axis:\nDivides the Y coordinates of all data points by the divisor."
                )


