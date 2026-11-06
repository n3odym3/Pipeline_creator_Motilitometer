from typing import List, Tuple, Optional, Union, Dict, Any
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class OffsetProcessor(DataProcessorBase):
    """Add an offset (shift) to selected axis."""
    
    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        self.offset: float = 0.0
        self.axis: str = "X"
        self.tooltip_text: Union[int, str] = -1
    
    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Add offset to selected axis values."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels
        
        if self.axis == "X":
            offset_x = [x + self.offset for x in x_list]
            return offset_x, y_list, labels
        else:
            offset_y = [y + self.offset for y in y_list]
            return x_list, offset_y, labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls."""
        def update_tooltip(app_data: str) -> None:
            if not hasattr(self, 'tooltip_text') or not dpg.does_item_exist(self.tooltip_text):
                return
            if app_data == "X":
                dpg.set_value(
                    self.tooltip_text, 
                    "X Axis:\nShifts the X coordinates of all data points by adding the offset."
                )
            elif app_data == "Y":
                dpg.set_value(
                    self.tooltip_text, 
                    "Y Axis:\nShifts the Y coordinates of all data points by adding the offset."
                )

        def combo_callback(sender: Union[int, str], app_data: str, user_data: Any, *args: Any, **kwargs: Any) -> None:
            setattr(self, "axis", app_data)
            update_tooltip(app_data)

        axis_widget = dpg.add_combo(
            items=["X", "Y"],
            label="Axis",
            default_value=self.axis,
            callback=combo_callback,
            parent=parent,
            width=100
        )
        self.register_param("axis", axis_widget)
        with dpg.tooltip(axis_widget):
            self.tooltip_text = dpg.add_text("", wrap=250)
            
        update_tooltip(self.axis)

        offset_widget = dpg.add_input_float(
            label="Offset",
            default_value=self.offset,
            callback=lambda s, v: setattr(self, "offset", v),
            parent=parent,
            width=100,
            step=-1
        )
        self.register_param("offset", offset_widget)
        with dpg.tooltip(offset_widget):
            dpg.add_text("Constant value to add to the selected axis.", wrap=250)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        if hasattr(self, 'tooltip_text') and dpg.does_item_exist(self.tooltip_text):
            if self.axis == "X":
                dpg.set_value(
                    self.tooltip_text, 
                    "X Axis:\nShifts the X coordinates of all data points by adding the offset."
                )
            elif self.axis == "Y":
                dpg.set_value(
                    self.tooltip_text, 
                    "Y Axis:\nShifts the Y coordinates of all data points by adding the offset."
                )


