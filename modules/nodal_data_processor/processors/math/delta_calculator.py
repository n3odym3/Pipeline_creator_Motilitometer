from typing import List, Tuple, Optional, Union, Dict, Any
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class DeltaCalculatorProcessor(DataProcessorBase):
    """Subtract a reference value (single index or range average) from all Y values."""
    
    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        self.mode: str = "Single Index"
        self.index_start: int = 0
        self.index_end: int = 10
        self.axis: str = "Y"
        self.start_widget: Union[int, str] = -1
        self.end_widget: Union[int, str] = -1
        self.tooltip_text: Union[int, str] = -1
    
    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Subtract reference value from signal."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels
        
        target_list = y_list if self.axis == "Y" else x_list
        n = len(target_list)
        
        # Clamp indices
        start = max(0, min(self.index_start, n - 1))
        end = max(0, min(self.index_end, n - 1))
        
        if start > end:
            start, end = end, start
            
        subtract_val = 0.0
        
        if self.mode == "Single Index":
            subtract_val = target_list[start]
        else:  # Range Average
            subset = target_list[start:end + 1]
            if subset:
                subtract_val = sum(subset) / len(subset)
            else:
                subtract_val = 0.0
                
        processed_list = [val - subtract_val for val in target_list]
        
        if self.axis == "Y":
            return x_list, processed_list, labels
        else:
            return processed_list, y_list, labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls."""
        axis_widget = dpg.add_combo(
            items=["X", "Y"],
            label="Axis",
            default_value=self.axis,
            callback=lambda s, v: setattr(self, "axis", v),
            parent=parent,
            width=120
        )
        self.register_param("axis", axis_widget)
        with dpg.tooltip(axis_widget):
            dpg.add_text("Axis to apply subtraction to (usually Y).", wrap=250)

        mode_widget = dpg.add_combo(
            items=["Single Index", "Range Average"],
            label="Mode",
            default_value=self.mode,
            callback=self._on_mode_change,
            parent=parent,
            width=120
        )
        self.register_param("mode", mode_widget)
        with dpg.tooltip(mode_widget):
            self.tooltip_text = dpg.add_text("", wrap=250)

        self.start_widget = dpg.add_input_int(
            label="Index Start" if self.mode == "Range Average" else "Index",
            default_value=self.index_start,
            callback=lambda s, v: setattr(self, "index_start", v),
            parent=parent,
            width=120
        )
        self.register_param("index_start", self.start_widget)
        with dpg.tooltip(self.start_widget):
            dpg.add_text("Index of reference point or starting index of range.", wrap=250)

        self.end_widget = dpg.add_input_int(
            label="Index End",
            default_value=self.index_end,
            callback=lambda s, v: setattr(self, "index_end", v),
            parent=parent,
            width=120,
            show=(self.mode == "Range Average")
        )
        self.register_param("index_end", self.end_widget)
        with dpg.tooltip(self.end_widget):
            dpg.add_text("Ending index of range (inclusive).", wrap=250)

        self._update_ui_visibility()


    def _on_mode_change(self, sender: Union[int, str], value: str, *args: Any, **kwargs: Any) -> None:
        self.mode = value
        self._update_ui_visibility()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Override to update UI visibility after setting parameters."""
        super().set_parameters(params)
        self._update_ui_visibility()

    def _update_ui_visibility(self) -> None:
        """Update visibility and labels of widgets based on current mode."""
        if hasattr(self, 'start_widget') and dpg.does_item_exist(self.start_widget):
            dpg.configure_item(self.start_widget, label="Index Start" if self.mode == "Range Average" else "Index")
        if hasattr(self, 'end_widget') and dpg.does_item_exist(self.end_widget):
            dpg.configure_item(self.end_widget, show=(self.mode == "Range Average"))
        if hasattr(self, 'tooltip_text') and dpg.does_item_exist(self.tooltip_text):
            if self.mode == "Single Index":
                dpg.set_value(
                    self.tooltip_text, 
                    "Single Index Mode:\nSubtracts the value at a specific reference index (defined by Index) from all other data points."
                )
            elif self.mode == "Range Average":
                dpg.set_value(
                    self.tooltip_text, 
                    "Range Average Mode:\nSubtracts the average value of a range of indices (from Index Start to Index End, inclusive) from all other data points."
                )

