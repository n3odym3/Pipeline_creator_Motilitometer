import re
from typing import List, Tuple, Optional, Union, Dict, Any
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class LabelFilterProcessor(DataProcessorBase):
    """Filter data points by label pattern (e.g. keep only F0, Fm, or custom pattern)."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        self.pattern: str = ""
        self.mode: str = "contains"  # contains, starts_with, regex
        self.tooltip_text: Union[int, str] = -1

    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Filter data points whose label matches the configured pattern."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        if not labels or not self.pattern:
            return x_list, y_list, labels

        out_x = []
        out_y = []
        out_labels = []

        for i, label in enumerate(labels):
            if i >= len(x_list) or i >= len(y_list):
                break

            name = label.lower()
            pat = self.pattern.lower()

            match = False
            if self.mode == "contains":
                match = pat in name
            elif self.mode == "starts_with":
                match = name.startswith(pat)
            elif self.mode == "regex":
                try:
                    match = bool(re.search(self.pattern, label, re.IGNORECASE))
                except re.error:
                    match = False

            if match:
                out_x.append(x_list[i])
                out_y.append(y_list[i])
                out_labels.append(label)

        # If nothing matched, return original data
        if not out_x:
            return x_list, y_list, labels

        return out_x, out_y, out_labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls for filter configuration."""
        def update_tooltip(app_data: str) -> None:
            if not hasattr(self, 'tooltip_text') or not dpg.does_item_exist(self.tooltip_text):
                return
            if app_data == "contains":
                dpg.set_value(
                    self.tooltip_text, 
                    "contains Mode:\nMatches if the label contains the pattern substring."
                )
            elif app_data == "starts_with":
                dpg.set_value(
                    self.tooltip_text, 
                    "starts_with Mode:\nMatches if the label starts with the pattern prefix."
                )
            elif app_data == "regex":
                dpg.set_value(
                    self.tooltip_text, 
                    "regex Mode:\nMatches if the label matches the regular expression pattern."
                )

        def combo_callback(sender: Union[int, str], app_data: str, user_data: Any, *args: Any, **kwargs: Any) -> None:
            setattr(self, "mode", app_data)
            update_tooltip(app_data)

        mode_widget = dpg.add_combo(
            items=["contains", "starts_with", "regex"],
            default_value=self.mode,
            label="Mode",
            callback=combo_callback,
            parent=parent,
            width=120
        )
        self.register_param("mode", mode_widget)
        with dpg.tooltip(mode_widget):
            self.tooltip_text = dpg.add_text("", wrap=250)
            
        update_tooltip(self.mode)

        pattern_widget = dpg.add_input_text(
            label="Pattern",
            default_value=self.pattern,
            callback=lambda s, v: setattr(self, "pattern", v),
            parent=parent,
            width=120
        )
        self.register_param("pattern", pattern_widget)
        with dpg.tooltip(pattern_widget):
            dpg.add_text("The string or regex pattern to search for (case-insensitive).", wrap=250)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        if hasattr(self, 'tooltip_text') and dpg.does_item_exist(self.tooltip_text):
            if self.mode == "contains":
                dpg.set_value(
                    self.tooltip_text, 
                    "contains Mode:\nMatches if the label contains the pattern substring."
                )
            elif self.mode == "starts_with":
                dpg.set_value(
                    self.tooltip_text, 
                    "starts_with Mode:\nMatches if the label starts with the pattern prefix."
                )
            elif self.mode == "regex":
                dpg.set_value(
                    self.tooltip_text, 
                    "regex Mode:\nMatches if the label matches the regular expression pattern."
                )


