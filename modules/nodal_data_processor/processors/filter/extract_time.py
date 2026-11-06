from pathlib import Path
from typing import List, Tuple, Optional, Union
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class ExtractTimeProcessor(DataProcessorBase):
    """Replace X values with timestamps extracted from labels (select by index)."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        self.separator: str = "_"
        self.index: int = -1  # Default to last value

    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Replace X values with numeric timestamps parsed from labels."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        if not labels or len(labels) != len(x_list):
            return x_list, y_list, labels

        new_x = []
        for i, label in enumerate(labels):
            # Remove file extension
            name = Path(label).stem if '.' in label else label
            parts = name.split(self.separator)
            
            # Filter for numeric parts
            numeric_parts = [p for p in parts if p.isdigit()]

            timestamp = None
            if numeric_parts:
                try:
                    # Use index to select which numeric part to use
                    val = numeric_parts[self.index]
                    timestamp = float(val)
                except (IndexError, ValueError):
                    pass

            if timestamp is not None:
                new_x.append(timestamp)
            else:
                new_x.append(x_list[i])  # Keep original if no timestamp found

        return new_x, y_list, labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls."""
        sep_widget = dpg.add_input_text(
            label="Separator",
            default_value=self.separator,
            callback=lambda s, v: setattr(self, "separator", v),
            parent=parent,
            width=80
        )
        self.register_param("separator", sep_widget)
        with dpg.tooltip(sep_widget):
            dpg.add_text("Character separating parts of the label (e.g. '_' or '-').", wrap=250)

        idx_widget = dpg.add_input_int(
            label="Index",
            default_value=self.index,
            callback=lambda s, v: setattr(self, "index", v),
            parent=parent,
            width=80,
            step=-1
        )
        self.register_param("index", idx_widget)
        with dpg.tooltip(idx_widget):
            dpg.add_text("Index of the numeric part to use as timestamp (e.g. -1 for the last part).", wrap=250)



