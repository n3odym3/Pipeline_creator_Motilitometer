import math
from typing import List, Tuple, Optional, Union
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class Log10Y(DataProcessorBase):
    """Transform Y values to their base-10 logarithm."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Convert Y to log10(Y), dropping any Y <= 0."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        new_x = []
        new_y = []
        new_labels = [] if labels is not None else None

        for i, y in enumerate(y_list):
            if y > 0:
                new_x.append(x_list[i])
                new_y.append(math.log10(y))
                if labels is not None and new_labels is not None:
                    new_labels.append(labels[i])

        return new_x, new_y, new_labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """UI for the log10 Y processor."""
        dpg.add_text("Y → log10(Y)", parent=parent, color=(180, 180, 180))
        dpg.add_text("Drops Y <= 0", parent=parent, color=(150, 150, 150))

