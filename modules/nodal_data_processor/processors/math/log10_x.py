import math
from typing import List, Tuple, Optional, Union
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class Log10X(DataProcessorBase):
    """Transform X values to their base-10 logarithm."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Convert X to log10(X), dropping any X <= 0."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        new_x = []
        new_y = []
        new_labels = [] if labels is not None else None

        for i, x in enumerate(x_list):
            if x > 0:
                new_x.append(math.log10(x))
                new_y.append(y_list[i])
                if labels is not None and new_labels is not None:
                    new_labels.append(labels[i])

        return new_x, new_y, new_labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """UI for the log10 X processor."""
        dpg.add_text("X → log10(X)", parent=parent, color=(180, 180, 180))
        dpg.add_text("Drops X <= 0", parent=parent, color=(150, 150, 150))

