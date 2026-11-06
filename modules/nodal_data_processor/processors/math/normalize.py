from typing import List, Tuple, Optional, Union
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class Normalize(DataProcessorBase):
    """Normalize Y values to the [0, 1] range (min-max normalization)."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Scale Y values so that min → 0 and max → 1."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        y_min = min(y_list)
        y_max = max(y_list)
        y_range = y_max - y_min

        if y_range == 0:
            # All values are identical — return all zeros
            normalized_y = [0.0] * len(y_list)
        else:
            normalized_y = [(y - y_min) / y_range for y in y_list]

        return x_list, normalized_y, labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """No parameters needed for min-max normalization."""
        dpg.add_text("Min → 0, Max → 1", parent=parent, color=(180, 180, 180))

