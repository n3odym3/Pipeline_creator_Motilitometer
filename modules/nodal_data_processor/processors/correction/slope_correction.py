from typing import List, Tuple, Optional, Union
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class SlopeCorrection(DataProcessorBase):
    """Correct a linear slope between two reference points.

    Computes the slope between the points at Index 1 and Index 2,
    then subtracts the linear trend from the whole series so that
    the point at Index 2 is brought down to the same Y value as
    the point at Index 1.
    """

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        self.index_1: int = 0
        self.index_2: int = -1  # -1 means last point
        

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Apply slope correction between the two reference indices."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        n = len(x_list)

        # Resolve negative indices (e.g. -1 → last point)
        i1 = self.index_1 if self.index_1 >= 0 else n + self.index_1
        i2 = self.index_2 if self.index_2 >= 0 else n + self.index_2

        # Clamp to valid range
        i1 = max(0, min(i1, n - 1))
        i2 = max(0, min(i2, n - 1))

        if i1 == i2:
            # Cannot compute slope between the same point
            return x_list, y_list, labels

        x1, y1 = x_list[i1], y_list[i1]
        x2, y2 = x_list[i2], y_list[i2]

        dx = x2 - x1
        if dx == 0:
            # X values are identical — fall back to index-based slope
            dy = y2 - y1
            slope_per_step = dy / (i2 - i1)
            corrected_y = [
                y - slope_per_step * (i - i1) for i, y in enumerate(y_list)
            ]
        else:
            slope = (y2 - y1) / dx
            corrected_y = [y - slope * (x - x1) for x, y in zip(x_list, y_list)]

        return x_list, corrected_y, labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls for the two reference indices."""
        w1 = dpg.add_input_int(
            label="Index 1",
            default_value=self.index_1,
            callback=lambda s, v: setattr(self, "index_1", v),
            parent=parent,
            width=80,
            step=-1
        )
        self.register_param("index_1", w1)
        with dpg.tooltip(w1):
            dpg.add_text("First reference index (e.g. 0 for first point, supports negative indexing like -1).", wrap=250)

        w2 = dpg.add_input_int(
            label="Index 2",
            default_value=self.index_2,
            callback=lambda s, v: setattr(self, "index_2", v),
            parent=parent,
            width=80,
            step=-1
        )
        self.register_param("index_2", w2)
        with dpg.tooltip(w2):
            dpg.add_text("Second reference index (brought down to Index 1 Y-value, supports negative indexing).", wrap=250)


