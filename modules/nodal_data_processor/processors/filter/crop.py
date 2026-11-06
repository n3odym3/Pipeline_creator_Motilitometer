from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from loguru import logger
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class Crop(DataProcessorBase):
    """
    Crop or slice data by Range (From... To...) or by Trimming edges (Crop Left / Right).
    Supports cropping by Index (point count) or by X coordinate values.
    """

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)

        self.mode: str = "Range (From ... To ...)"  # "Range (From ... To ...)" or "Trim Edges (Left / Right)"
        self.unit: str = "Index (Points)"          # "Index (Points)" or "X Value"

        # Range parameters
        self.start_idx: int = 0
        self.end_idx: int = -1
        self.start_x: float = 0.0
        self.end_x: float = 100.0

        # Trim parameters
        self.crop_left_idx: int = 0
        self.crop_right_idx: int = 0
        self.crop_left_x: float = 0.0
        self.crop_right_x: float = 0.0

        # UI widget tags
        self.mode_widget: Optional[Union[int, str]] = None
        self.unit_widget: Optional[Union[int, str]] = None

        self.start_idx_widget: Optional[Union[int, str]] = None
        self.end_idx_widget: Optional[Union[int, str]] = None
        self.start_x_widget: Optional[Union[int, str]] = None
        self.end_x_widget: Optional[Union[int, str]] = None

        self.crop_left_idx_widget: Optional[Union[int, str]] = None
        self.crop_right_idx_widget: Optional[Union[int, str]] = None
        self.crop_left_x_widget: Optional[Union[int, str]] = None
        self.crop_right_x_widget: Optional[Union[int, str]] = None

        self.tooltip_text: Optional[Union[int, str]] = None

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """
        Crop data based on current configuration.
        Synchronously slices x_list, y_list, and labels.
        """
        if not self.validate_data(x_list, y_list):
            return x_list.copy(), y_list.copy(), labels

        n = len(x_list)
        if n == 0:
            return x_list.copy(), y_list.copy(), labels

        indices_to_keep: List[int] = []

        try:
            if self.mode == "Range (From ... To ...)":
                if self.unit == "Index (Points)":
                    s = max(0, self.start_idx)
                    e = n - 1 if self.end_idx == -1 else min(n - 1, self.end_idx)
                    if s <= e:
                        indices_to_keep = list(range(s, e + 1))
                else:  # "X Value"
                    min_x = min(self.start_x, self.end_x)
                    max_x = max(self.start_x, self.end_x)
                    indices_to_keep = [i for i, x in enumerate(x_list) if min_x <= x <= max_x]

            else:  # "Trim Edges (Left / Right)"
                if self.unit == "Index (Points)":
                    left = max(0, int(self.crop_left_idx))
                    right = max(0, int(self.crop_right_idx))
                    s = left
                    e = max(0, n - right)
                    if s < e:
                        indices_to_keep = list(range(s, e))
                else:  # "X Value"
                    x_min = min(x_list)
                    x_max = max(x_list)
                    left_cutoff = x_min + max(0.0, float(self.crop_left_x))
                    right_cutoff = x_max - max(0.0, float(self.crop_right_x))
                    indices_to_keep = [i for i, x in enumerate(x_list) if left_cutoff <= x <= right_cutoff]

            if not indices_to_keep:
                logger.warning("Crop resulted in 0 points matching criteria; returning original data.")
                return x_list.copy(), y_list.copy(), labels

            new_x = [x_list[i] for i in indices_to_keep]
            new_y = [y_list[i] for i in indices_to_keep]
            new_labels = [labels[i] for i in indices_to_keep] if labels and len(labels) == n else labels

            return new_x, new_y, new_labels

        except Exception as e:
            logger.error(f"Error in Crop processor: {e}")
            return x_list.copy(), y_list.copy(), labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls with dynamic visibility based on mode and unit."""
        # Mode selector
        self.mode_widget = dpg.add_combo(
            items=["Range (From ... To ...)", "Trim Edges (Left / Right)"],
            label="Mode",
            default_value=self.mode,
            callback=self._on_mode_change,
            parent=parent,
            width=150,
        )
        self.register_param("mode", self.mode_widget)
        with dpg.tooltip(self.mode_widget):
            self.tooltip_text = dpg.add_text("", wrap=260)

        # Unit selector
        self.unit_widget = dpg.add_combo(
            items=["Index (Points)", "X Value"],
            label="Unit",
            default_value=self.unit,
            callback=self._on_unit_change,
            parent=parent,
            width=150,
        )
        self.register_param("unit", self.unit_widget)
        with dpg.tooltip(self.unit_widget):
            dpg.add_text(
                "Unit used for crop boundaries:\n"
                "- Index (Points): discrete point count (0, 1, 2...).\n"
                "- X Value: real values on the X axis (e.g. time, wavelength).",
                wrap=260,
            )

        # Range - Index
        self.start_idx_widget = dpg.add_input_int(
            label="Start Index",
            default_value=self.start_idx,
            callback=lambda s, v: self.update_setting("start_idx", max(0, v)),
            parent=parent,
            width=150,
            step=1,
        )
        self.register_param("start_idx", self.start_idx_widget)
        with dpg.tooltip(self.start_idx_widget):
            dpg.add_text("First index to keep (0-indexed).", wrap=260)

        self.end_idx_widget = dpg.add_input_int(
            label="End Index",
            default_value=self.end_idx,
            callback=lambda s, v: self.update_setting("end_idx", v),
            parent=parent,
            width=150,
            step=1,
        )
        self.register_param("end_idx", self.end_idx_widget)
        with dpg.tooltip(self.end_idx_widget):
            dpg.add_text("Last index to keep (inclusive). Set to -1 to crop up to the very last point.", wrap=260)

        # Range - X Value
        self.start_x_widget = dpg.add_input_float(
            label="Min X",
            default_value=self.start_x,
            callback=lambda s, v: self.update_setting("start_x", float(v)),
            parent=parent,
            width=150,
            step=1.0,
            format="%.2f",
        )
        self.register_param("start_x", self.start_x_widget)
        with dpg.tooltip(self.start_x_widget):
            dpg.add_text("Minimum X value to keep.", wrap=260)

        self.end_x_widget = dpg.add_input_float(
            label="Max X",
            default_value=self.end_x,
            callback=lambda s, v: self.update_setting("end_x", float(v)),
            parent=parent,
            width=150,
            step=1.0,
            format="%.2f",
        )
        self.register_param("end_x", self.end_x_widget)
        with dpg.tooltip(self.end_x_widget):
            dpg.add_text("Maximum X value to keep.", wrap=260)

        # Trim - Index
        self.crop_left_idx_widget = dpg.add_input_int(
            label="Crop Left",
            default_value=self.crop_left_idx,
            callback=lambda s, v: self.update_setting("crop_left_idx", max(0, v)),
            parent=parent,
            width=150,
            step=1,
        )
        self.register_param("crop_left_idx", self.crop_left_idx_widget)
        with dpg.tooltip(self.crop_left_idx_widget):
            dpg.add_text("Number of points to remove from the beginning (left).", wrap=260)

        self.crop_right_idx_widget = dpg.add_input_int(
            label="Crop Right",
            default_value=self.crop_right_idx,
            callback=lambda s, v: self.update_setting("crop_right_idx", max(0, v)),
            parent=parent,
            width=150,
            step=1,
        )
        self.register_param("crop_right_idx", self.crop_right_idx_widget)
        with dpg.tooltip(self.crop_right_idx_widget):
            dpg.add_text("Number of points to remove from the end (right).", wrap=260)

        # Trim - X Value
        self.crop_left_x_widget = dpg.add_input_float(
            label="Crop Left (dX)",
            default_value=self.crop_left_x,
            callback=lambda s, v: self.update_setting("crop_left_x", max(0.0, float(v))),
            parent=parent,
            width=150,
            step=1.0,
            format="%.2f",
        )
        self.register_param("crop_left_x", self.crop_left_x_widget)
        with dpg.tooltip(self.crop_left_x_widget):
            dpg.add_text("Amount of X interval to cut from the left (points with X < X_min + dX will be trimmed).", wrap=260)

        self.crop_right_x_widget = dpg.add_input_float(
            label="Crop Right (dX)",
            default_value=self.crop_right_x,
            callback=lambda s, v: self.update_setting("crop_right_x", max(0.0, float(v))),
            parent=parent,
            width=150,
            step=1.0,
            format="%.2f",
        )
        self.register_param("crop_right_x", self.crop_right_x_widget)
        with dpg.tooltip(self.crop_right_x_widget):
            dpg.add_text("Amount of X interval to cut from the right (points with X > X_max - dX will be trimmed).", wrap=260)

        self._update_ui_visibility()

    def _on_mode_change(self, sender: Union[int, str], value: str, *args: Any, **kwargs: Any) -> None:
        self.update_setting("mode", value)
        self._update_ui_visibility()

    def _on_unit_change(self, sender: Union[int, str], value: str, *args: Any, **kwargs: Any) -> None:
        self.update_setting("unit", value)
        self._update_ui_visibility()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        self._update_ui_visibility()

    def _update_ui_visibility(self) -> None:
        """Update visibility of parameter widgets depending on current mode and unit."""
        is_range = (self.mode == "Range (From ... To ...)")
        is_trim = (self.mode == "Trim Edges (Left / Right)")
        is_index = (self.unit == "Index (Points)")
        is_x = (self.unit == "X Value")

        # Range - Index
        show_range_idx = is_range and is_index
        if self.start_idx_widget and dpg.does_item_exist(self.start_idx_widget):
            dpg.configure_item(self.start_idx_widget, show=show_range_idx)
        if self.end_idx_widget and dpg.does_item_exist(self.end_idx_widget):
            dpg.configure_item(self.end_idx_widget, show=show_range_idx)

        # Range - X
        show_range_x = is_range and is_x
        if self.start_x_widget and dpg.does_item_exist(self.start_x_widget):
            dpg.configure_item(self.start_x_widget, show=show_range_x)
        if self.end_x_widget and dpg.does_item_exist(self.end_x_widget):
            dpg.configure_item(self.end_x_widget, show=show_range_x)

        # Trim - Index
        show_trim_idx = is_trim and is_index
        if self.crop_left_idx_widget and dpg.does_item_exist(self.crop_left_idx_widget):
            dpg.configure_item(self.crop_left_idx_widget, show=show_trim_idx)
        if self.crop_right_idx_widget and dpg.does_item_exist(self.crop_right_idx_widget):
            dpg.configure_item(self.crop_right_idx_widget, show=show_trim_idx)

        # Trim - X
        show_trim_x = is_trim and is_x
        if self.crop_left_x_widget and dpg.does_item_exist(self.crop_left_x_widget):
            dpg.configure_item(self.crop_left_x_widget, show=show_trim_x)
        if self.crop_right_x_widget and dpg.does_item_exist(self.crop_right_x_widget):
            dpg.configure_item(self.crop_right_x_widget, show=show_trim_x)

        # Tooltip text
        if self.tooltip_text and dpg.does_item_exist(self.tooltip_text):
            if is_range:
                dpg.set_value(
                    self.tooltip_text,
                    "Range Mode:\nSelects and keeps data points falling strictly between the Start and End bounds."
                )
            else:
                dpg.set_value(
                    self.tooltip_text,
                    "Trim Edges Mode:\nCuts/removes a specified amount of points or X interval from the left and/or right edges."
                )
