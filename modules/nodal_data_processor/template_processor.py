"""
Template for creating a new data processor.

Copy this file, rename the class, update docstring, and implement your logic.
Place in: modules/nodal_data_processor/processors/ or subfolders
"""

from typing import List, Tuple, Optional, Union
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class TemplateProcessor(DataProcessorBase):
    """Brief description of what this processor does."""
    
    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        
        # Define your parameters
        self.parameter1: float = 0.0
        self.parameter2: float = 1.0
    
    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Process the input data."""
        if not self.validate_data(x_list, y_list):
            return x_list.copy(), y_list.copy(), labels
        
        # YOUR PROCESSING LOGIC HERE
        output_x = [x * self.parameter2 + self.parameter1 for x in x_list]
        output_y = y_list.copy()
        
        return output_x, output_y, labels
    
    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls - use dpg.add_* then register_param()."""
        
        # Parameter 1
        widget1 = dpg.add_input_float(
            label="Parameter 1",
            default_value=self.parameter1,
            callback=lambda s, v: setattr(self, "parameter1", v),
            parent=parent,
            width=100,
            step=0.1
        )
        self.register_param("parameter1", widget1)
        
        # Parameter 2
        widget2 = dpg.add_input_float(
            label="Parameter 2",
            default_value=self.parameter2,
            callback=lambda s, v: setattr(self, "parameter2", v),
            parent=parent,
            width=100,
            step=0.1
        )
        self.register_param("parameter2", widget2)


