from typing import List, Tuple, Dict, Any, Optional, Union
import dearpygui.dearpygui as dpg
from abc import ABC, abstractmethod


class DataProcessorBase(ABC):
    """
    Base class for all data processors in the nodal data processor system.
    
    Each processor takes X and Y lists as input and returns modified X and Y lists.
    Processors can have configurable parameters exposed via DearPyGUI UI.
    """
    
    def __init__(self, node_id: Union[int, str]):
        """
        Initialize a data processor.
        
        Args:
            node_id: The DearPyGUI node ID that contains this processor
        """
        self.node_id: Union[int, str] = node_id
        self.uuid: int = dpg.generate_uuid()
        
        # Auto-generate name from class name
        self.name: str = self.__class__.__name__.replace("Processor", "").replace("_", " ")
        
        # Auto-extract description from docstring
        self.description: str = self._extract_description()
        
        # Track parameters for auto-serialization
        self._params: List[str] = []
        self._ui_widgets: Dict[str, Union[int, str]] = {}
        
        # Default outputs (can be overridden by subclasses)
        from core.input_output_types import IOTypes
        self.outputs: Dict[str, Any] = {"Out": IOTypes.DATALIST}
        self.auxiliary_outputs: List[str] = []
    
    def _extract_description(self) -> str:
        """Extract first line of docstring as description."""
        doc = self.__class__.__doc__
        if doc:
            for line in doc.strip().split('\n'):
                line = line.strip()
                if line:
                    return line
        return "No description available"
    
    def register_param(self, param_name: str, widget_id: Union[int, str]) -> Union[int, str]:
        """
        Register a parameter for auto-serialization.
        
        Usage:
            widget = dpg.add_input_float(...)
            self.register_param("offset", widget)
        
        Args:
            param_name: Name of the parameter attribute (e.g., "offset")
            widget_id: DPG widget ID returned by dpg.add_*()
            
        Returns:
            The widget_id (for chaining if needed)
        """
        if param_name not in self._params:
            self._params.append(param_name)
        self._ui_widgets[param_name] = widget_id
        return widget_id

    
    @abstractmethod
    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], Optional[List[str]]], Dict[str, Any]]:
        """
        Process the input data and return modified data.
        
        Args:
            x_list: List of X values
            y_list: List of Y values
            labels: Optional list of labels (e.g. image filenames) associated with each data point
            
        Returns:
            Tuple of (modified_x_list, modified_y_list), (modified_x_list, modified_y_list, modified_labels),
            or a dictionary mapping output names to their results.
        """
        pass
    
    def validate_data(self, x_list: List[float], y_list: List[float]) -> bool:
        """
        Validate that input data is properly formatted.
        
        Args:
            x_list: List of X values
            y_list: List of Y values
            
        Returns:
            True if data is valid, False otherwise
        """
        if not isinstance(x_list, list) or not isinstance(y_list, list):
            return False
        if len(x_list) != len(y_list):
            return False
        if len(x_list) == 0:
            return False
        return True
    
    def create_ui(self, parent: Union[int, str]) -> None:
        """
        Create UI controls for this processor's parameters.
        Override this method to add custom controls.
        
        Args:
            parent: The parent DearPyGUI item to attach UI to
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Auto-serialize all registered parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        params = {}
        for param_name in self._params:
            if hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)
        return params
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Auto-deserialize and sync UI for all registered parameters.
        
        Args:
            params: Dictionary of parameter names and values
        """
        for param_name, value in params.items():
            # Set attribute if it exists
            if hasattr(self, param_name):
                setattr(self, param_name, value)
                
                # Auto-sync UI widget if registered
                if param_name in self._ui_widgets:
                    widget_id = self._ui_widgets[param_name]
                    if dpg.does_item_exist(widget_id):
                        dpg.set_value(widget_id, value)
    
    def update_setting(self, param_name: str, value: Any) -> None:
        """
        Update a setting and trigger a re-run of the pipeline if the editor is present.
        """
        setattr(self, param_name, value)
        if hasattr(self, 'node_editor') and self.node_editor:
            parent_win = getattr(self.node_editor, 'parent_win', None)
            if parent_win and hasattr(parent_win, 'reprocess_last_sample'):
                parent_win.reprocess_last_sample()
    
    def get_display_name(self) -> str:
        """
        Get the display name for this processor.
        
        Returns:
            Display name string
        """
        return self.name


