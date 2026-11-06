from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import dearpygui.dearpygui as dpg
from loguru import logger

from config.display_scaling import display_scaling
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from modules.nodal_data_processor.data_node_editor import DataNodeEditor


class NodalDataProcessor(WindowBase):
    """
    Nodal Data Processor module.
    
    Provides a node-based editor for creating modular data processing pipelines.
    Users can connect processor nodes to manipulate sample data without modifying source code.
    """
    
    # 1. Window Configuration
    label = "Nodal Data Processor"
    win_width = display_scaling.scale(800)
    win_height = display_scaling.scale(600)

    # 2. Input/Output definitions
    accepted_input_types = [IOTypes.SAMPLE, IOTypes.DATALIST, IOTypes.CMD_DICT]
    outputs = {
        "Processed Data": IOTypes.SAMPLE,
        "Metadata": IOTypes.ANY,
    }

    # 3. Persistent fields
    persistent_fields = ["label"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Nodal Data Processor window.
        """
        super().__init__(**kwargs)

        # Internal state
        self._is_closing: bool = False
        self.last_input_sample: Optional[Dict[str, Any]] = None
        self.last_output_sample: Optional[Dict[str, Any]] = None

        # Build the UI
        with self.window():
            # Create the node editor
            self.node_editor: DataNodeEditor = DataNodeEditor(parent_tag=self.winID, parent_win=self)

            # Status text
            self.status_text_tag: str = f"nodal_status_{self.UUID}"
            dpg.add_text("Ready", tag=self.status_text_tag, color=(200, 200, 200))

        self.update_permission()

    def update_permission(self) -> None:
        """Adapts UI according to application mode."""
        from core.app_state import app_state

        mode = app_state.mode
        is_user = (mode == "user")

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)
    
    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Callback for receiving data - accepts SAMPLE, legacy DATALIST formats, or CMD_DICT for commands.
        """
        data_type = kwargs.get('data_type')
        if data_type == IOTypes.CMD_DICT:
            cmd = kwargs.get('data') if kwargs.get('data') is not None else kwargs.get('sample')
            if cmd is None and len(args) > 0:
                cmd = args[0]
            if isinstance(cmd, dict) and 'load_flow' in cmd:
                file_path = cmd['load_flow']
                if isinstance(file_path, str) and file_path.endswith('.json'):
                    resolved_path = Path(file_path)
                    is_project_relative = file_path.startswith('/') or file_path.startswith('\\')
                    if not resolved_path.is_absolute() or is_project_relative:
                        # Try relative to project root
                        root_dir = Path(__file__).resolve().parents[2]
                        rel_path = file_path.lstrip('/\\')
                        test_path_root = root_dir / rel_path
                        
                        if test_path_root.exists():
                            resolved_path = test_path_root
                        else:
                            # Try relative to flows directory
                            test_path_flows = Path(self.node_editor.flows_dir) / rel_path
                            if test_path_flows.exists():
                                resolved_path = test_path_flows
                            else:
                                # Try fallback base name in flows directory
                                test_path_flows_base = Path(self.node_editor.flows_dir) / Path(file_path).name
                                if test_path_flows_base.exists():
                                    resolved_path = test_path_flows_base
                    self.node_editor.load_from_file(str(resolved_path))
            
            # Forward the command to connected modules on Processed Data
            for module in self.connections.get("Processed Data", []):
                try:
                    module.input_cb(data=cmd, data_type=IOTypes.CMD_DICT)
                except Exception as e:
                    logger.error(f"Error forwarding command to {module.label}: {e}")
            return


        # Try SAMPLE format first (new standard)
        sample: Optional[Dict[str, Any]] = kwargs.get('sample')
        input_sample: Optional[Dict[str, Any]] = None
        x_data: Optional[List[float]] = None
        y_data: Optional[List[float]] = None
        
        if sample:
            # SAMPLE format
            input_sample = sample
            x_data = sample.get('x')
            y_data = sample.get('y')
        else:
            # Legacy DATALIST format
            x_data = kwargs.get('x') if kwargs.get('x') is not None else kwargs.get('x_data')
            y_data = kwargs.get('y') if kwargs.get('y') is not None else kwargs.get('y_data')
            
            # Try parsing from args
            if not y_data and len(args) > 0:
                data = args[0]
                if isinstance(data, dict):
                    x_data = data.get('x')
                    y_data = data.get('y')
                elif isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
                        x_data = [p[0] for p in data]
                        y_data = [p[1] for p in data]
                    else:
                        y_data = data
                        x_data = list(range(len(y_data)))
            
            # Auto-generate X if missing
            if y_data and not x_data:
                x_data = list(range(len(y_data)))
            
            if not x_data or not y_data:
                logger.warning(f"Invalid data in {self.label}")
                return
        
        # Store input
        if not input_sample and x_data and y_data:
            input_sample = {
                "name": "Data",
                "uuid": None,
                "x": x_data,
                "y": y_data,
                "action": "select"
            }
        self.last_input_sample = input_sample
        
        # Check action - pass through rename/unselect without processing
        action: str = input_sample.get("action", "select") if input_sample else "select"
        
        if action in ["unselect", "rename", "delete"]:
            # Pass through without processing
            if input_sample:
                output_sample = input_sample.copy()
            else:
                output_sample = {
                    "name": "Data",
                    "uuid": None,
                    "x": x_data,
                    "y": y_data,
                    "action": action
                }
            self.last_output_sample = output_sample
            self._send_output(output_sample)

            # Propagate action to potential extra (auxiliary) samples
            base_id = input_sample.get("uuid") or input_sample.get("name") if input_sample else None
            if base_id:
                aux_keys = set()
                if hasattr(self, "node_editor") and self.node_editor:
                    for node in self.node_editor.node_map.values():
                        if getattr(node, "auxiliary_outputs", None):
                            aux_keys.update(node.auxiliary_outputs)
                
                for k in sorted(aux_keys):
                    extra_uuid = f"{base_id}_meta_{k}"
                    extra_sample = {
                        "uuid": extra_uuid,
                        "action": action
                    }
                    if action == "rename":
                        new_name = input_sample.get("name", "Data")
                        extra_sample["name"] = f"{new_name}_{k}"
                    self._send_output(extra_sample)
            return
        
        # Execute pipeline for select/update actions
        try:
            labels: Optional[List[str]] = input_sample.get('labels') if input_sample else None
            processed_x, processed_y, processed_labels, metadata = self.node_editor.execute_pipeline(x_data, y_data, labels=labels)
            
            output_sample = {
                "name": input_sample.get('name', 'Data') if input_sample else "Data",
                "uuid": input_sample.get('uuid') if input_sample else None,
                "x": processed_x,
                "y": processed_y,
                "labels": processed_labels,
                "action": action
            }
            
            self.last_output_sample = output_sample
            if not metadata or not metadata.get("__skip_main_output__"):
                self._send_output(output_sample)
            
            # Send metadata if present
            if metadata:
                self._send_metadata(metadata, base_sample=input_sample)
            
        except Exception as e:
            logger.error(f"Error in {self.label}: {e}")
            
    def reprocess_last_sample(self) -> None:
        """Re-process the last received sample using the current pipeline configuration."""
        if self.last_input_sample:
            self.input_cb(sample=self.last_input_sample)
            
    def _send_output(self, sample: Dict[str, Any]) -> None:
        """Send processed SAMPLE to connected modules."""
        targets = self.connections.get("Processed Data")
        if not targets:
            return
        for module in targets:
            try:
                module.input_cb(sample=sample, data_type=IOTypes.SAMPLE)
            except Exception as e:
                logger.error(f"Error sending to {module.label}: {e}")
 
    def _send_metadata(self, metadata: Dict[str, Any], base_sample: Optional[Dict[str, Any]] = None) -> None:
        """Send metadata to connected modules. Detects and sends auxiliary samples if found."""
        proc_targets = self.connections.get("Processed Data")
        meta_targets = self.connections.get("Metadata")
        if not proc_targets and not meta_targets:
            return

        # Check for extra samples to be displayed (superimposed)
        # Any dict in metadata that has 'x', 'y', and 'name' is treated as an extra sample
        # Derive its UUID from the base sample name for real-time updates
        if proc_targets:
            for k, v in metadata.items():
                if isinstance(v, dict) and "x" in v and "y" in v and "name" in v:
                    # If name starts with an underscore, prepend the base sample name
                    if base_sample and v.get("name", "").startswith("_"):
                        base_name = base_sample.get("name", "Data")
                        v["name"] = f"{base_name}{v['name']}"

                    if "uuid" not in v:
                        # Prefer UUID for stability across renames, fallback to name
                        base_id = base_sample.get("uuid") or base_sample.get("name") if base_sample else self.UUID
                        v["uuid"] = f"{base_id}_meta_{k}"
                    self._send_output(v)

        if not meta_targets:
            return

        # Proceed with standard metadata output
        data = metadata
        if len(metadata) == 1:
            data = next(iter(metadata.values()))
            
        for module in meta_targets:
            try:
                module.input_cb(data, data=data, sample=data, data_type=IOTypes.ANY)
            except Exception as e:
                logger.error(f"Error sending metadata to {module.label}: {e}")
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the module state."""
        state = super().serialize()
        state['pipeline'] = self.node_editor.serialize_pipeline()
        return state
    
    def deserialize(self, state: Dict[str, Any]) -> None:
        """Deserialize the module state."""
        super().deserialize(state)
        if 'pipeline' in state:
            self.node_editor.deserialize_pipeline(state['pipeline'])

    def on_close(self) -> None:
        """Clean up the node editor resources."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        if hasattr(self, 'node_editor'):
            self.node_editor.cleanup()
        super().on_close()


EXPORTED_CLASS = NodalDataProcessor
EXPORTED_NAME = "Nodal Data Processor"
