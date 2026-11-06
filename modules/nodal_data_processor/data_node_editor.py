from pathlib import Path
import importlib.util
import inspect
import json
import dearpygui.dearpygui as dpg
from typing import Dict, List, Tuple, Optional, Any, Type, Union
from loguru import logger
from collections import defaultdict

from modules.nodal_data_processor.data_processor_base import DataProcessorBase
from core.file_explorer import file_explorer
from core.search_utils import fuzzy_score

class DataNodeEditor:
    """
    Node editor for data processing pipelines.
    Manages a graph of data processor nodes that can be connected and executed.
    """
    
    def __init__(self, parent_tag: Union[int, str], label: str = "Data Node Editor", parent_win: Optional[Any] = None) -> None:
        """
        Initialize the data node editor.
        
        Args:
            parent_tag: Tag of the parent window
            label: Label for the editor
            parent_win: Reference to the wrapping window class instance
        """
        self.parent_tag: Union[int, str] = parent_tag
        self.label: str = label
        self.parent_win: Optional[Any] = parent_win
        self.uuid: int = dpg.generate_uuid()
        
        # Flows default folder
        self.flows_dir: Path = Path(__file__).resolve().parent / "flows"
        try:
            self.flows_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        
        # Node tracking
        self.node_map: Dict[Union[int, str], DataProcessorBase] = {}  # Maps node_id to processor instance
        self.link_map: Dict[Union[int, str], Tuple[Union[int, str], Union[int, str]]] = {}  # Maps link_id to (output_attr_id, input_attr_id)
        self._pinned_node_labels: Dict[Union[int, str], str] = {}
        
        # UI tags
        self.editor_tag: str = f"data_node_editor_{self.uuid}"
        self.popup_tag: str = f"data_node_popup_{self.uuid}"
        self.anchor_node_tag: str = f"data_anchor_node_{self.uuid}"
        self.handler_registry_tag: str = f"handler_registry_{self.uuid}"
        
        # Mouse position
        self.mouse_pos: List[float] = [0.0, 0.0]
        
        # Flag to prevent race condition between left_click_callback and add_node
        self._popup_click_handled: bool = False
        
        # Load available processors
        self.available_processors: Dict[str, Type[DataProcessorBase]] = self._load_processors()
        
        # Build the UI
        self._build_ui()
    
    def _load_processors(self, processors_dir: Optional[Union[str, Path]] = None) -> Dict[str, Type[DataProcessorBase]]:
        """
        Discover and load all data processor classes from the processors directory.
        Supports subdirectories for category organization.
        
        Args:
            processors_dir: Path to the processors directory. If None, uses default location.
            
        Returns:
            Dictionary mapping "category/processor_name" to their classes
            Root level processors use just "processor_name"
        """
        p_dir = Path(processors_dir) if processors_dir is not None else Path(__file__).resolve().parent / "processors"
        
        processors: Dict[str, Type[DataProcessorBase]] = {}
        
        if not p_dir.exists():
            logger.warning(f"Processors directory not found: {p_dir}")
            return processors
        
        # Recursively scan for Python files
        def scan_directory(directory: Path, category_path: str = "") -> None:
            """Recursively scan directories for processor files."""
            try:
                for item in directory.iterdir():
                    # If it's a directory, scan it recursively
                    if item.is_dir():
                        # Skip __pycache__ and directories starting with _
                        if item.name.startswith("_"):
                            continue
                        
                        # Build category path (e.g., "offset", "normalize")
                        # Use forward slashes for consistency across platforms
                        new_category = f"{category_path}/{item.name}" if category_path else item.name
                        scan_directory(item, new_category)
                    
                    # If it's a Python file, try to load it
                    elif item.is_file() and item.name.endswith(".py") and not item.name.startswith("_"):
                        filepath = str(item.resolve())
                        module_name = item.stem
                        
                        try:
                            # Load the module
                            spec = importlib.util.spec_from_file_location(module_name, filepath)
                            if spec is None or spec.loader is None:
                                logger.warning(f"Could not load spec for {item.name}")
                                continue
                            
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Find DataProcessorBase subclasses in the module
                            for name, obj in inspect.getmembers(module):
                                if (inspect.isclass(obj) and 
                                    any(base.__name__ == "DataProcessorBase" for base in obj.__mro__) and 
                                    obj.__name__ != "DataProcessorBase"):
                                    
                                    # Build the key with category path
                                    if category_path:
                                        processor_key = f"{category_path}/{obj.__name__}"
                                    else:
                                        processor_key = obj.__name__
                                    
                                    processors[processor_key] = obj
                        
                        except Exception as e:
                            logger.error(f"Failed to load processor from {item.name}: {e}")
            
            except Exception as e:
                logger.error(f"Failed to scan directory {directory}: {e}")
        
        # Start scanning from root processors directory
        scan_directory(p_dir)
        
        return processors
    
    def _get_processor_info(self, processor_class: Type[DataProcessorBase]) -> Dict[str, str]:
        """
        Get information about a processor class.
        
        Args:
            processor_class: The processor class to inspect
            
        Returns:
            Dictionary with 'name' and 'description' keys
        """
        # Create a dummy instance to get display name
        try:
            # Use a dummy node_id for inspection
            dummy_instance = processor_class(node_id=-1)
            name = dummy_instance.get_display_name()
            description = dummy_instance.description
        except Exception as e:
            logger.warning(f"Could not create instance of {processor_class.__name__}: {e}")
            name = processor_class.__name__
            description = processor_class.__doc__ or "No description"
        
        return {
            "name": name,
            "description": description
        }
    
    def _build_ui(self) -> None:
        """Build the node editor UI."""
        with dpg.group(parent=self.parent_tag):
            # Control buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Save Subflow",
                    callback=lambda *args: self._show_save_dialog()
                )
                dpg.add_button(
                    label="Load Subflow",
                    callback=lambda *args: self._show_load_dialog()
                )
                dpg.add_button(
                    label="Clear All",
                    callback=lambda *args: self.clear_all_nodes()
                )
            
            # Node editor canvas
            with dpg.node_editor(
                callback=self.link_callback,
                delink_callback=self.delink_callback,
                tag=self.editor_tag,
                minimap=True,
                minimap_location=dpg.mvNodeMiniMap_Location_BottomRight
            ):
                # Raw Data input node (non-draggable)
                with dpg.node(label="Raw Data", draggable=False, tag=self.anchor_node_tag):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                        dpg.add_text("Out")
        
        # Global Handler Registry (tagged for cleanup)
        self.handler_registry_tag = f"handler_registry_{self.uuid}"
        with dpg.handler_registry(tag=self.handler_registry_tag):
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right,
                callback=self.right_click_callback
            )
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=self.left_click_callback
            )
        
        # Popup menu for adding nodes
        self._build_popup_menu()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up the handler registry
        if dpg.does_item_exist(self.handler_registry_tag):
            dpg.delete_item(self.handler_registry_tag)
            
        # Clean up any tagged items if needed
        if dpg.does_item_exist(self.popup_tag):
            dpg.delete_item(self.popup_tag)
    
    def _populate_module_list(self, filter_text: str) -> None:
        """Populate the list of available processors based on filter text."""
        list_tag = f"{self.popup_tag}_list"
        
        # Clear existing buttons
        if dpg.does_item_exist(list_tag):
            dpg.delete_item(list_tag, children_only=True)
        else:
            return

        filter_text = filter_text.lower()
        
        # Organize processors by category
        categories: Dict[str, List[Tuple[str, Type]]] = {}
        
        for proc_key, proc_class in self.available_processors.items():
            # Split path: "offset/OffsetXProcessor" -> category="offset", name="OffsetXProcessor"
            if "/" in proc_key:
                parts = proc_key.split("/")
                category = "/".join(parts[:-1])  # Support nested categories
                proc_name = parts[-1]
            else:
                category = ""  # Root category
                proc_name = proc_key
            
            # Skip if filtered out
            full_name = f"{category}/{proc_name}" if category else proc_name
            if filter_text:
                score = fuzzy_score(filter_text, full_name)
                # Also check category and name individually for better matching
                score = max(score, fuzzy_score(filter_text, proc_name))
                if category:
                    score = max(score, fuzzy_score(filter_text, category))
                
                if score == 0:
                    continue
            
            if category not in categories:
                categories[category] = []
            categories[category].append((proc_name, proc_class))
        
        # Sort categories, with root ("") first
        sorted_categories = sorted(categories.items(), key=lambda x: (x[0] != "", x[0]))
        
        # Build UI for each category
        for category, processors in sorted_categories:
            # If root category (no folder), add processors directly
            if not category:
                # Sort processors within root category
                processors.sort(key=lambda x: x[0])
                
                for proc_name, proc_class in processors:
                    info = self._get_processor_info(proc_class)
                    
                    # Create button for this processor
                    btn = dpg.add_button(
                        label=info["name"],
                        callback=self.add_node,
                        user_data=proc_class,
                        width=-1,
                        parent=list_tag
                    )
                    
                    # Add tooltip
                    if info["description"]:
                        with dpg.tooltip(parent=btn):
                            dpg.add_text(info["description"], wrap=250)
            else:
                # Use collapsible header for categories
                # Auto-expand when filtering, collapsed otherwise
                default_open = bool(filter_text)
                
                with dpg.collapsing_header(label=category, default_open=default_open, parent=list_tag):
                    # Sort processors within category
                    processors.sort(key=lambda x: x[0])
                    
                    for proc_name, proc_class in processors:
                        info = self._get_processor_info(proc_class)
                        
                        # Create button for this processor
                        btn = dpg.add_button(
                            label=info["name"],
                            callback=self.add_node,
                            user_data=proc_class,
                            width=-1
                        )
                        
                        # Add tooltip
                        if info["description"]:
                            with dpg.tooltip(parent=btn):
                                dpg.add_text(info["description"], wrap=250)

    def _on_search_change(self, sender: Union[int, str], app_data: str, user_data: Any = None, *args: Any) -> None:
        """Handle search input change."""
        self._populate_module_list(app_data)

    def _build_popup_menu(self) -> None:
        """Build the popup menu for adding processor nodes."""
        with dpg.window(
            tag=self.popup_tag,
            show=False,
            no_title_bar=True,
            width=300,
            height=400
        ):
            # Search bar
            dpg.add_input_text(
                tag=f"{self.popup_tag}_search",
                hint="Search...",
                width=-1,
                callback=self._on_search_change
            )
            
            dpg.add_separator()
            
            # Scrollable list container
            with dpg.child_window(tag=f"{self.popup_tag}_list", height=-1):
                pass
                
        # Initial population
        self._populate_module_list("")

    def _show_error_dialog(self, title: str, message: str) -> None:
        """Show a modal error dialog."""
        # Clean up previous dialog if exists
        tag = f"error_dialog_{self.uuid}"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
            
        with dpg.window(label=title, modal=True, show=True, tag=tag, width=400, height=200):
            dpg.add_text(message, wrap=380, color=(255, 100, 100))
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=75, callback=lambda *args: dpg.delete_item(tag))
    
    def get_mouse_pos(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate mouse position relative to the editor area.
        
        Returns:
            Tuple of (relative_pos, absolute_pos)
        """
        abs_pos = dpg.get_mouse_pos(local=False)
        ref_screen_pos = dpg.get_item_rect_min(self.anchor_node_tag)
        ref_editor_pos = dpg.get_item_pos(self.anchor_node_tag)
        pan_offset = (
            ref_editor_pos[0] - ref_screen_pos[0],
            ref_editor_pos[1] - ref_screen_pos[1]
        )
        rel_pos = (abs_pos[0] + pan_offset[0], abs_pos[1] + pan_offset[1])
        return rel_pos, abs_pos
    
    def right_click_callback(self, sender: Union[int, str], app_data: Any, user_data: Any = None, *args: Any) -> None:
        """Open popup menu at mouse location if editor is hovered."""
        if dpg.does_item_exist(self.editor_tag) and dpg.is_item_hovered(self.editor_tag):
            # If ANY child (node, anchor) is hovered, let their own popup handle it.
            for node_id in self.node_map:
                if dpg.get_item_state(node_id).get('hovered'):
                    return
            # Also check anchor node
            if dpg.get_item_state(self.anchor_node_tag).get('hovered'):
                return
                
            self.mouse_pos, absolute_mouse_pos = self.get_mouse_pos()
            
            # Reset search and repopulate list
            search_tag = f"{self.popup_tag}_search"
            if dpg.does_item_exist(search_tag):
                dpg.set_value(search_tag, "")
                self._populate_module_list("")
            
            dpg.configure_item(self.popup_tag, pos=absolute_mouse_pos, show=True)
            dpg.focus_item(search_tag)
    
    def _is_any_popup_child_hovered(self) -> bool:
        """Recursively check if any child of the popup is hovered."""
        def _check_children(item_id: Union[int, str]) -> bool:
            try:
                if dpg.is_item_hovered(item_id):
                    return True
            except Exception:
                pass
            # Check all child slots (0 and 1)
            for slot in (0, 1):
                try:
                    children = dpg.get_item_children(item_id, slot)
                    if children:
                        for child in children:
                            if _check_children(child):
                                return True
                except Exception:
                    pass
            return False
        return _check_children(self.popup_tag)

    def left_click_callback(self, sender: Union[int, str], app_data: Any, user_data: Any = None, *args: Any) -> None:
        """Hide popup menu if clicking outside it."""
        if not dpg.does_item_exist(self.popup_tag):
            return
 
        if dpg.is_item_visible(self.popup_tag):
            # If add_node already handled this click, just reset the flag
            if self._popup_click_handled:
                self._popup_click_handled = False
                return
            
            # Check if clicking on popup or any of its children
            if self._is_any_popup_child_hovered():
                return
            
            dpg.configure_item(self.popup_tag, show=False)
    
    def add_node(self, sender: Optional[Union[int, str]], app_data: Any, processor_class: Type[DataProcessorBase], *args: Any) -> None:
        """
        Add a new processor node to the editor.
        """
        self._popup_click_handled = True
        
        if processor_class is None:
            self._show_error_dialog("Error", "Processor class is None")
            return

        try:
            # Get name from class
            processor_name = processor_class.__name__
            
            # Ensure valid position
            pos = self.mouse_pos
            if pos[0] == 0 and pos[1] == 0:
                pos = [100, 100]
            
            # Create the node
            with dpg.node(
                label=processor_name,
                parent=self.editor_tag,
                pos=pos
            ) as node_id:
                
                # Input attribute
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                    dpg.add_text("In")
                
                # Create processor instance
                processor = processor_class(node_id=node_id)
                processor.node_editor = self
                self.node_map[node_id] = processor
                
                # UI container for processor controls
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    ui_group = dpg.add_group()
                    processor.create_ui(ui_group)
                
                # Output attributes
                for output_name in processor.outputs.keys():
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                        dpg.add_text(output_name)
            
                # Right-click popup for this node
                with dpg.popup(node_id, mousebutton=dpg.mvMouseButton_Right):
                    dpg.add_button(
                        label="Send to Menu Bar",
                        callback=self._pin_node_to_menu_bar,
                        user_data=node_id
                    )
                    dpg.add_separator()
                    dpg.add_button(
                        label="Delete Node",
                        callback=self.delete_node,
                        user_data=node_id
                    )
                
                dpg.configure_item(self.popup_tag, show=False)
            
        except Exception as e:
            msg = f"Failed to add node: {e}"
            logger.error(msg)
            import traceback
            traceback.print_exc()
            self._show_error_dialog("Node Creation Error", msg)
    
    def _pin_node_to_menu_bar(self, sender: Union[int, str], app_data: Any, node_id: Union[int, str], *args: Any) -> None:
        """Pin this processor node as an entry in the main window's Pinned menu."""
        try:
            from core.main_win import main_win
        except ImportError:
            logger.warning("Could not import main_win — menu bar pinning unavailable")
            return

        processor = self.node_map.get(node_id)
        if processor is None:
            return

        # Use the processor display name as the menu label
        label = processor.get_display_name()

        # Determine which window to show: walk up the parent chain to find the
        # NodalDataProcessor_win that wraps this node editor. As a fallback,
        # just show/focus the parent DPG window.
        parent_window_tag = self.parent_tag

        def _show_and_focus():
            if dpg.does_item_exist(parent_window_tag):
                dpg.configure_item(parent_window_tag, show=True)
                dpg.focus_item(parent_window_tag)

        main_win.add_pinned_menu_item(label, _show_and_focus)

        # Remember which label is pinned on this node for cleanup on delete
        if not hasattr(self, '_pinned_node_labels'):
            self._pinned_node_labels: Dict[Union[int, str], str] = {}
        self._pinned_node_labels[node_id] = label

    def delete_node(self, sender: Union[int, str], app_data: Any, node_id: Union[int, str], *args: Any) -> None:
        """Delete a node from the editor."""
        if node_id in self.node_map:
            # Remove from map
            del self.node_map[node_id]

            # Remove pinned menu bar entry if this node was pinned
            pinned = getattr(self, '_pinned_node_labels', {})
            if node_id in pinned:
                label = pinned.pop(node_id)
                try:
                    from core.main_win import main_win
                    main_win.remove_pinned_menu_item(label)
                except ImportError:
                    pass
            
            # Remove links connected to this node
            to_remove = []
            for link_id, (src_attr, tgt_attr) in self.link_map.items():
                # Safety check: ensure attributes still exist
                if not dpg.does_item_exist(src_attr) or not dpg.does_item_exist(tgt_attr):
                    to_remove.append(link_id)
                    continue
                    
                src_node = dpg.get_item_parent(src_attr)
                tgt_node = dpg.get_item_parent(tgt_attr)
                if src_node == node_id or tgt_node == node_id:
                    to_remove.append(link_id)
            
            for link_id in to_remove:
                dpg.delete_item(link_id)
                del self.link_map[link_id]
            
            # Delete the node UI
            dpg.delete_item(node_id)
    
    def clear_all_nodes(self) -> None:
        """Clear all nodes from the editor."""
        # Delete all links
        for link_id in list(self.link_map.keys()):
            if dpg.does_item_exist(link_id):
                dpg.delete_item(link_id)
        self.link_map.clear()
        
        # Delete all nodes
        for node_id in list(self.node_map.keys()):
            if dpg.does_item_exist(node_id):
                dpg.delete_item(node_id)
        self.node_map.clear()
        
        logger.success("Cleared all nodes from data editor")
    def link_callback(self, sender: Union[int, str], app_data: Tuple[Union[int, str], Union[int, str]], user_data: Any = None, *args: Any) -> None:
        """Callback when a link is created between nodes."""
        attr_a, attr_b = app_data
        
        # Get parent nodes
        node_a = dpg.get_item_parent(attr_a)
        node_b = dpg.get_item_parent(attr_b)

        # Determine which one is the output
        from_node, from_attr, to_node, to_attr = None, None, None, None
        
        # Check if attr_a is an output
        conf_a = dpg.get_item_configuration(attr_a)
        if int(conf_a.get("attribute_type", -1)) == int(dpg.mvNode_Attr_Output):
            from_node, from_attr = node_a, attr_a
            to_node, to_attr = node_b, attr_b
        else:
            conf_b = dpg.get_item_configuration(attr_b)
            if int(conf_b.get("attribute_type", -1)) == int(dpg.mvNode_Attr_Output):
                from_node, from_attr = node_b, attr_b
                to_node, to_attr = node_a, attr_a
            else:
                # Should not happen in a valid DPG link
                return

        # Check if from_node already has an output connection
        for existing_from, existing_to in self.link_map.values():
            if not dpg.does_item_exist(existing_from):
                continue
            existing_from_node = dpg.get_item_parent(existing_from)
            if existing_from_node == from_node:
                return
        
        # Check if to_node already has an input connection
        for existing_from, existing_to in self.link_map.values():
            if not dpg.does_item_exist(existing_to):
                continue
            existing_to_node = dpg.get_item_parent(existing_to)
            if existing_to_node == to_node:
                return
        
        # Build temporary connection graph with the new connection
        temp_connections = self._get_node_connections()
        
        # Only check nodes in node_map (exclude anchor node)
        if (from_node in self.node_map or from_node == self.anchor_node_tag) and to_node in self.node_map:
            # Add the new connection temporarily
            if from_node not in temp_connections:
                temp_connections[from_node] = []
            temp_connections[from_node].append(to_node)
            
            # Check for cycles
            if self._has_cycle(temp_connections):
                return
        
        # All validations passed - create the link
        link_id = dpg.generate_uuid()
        dpg.add_node_link(from_attr, to_attr, parent=self.editor_tag, tag=link_id)
        self.link_map[link_id] = (from_attr, to_attr)

    
    def delink_callback(self, sender: Union[int, str], app_data: Union[int, str], user_data: Any = None, *args: Any) -> None:
        """Callback when a link is removed."""
        link_id = app_data
        
        if link_id in self.link_map:
            del self.link_map[link_id]
        
        dpg.delete_item(link_id)
    
    def _has_cycle(self, connections: Dict[Union[int, str], List[Union[int, str]]]) -> bool:
        """
        Detect if the connection graph has a cycle using DFS.
        
        Args:
            connections: Dictionary mapping node_id to list of downstream node_ids
            
        Returns:
            True if a cycle is detected, False otherwise
        """
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in connections.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True  # Cycle detected
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes
        for node in connections.keys():
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def _get_node_connections(self) -> Dict[Union[int, str], List[Union[int, str]]]:
        """
        Build a map of node connections for execution order.
        
        Returns:
            Dictionary mapping node_id to list of downstream node_ids
        """
        connections: Dict[Union[int, str], List[Union[int, str]]] = defaultdict(list)
        
        for from_attr, to_attr in self.link_map.values():
            # Safety check: ensure attributes still exist
            if not dpg.does_item_exist(from_attr) or not dpg.does_item_exist(to_attr):
                continue
                
            from_node = dpg.get_item_parent(from_attr)
            to_node = dpg.get_item_parent(to_attr)
            
            # Include connections FROM anchor node TO processor nodes
            # But only add to connections if both nodes exist
            # Skip connections where to_node is not in node_map (invalid)
            if to_node not in self.node_map:
                continue
            
            # Check if from_node is the anchor
            is_anchor = (from_node == self.anchor_node_tag)
            if not is_anchor and dpg.does_item_exist(from_node):
                try:
                    if dpg.get_item_alias(from_node) == self.anchor_node_tag:
                        is_anchor = True
                except Exception:
                    pass

            # If from_node is in node_map or is the anchor, track the connection
            if from_node in self.node_map or is_anchor:
                # Use the normalized anchor tag for consistency
                src_key = self.anchor_node_tag if is_anchor else from_node
                connections[src_key].append(to_node)
            else:
                # Unknown from_node, skip
                continue
        
        # Clean up orphaned links (links with deleted nodes)
        orphaned_links = [
            link_id for link_id, (src_attr, tgt_attr) in self.link_map.items()
            if not dpg.does_item_exist(src_attr) or not dpg.does_item_exist(tgt_attr)
        ]
        for link_id in orphaned_links:
            logger.debug(f"Cleaning up orphaned link {link_id}")
            if link_id in self.link_map:
                del self.link_map[link_id]
        
        return connections
    
    def _topological_sort(self, connections: Dict[Union[int, str], List[Union[int, str]]]) -> List[Union[int, str]]:
        """
        Perform topological sort to determine execution order.
        Only includes nodes that are part of the connection graph.
        
        Args:
            connections: Dictionary of node connections
            
        Returns:
            List of node_ids in execution order
        """
        # Build set of all nodes that have connections
        connected_nodes = set(connections.keys())
        for targets in connections.values():
            connected_nodes.update(targets)
        
        # Only consider nodes that are in the connection graph
        in_degree = {node_id: 0 for node_id in connected_nodes}
        for targets in connections.values():
            for target in targets:
                if target in in_degree:
                    in_degree[target] += 1
        
        # Find starting nodes - ONLY start from anchor to ensure 
        # we only process the flow connected to raw data
        queue = []
        if self.anchor_node_tag in in_degree:
            # Anchor itself should have in_degree 0 as it's a source
            if in_degree[self.anchor_node_tag] == 0:
                queue.append(self.anchor_node_tag)
        
        result = []
        while queue:
            node_id = queue.pop(0)
            
            # Add to result if it's a processor node (exclude the anchor)
            if node_id in self.node_map:
                result.append(node_id)
            
            # Reduce in-degree for connected nodes
            for target in connections.get(node_id, []):
                if target in in_degree:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        queue.append(target)
        
        return result
    
    def _get_output_name_from_attr(self, node_id: Union[int, str], attr_id: Union[int, str]) -> str:
        """
        Resolve the output name for a given attribute ID on a node.
        
        Args:
            node_id: DPG node ID
            attr_id: attribute ID
            
        Returns:
            Output name (e.g., "Out", "Equation")
        """
        processor = self.node_map.get(node_id)
        if not processor:
            return "Out"
            
        # Match using normalized IDs
        try:
            target_id = dpg.get_alias_id(attr_id) if isinstance(attr_id, str) else int(attr_id)
        except:
            target_id = attr_id

        children = dpg.get_item_children(node_id, slot=1)
        output_attrs = []
        for child in children:
            if "mvNodeAttribute" in dpg.get_item_type(child):
                try:
                    attr_type = int(dpg.get_item_configuration(child).get("attribute_type", -1))
                    if attr_type == int(dpg.mvNode_Attr_Output):
                        output_attrs.append(child)
                except (ValueError, TypeError):
                    continue
        
        try:
            # Normalize list IDs for comparison
            int_output_ids = [int(oid) for oid in output_attrs]
            index = int_output_ids.index(int(target_id))
            return list(processor.outputs.keys())[index]
        except (ValueError, IndexError, TypeError):
            return "Out"

    def execute_pipeline(
        self,
        x_data: Optional[List[float]] = None,
        y_data: Optional[List[float]] = None,
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]], Dict[str, Any]]:
        """
        Execute the data processing pipeline.
        
        Args:
            x_data: Input X data (if None, uses empty list)
            y_data: Input Y data (if None, uses empty list)
            labels: Optional list of labels (e.g. image filenames) for each data point
            
        Returns:
            Tuple of (processed_x_data, processed_y_data, processed_labels, metadata_dict)
        """
        if x_data is None:
            x_data = []
        if y_data is None:
            y_data = []
        
        # Build connection graph
        connections = self._get_node_connections()
        
        # Get execution order
        execution_order = self._topological_sort(connections)
        
        if not execution_order:
            return x_data, y_data, labels, {}
        
        # Data cache for each node: result from processor.process()
        node_data: Dict[Union[int, str], Any] = {}
        # Global metadata collection
        pipeline_metadata: Dict[str, Any] = {}
        
        # Execute nodes in order
        for node_id in execution_order:
            processor = self.node_map.get(node_id)
            if processor is None:
                continue
            
            # Get input data
            # Find upstream nodes
            upstream_nodes = [
                src_node for src_node, targets in connections.items()
                if node_id in targets
            ]
            
            input_x, input_y, input_labels = x_data, y_data, labels
            
            if upstream_nodes:
                # Use data from first upstream node
                upstream_id = upstream_nodes[0]
                if upstream_id in node_data:
                    results = node_data[upstream_id]
                    
                    # Find which output port is connected to this node's input
                    output_name = "Out" # Default
                    for link_id, (src_attr, tgt_attr) in self.link_map.items():
                        if dpg.get_item_parent(tgt_attr) == node_id and dpg.get_item_parent(src_attr) == upstream_id:
                            output_name = self._get_output_name_from_attr(upstream_id, src_attr)
                            break
                    
                    if isinstance(results, dict):
                        # Multi-output processor
                        data = results.get(output_name)
                        if isinstance(data, (tuple, list)) and len(data) >= 2:
                             input_x, input_y = data[0], data[1]
                             input_labels = data[2] if len(data) > 2 else input_labels
                    elif isinstance(results, (tuple, list)):
                        # Handle standard 2 or 3-tuple return
                        input_x = results[0]
                        input_y = results[1]
                        if len(results) > 2:
                            input_labels = results[2]
            
            # Process the data
            try:
                result = processor.process(input_x, input_y, labels=input_labels)
                node_data[node_id] = result
                
                # Collect metadata (any key other than "Out")
                if isinstance(result, dict):
                    for k, v in result.items():
                        if k != "Out":
                            # Use node name or ID to avoid collisions if necessary, 
                            # but for now just store globally
                            pipeline_metadata[k] = v
                            
            except TypeError:
                try:
                    result = processor.process(input_x, input_y)
                    if isinstance(result, tuple):
                        if len(result) == 2:
                            node_data[node_id] = (result[0], result[1], input_labels)
                        else:
                            node_data[node_id] = result
                    else:
                        node_data[node_id] = result
                        if isinstance(result, dict):
                            for k, v in result.items():
                                if k != "Out":
                                    pipeline_metadata[k] = v
                except Exception as e:
                    logger.error(f"Error executing processor {processor.name}: {e}")
                    node_data[node_id] = (input_x, input_y, input_labels)
            except Exception as e:
                logger.error(f"Error executing processor {processor.name}: {e}")
                node_data[node_id] = (input_x, input_y, input_labels)  # Pass through on error
        
        # Find the last node in execution order
        if execution_order:
            last_node_id = execution_order[-1]
            if last_node_id in node_data:
                final_result = node_data[last_node_id]
                if isinstance(final_result, dict):
                    # For multi-output nodes, default to "Out" as the pipeline result
                    # or the first available output
                    data = final_result.get("Out") or list(final_result.values())[0]
                    if isinstance(data, (tuple, list)) and len(data) >= 2:
                        lx, ly = data[0], data[1]
                        llabels = data[2] if len(data) > 2 else labels
                        return lx, ly, llabels, pipeline_metadata
                elif isinstance(final_result, (tuple, list)):
                    lx = final_result[0]
                    ly = final_result[1]
                    llabels = final_result[2] if len(final_result) > 2 else labels
                    return lx, ly, llabels, pipeline_metadata
        
        return x_data, y_data, labels, pipeline_metadata
    
    def serialize_pipeline(self) -> Dict[str, Any]:
        """
        Serialize the current pipeline to a dictionary.
        
        Returns:
            Dictionary containing nodes, connections, and metadata
        """
        nodes_data = []
        id_mapping = {}  # Maps real node_id to sequential "node_0", "node_1"
        
        # Serialize nodes
        for idx, (node_id, processor) in enumerate(self.node_map.items()):
            seq_id = f"node_{idx}"
            id_mapping[node_id] = seq_id
            
            # Get processor info
            processor_class = processor.__class__
            module_path = processor_class.__module__.replace('modules.nodal_data_processor.processors.', '')
            
            # Get node position
            pos = dpg.get_item_pos(node_id) if dpg.does_item_exist(node_id) else [100, 100]
            
            nodes_data.append({
                "id": seq_id,
                "processor_type": processor_class.__name__,
                "processor_module": module_path,
                "parameters": processor.get_parameters(),
                "position": list(pos)
            })
        
        # Serialize connections
        connections_data = []
        for from_attr, to_attr in self.link_map.values():
            if not dpg.does_item_exist(from_attr) or not dpg.does_item_exist(to_attr):
                continue
                
            from_node = dpg.get_item_parent(from_attr)
            to_node = dpg.get_item_parent(to_attr)
            
            is_anchor = (from_node == self.anchor_node_tag)
            if not is_anchor and dpg.does_item_exist(from_node):
                try:
                    if dpg.get_item_alias(from_node) == self.anchor_node_tag:
                        is_anchor = True
                except Exception:
                    pass

            if is_anchor:
                from_id_mapped = "raw_data"
            elif from_node in self.node_map:
                from_id_mapped = id_mapping[from_node]
            else:
                continue
            
            if to_node in self.node_map:
                to_id_mapped = id_mapping[to_node]
            else:
                continue
            
            connections_data.append({
                "from_node": from_id_mapped,
                "to_node": to_id_mapped
            })
        
        return {
            "version": "1.0",
            "nodes": nodes_data,
            "connections": connections_data
        }
    
    def deserialize_pipeline(self, data: Dict[str, Any], clear_existing: bool = True) -> None:
        """
        Load pipeline from dictionary data.
        
        Args:
            data: Dictionary containing pipeline data
            clear_existing: If True, clear existing nodes before loading
        """
        if clear_existing:
            self.clear_all_nodes()
        
        id_mapping = {}  # Maps "node_0" to real DPG node_id
        
        # Create nodes
        for node_data in data.get("nodes", []):
            processor_module = node_data["processor_module"]
            processor_type = node_data["processor_type"]
            
            # Find processor class
            processor_class = None
            proc_name = None
            for name, proc_cls in self.available_processors.items():
                if proc_cls.__name__ == processor_type:
                    processor_class = proc_cls
                    proc_name = name
                    break
            
            if processor_class is None:
                logger.warning(f"Processor {processor_type} not found, skipping")
                continue
            
            # Set mouse position for node placement
            self.mouse_pos = node_data.get("position", [100, 100])
            
            # Create node (add_node uses processor_class as user_data argument)
            self.add_node(sender=None, app_data=None, processor_class=processor_class)
            
            # Get the newly created node (last one added)
            if self.node_map:
                node_id = list(self.node_map.keys())[-1]
                processor = self.node_map[node_id]
                
                # Set parameters
                processor.set_parameters(node_data["parameters"])
                
                # Track mapping
                id_mapping[node_data["id"]] = node_id
        
        # Create connections
        for conn in data.get("connections", []):
            from_node_str = conn["from_node"]
            if from_node_str == "raw_data":
                from_id = self.anchor_node_tag
            else:
                from_id = id_mapping.get(from_node_str)
                
            to_id = id_mapping.get(conn["to_node"])
            
            if from_id and to_id:
                # Get attributes for connection
                # Input attribute is first, output attribute is last
                from_attrs = dpg.get_item_children(from_id, slot=1)
                to_attrs = dpg.get_item_children(to_id, slot=1)
                
                if from_attrs and to_attrs:
                    # Find output attribute (last attribute in from_node)
                    from_attr = None
                    for attr_id in reversed(from_attrs):
                        if dpg.get_item_type(attr_id) == "mvAppItemType::mvNodeAttribute":
                            from_attr = attr_id
                            break
                    
                    # Find input attribute (first attribute in to_node)
                    to_attr = None
                    for attr_id in to_attrs:
                        if dpg.get_item_type(attr_id) == "mvAppItemType::mvNodeAttribute":
                            to_attr = attr_id
                            break
                    
                    # Create connection
                    if from_attr and to_attr:
                        link_id = dpg.generate_uuid()
                        dpg.add_node_link(from_attr, to_attr, parent=self.editor_tag, tag=link_id)
                        self.link_map[link_id] = (from_attr, to_attr)
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save pipeline to JSON file.
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self.serialize_pipeline()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.success(f"Saved pipeline to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load pipeline from JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.deserialize_pipeline(data)
            logger.success(f"Loaded pipeline from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return False
    
    def _show_save_dialog(self):
        """Show file save dialog for pipeline."""
        filepath = file_explorer.save_file(
            default_path=str(self.flows_dir),
            default_name="pipeline.json",
            extensions=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.save_to_file(filepath)
    
    def _show_load_dialog(self):
        """Show file load dialog for pipeline."""
        filepath = file_explorer.select_file(
            default_path=str(self.flows_dir),
            extensions=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.load_from_file(filepath)




