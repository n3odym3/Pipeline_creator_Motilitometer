from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import dearpygui.dearpygui as dpg
from loguru import logger

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.file_explorer import file_explorer
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from .command_registry import command_registry


class SequenceWriterWin(WindowBase):
    """
    Sequence Writer window for composing, managing, and validating command sequences.

    Features:
        - Multi-line sequence text editor with automatic window resize scaling.
        - Drag-and-drop payload support for inserting saved sequence templates.
        - Built-in sequence template registry browser with recursive directory traversal.
        - Interactive command syntax help viewer with live regex and description lookup.
        - File export (.txt) and automatic sequence reloading via folder and group tracking.
    """

    label: str = "Sequence writer"
    win_width: int = display_scaling.scale(320)
    win_height: int = display_scaling.scale(380)
    description: str = (
        "Sequence Writer window for composing, managing, and validating command sequences."
    )

    accepted_input_types: list[str] = [
        IOTypes.TEXT,
        IOTypes.FOLDER_PATH,
        IOTypes.CMD_DICT,
    ]
    outputs: dict[str, str] = {
        "Sequence": IOTypes.TEXT,
    }
    persistent_fields: list[str] = ["label"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Sequence Writer window.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        if not self.outputs:
            self.outputs = {"Sequence": IOTypes.TEXT}
            if "Sequence" not in self.connections:
                self.connections["Sequence"] = []

        self.validate_seq_tag: str = f"seq_writer_validate_{self.UUID}"
        self.save_btn_tag: str = f"seq_writer_save_{self.UUID}"
        self.registry_btn_tag: str = f"seq_writer_registry_btn_{self.UUID}"
        self.help_btn_tag: str = f"seq_writer_help_btn_{self.UUID}"
        self.sequence_input_tag: str = f"seq_writer_input_{self.UUID}"
        self.handler_registry_tag: str = f"seq_writer_handler_reg_{self.UUID}"

        self.current_folder: Optional[str] = None
        self.raw_sequence: str = ""

        self._build_ui()
        self._setup_handlers()
        self.update_permission()

    def _build_ui(self) -> None:
        """Construct the DearPyGui user interface widgets."""
        scale = display_scaling.scale
        with self.window():
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Validate",
                    tag=self.validate_seq_tag,
                    callback=self._validate_sequence_cb,
                )
                dpg.add_button(
                    label="Seq registry",
                    tag=self.registry_btn_tag,
                    callback=self._show_registry_cb,
                )
                dpg.add_button(
                    label="Save",
                    tag=self.save_btn_tag,
                    callback=self._on_save_clicked,
                )
                dpg.add_button(
                    label="Help",
                    tag=self.help_btn_tag,
                    callback=self._show_help_cb,
                )

            dpg.add_input_text(
                tag=self.sequence_input_tag,
                width=-1,
                height=scale(280),
                multiline=True,
                on_enter=True,
                callback=self._validate_sequence_cb,
                payload_type="sequence",
                drop_callback=self._on_drop,
            )

    def _setup_handlers(self) -> None:
        """Register dynamic resize handler to automatically adapt input field height."""
        with dpg.item_handler_registry(tag=self.handler_registry_tag):
            dpg.add_item_resize_handler(callback=self._on_resize)
        dpg.bind_item_handler_registry(self.winID, self.handler_registry_tag)

    def _on_drop(self, sender: Any, app_data: Any, *args, **kwargs) -> None:
        """Append dragged sequence text into the editor."""
        current = dpg.get_value(self.sequence_input_tag) or ""
        sep = "\n" if current and not current.endswith("\n") else ""
        new_val = f"{current}{sep}{app_data}"
        dpg.set_value(sender, new_val)
        self.raw_sequence = new_val

    def _on_resize(self, *args: Any) -> None:
        """Dynamically scale the multi-line input box height based on window size."""
        if not dpg.does_item_exist(self.winID) or not dpg.does_item_exist(self.sequence_input_tag):
            return
        win_h = dpg.get_item_height(self.winID)
        if win_h and win_h > 0:
            scale = display_scaling.scale
            new_h = max(scale(40), win_h - scale(60))
            dpg.configure_item(self.sequence_input_tag, height=new_h)

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Disable closing in user mode while preserving control usability.
        """
        is_user = getattr(app_state, "mode", "dev") == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

        controls = [
            self.validate_seq_tag,
            self.save_btn_tag,
            self.registry_btn_tag,
            self.help_btn_tag,
            self.sequence_input_tag,
        ]
        for tag in controls:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=True)

    def is_ready(self) -> bool:
        """Check if the window is ready to process inputs."""
        return True

    def is_outputs_ready(self) -> bool:
        """Check if downstream connected modules are ready."""
        for connected_modules in self.connections.values():
            for module in connected_modules:
                is_ready = getattr(module, "is_ready", None)
                if callable(is_ready) and not is_ready():
                    return False
        return True

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming group names, tracking folder updates, or validation commands.
        """
        if data is None and "data" in kwargs:
            data = kwargs["data"]
        elif data is None and args:
            data = args[0]

        data_type = kwargs.get("data_type")
        data_content = data

        logger.info(f"[{self.label}] Received input: type='{data_type}'")

        if data_type == IOTypes.CMD_DICT or isinstance(data_content, dict):
            if isinstance(data_content, dict) and data_content.get("validate"):
                self._validate_sequence_cb(None, None)
                logger.info(f"[{self.label}] Validated sequence via command.")
            return

        if data_type == IOTypes.FOLDER_PATH or (isinstance(data_content, str) and os.path.isdir(data_content)):
            self.current_folder = data_content
            logger.info(f"[{self.label}] Updated tracking folder: {self.current_folder}")
            return

        if data_type == IOTypes.TEXT or isinstance(data_content, str):
            if self.current_folder and os.path.exists(self.current_folder):
                json_path = os.path.join(self.current_folder, f"{data_content}.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                            if "sequence" in payload:
                                seq_text = payload["sequence"]
                                if dpg.does_item_exist(self.sequence_input_tag):
                                    dpg.set_value(self.sequence_input_tag, seq_text)
                                self.raw_sequence = seq_text
                                logger.info(
                                    f"[{self.label}] Reloaded sequence for group '{data_content}' from {json_path}"
                                )
                                return
                    except Exception as e:
                        logger.error(f"[{self.label}] Failed to reload sequence from {json_path}: {e}")

            if data_content and ("\n" in data_content or "," in data_content):
                self.raw_sequence = data_content
                if dpg.does_item_exist(self.sequence_input_tag):
                    dpg.set_value(self.sequence_input_tag, self.raw_sequence)
            else:
                logger.warning(
                    f"[{self.label}] Received '{data_content}' but no corresponding JSON found in {self.current_folder}."
                )

    def trigger_cb(self, *args: Any, **kwargs: Any) -> None:
        """Send the current sequence text to downstream connected modules."""
        for output_key, connected_modules in self.connections.items():
            for module in list(connected_modules):
                try:
                    module.input_cb(self.raw_sequence, data_type=IOTypes.TEXT)
                except Exception as e:
                    logger.error(
                        f"[{self.label}] Error sending sequence to {getattr(module, 'label', 'module')} ({output_key}): {e}"
                    )

    def _validate_sequence_cb(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """Fetch current text from input widget and propagate downstream."""
        if dpg.does_item_exist(self.sequence_input_tag):
            self.raw_sequence = dpg.get_value(self.sequence_input_tag) or ""
        self.trigger_cb()

    def _on_save_clicked(self, *args: Any, **kwargs: Any) -> None:
        """Open the save file dialog targeting the sequences folder."""
        default_path = os.path.join(os.path.dirname(__file__), "sequences")
        os.makedirs(default_path, exist_ok=True)

        file_explorer.save_file(
            default_path=default_path,
            default_name="sequence.txt",
            extensions=[("Text files", "*.txt"), ("All files", "*.*")],
            callback=self._save_sequence_to_file,
        )

    def _save_sequence_to_file(
        self,
        file_path: str,
        ext: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Write sequence contents to disk."""
        if not file_path:
            return

        if ext == ".txt" and not file_path.lower().endswith(".txt"):
            file_path += ".txt"
        elif not file_path.lower().endswith(".txt") and not os.path.splitext(file_path)[1]:
            file_path += ".txt"

        try:
            seq_text = dpg.get_value(self.sequence_input_tag) if dpg.does_item_exist(self.sequence_input_tag) else self.raw_sequence
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(seq_text)
            logger.success(f"[{self.label}] Sequence saved to {file_path}")
        except Exception as e:
            logger.error(f"[{self.label}] Failed to save sequence to {file_path}: {e}")

    def _show_help_cb(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """Display reference window with all registered sequence commands."""
        help_win_tag = f"help_win_{self.UUID}"
        if dpg.does_item_exist(help_win_tag):
            dpg.show_item(help_win_tag)
            dpg.focus_item(help_win_tag)
            return

        scale = display_scaling.scale
        with dpg.window(
            label="Sequence Commands Help",
            width=scale(600),
            height=scale(400),
            pos=(self.pos[0] + scale(50), self.pos[1] + scale(50)),
            tag=help_win_tag,
        ):
            with dpg.table(
                header_row=True,
                resizable=True,
                borders_innerH=True,
                borders_innerV=True,
                borders_outerH=True,
                borders_outerV=True,
            ):
                dpg.add_table_column(label="Command Regex", width_stretch=True, init_width_or_weight=0.3)
                dpg.add_table_column(label="Description", width_stretch=True, init_width_or_weight=0.7)

                for regex, func, description in command_registry:
                    with dpg.table_row():
                        dpg.add_text(regex.pattern)
                        dpg.add_text(description)

    def _show_registry_cb(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """Display browser for exploring and dragging saved sequences."""
        registry_win_tag = f"seq_registry_win_{self.UUID}"
        if dpg.does_item_exist(registry_win_tag):
            dpg.show_item(registry_win_tag)
            dpg.focus_item(registry_win_tag)
            return

        base_dir = os.path.join(os.path.dirname(__file__), "sequences")
        os.makedirs(base_dir, exist_ok=True)

        scale = display_scaling.scale

        def build_ui_recursive(current_dir: str) -> None:
            try:
                items = os.listdir(current_dir)
            except OSError:
                return

            dirs = [d for d in items if os.path.isdir(os.path.join(current_dir, d))]
            files = [f for f in items if os.path.isfile(os.path.join(current_dir, f)) and f.endswith(".txt")]

            for sorted_dir in sorted(dirs):
                dir_path = os.path.join(current_dir, sorted_dir)
                has_txt = any(
                    fname.endswith(".txt")
                    for _, _, fnames in os.walk(dir_path)
                    for fname in fnames
                )
                if has_txt:
                    with dpg.collapsing_header(label=sorted_dir):
                        with dpg.group(indent=scale(20)):
                            build_ui_recursive(dir_path)

            if files:
                with dpg.group(indent=scale(20)):
                    for f in sorted(files):
                        file_path = os.path.join(current_dir, f)
                        button_name = os.path.splitext(f)[0]
                        try:
                            with open(file_path, "r", encoding="utf-8") as file_obj:
                                sequence_content = file_obj.read()
                        except Exception:
                            sequence_content = ""

                        rel_path = (
                            os.path.relpath(file_path, base_dir)
                            .replace("\\", "_")
                            .replace("/", "_")
                            .replace(".", "_")
                        )
                        btn_tag = f"seq_btn_{self.UUID}_{rel_path}"
                        dpg.add_button(label=button_name, width=-1, tag=btn_tag)

                        with dpg.drag_payload(
                            parent=btn_tag,
                            drag_data=sequence_content,
                            payload_type="sequence",
                        ):
                            dpg.add_text(f"Dragging: {button_name}")

                        with dpg.tooltip(btn_tag):
                            dpg.add_text(sequence_content)

        with dpg.window(
            label="Sequence Registry",
            width=scale(400),
            height=scale(400),
            pos=(self.pos[0] + scale(50), self.pos[1] + scale(50)),
            tag=registry_win_tag,
        ):
            build_ui_recursive(base_dir)

    def on_close(self) -> None:
        """Clean up DPG item handler registry and child windows."""
        if hasattr(self, "handler_registry_tag") and dpg.does_item_exist(self.handler_registry_tag):
            try:
                if dpg.does_item_exist(self.winID):
                    dpg.bind_item_handler_registry(self.winID, 0)
                dpg.delete_item(self.handler_registry_tag)
            except Exception:
                pass

        help_win = f"help_win_{self.UUID}"
        if dpg.does_item_exist(help_win):
            try:
                dpg.delete_item(help_win)
            except Exception:
                pass

        registry_win = f"seq_registry_win_{self.UUID}"
        if dpg.does_item_exist(registry_win):
            try:
                dpg.delete_item(registry_win)
            except Exception:
                pass

        super().on_close()


EXPORTED_CLASS = SequenceWriterWin
EXPORTED_NAME = "Sequence Writer"
