from __future__ import annotations

import datetime
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import dearpygui.dearpygui as dpg
from loguru import logger

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.input_output_types import IOTypes
from core.tutorial_manager import tutorial_manager
from core.window_base import WindowBase
from .sequence_processor import sequence_processor


class SequenceProcessorWin(WindowBase):
    """
    Sequence Processor window for running sequential commands and processes.

    Features:
        - Parses, validates, and visualizes command sequences in an interactive table.
        - Calculates total execution duration and tracks live remaining time and progress.
        - Manages synchronization signals with downstream capture/hardware modules.
        - Emits multi-channel command dictionaries and triggers.
        - Archives raw executed sequences with session metadata as JSON in the target dataset folder.
    """

    label: str = "Sequence processor"
    win_width: int = display_scaling.scale(320)
    win_height: int = display_scaling.scale(240)
    description: str = "Sequence Processor window for running sequential commands and processes."

    accepted_input_types: list[str] = [
        IOTypes.CMD_LIST,
        IOTypes.FOLDER_PATH,
        IOTypes.TEXT,
        IOTypes.CMD_DICT,
        IOTypes.TRIGGER,
    ]
    outputs: dict[str, str] = {
        "CMD dict": IOTypes.CMD_DICT,
        "CMD dict 2": IOTypes.CMD_DICT,
        "Trigger": IOTypes.TRIGGER,
    }
    persistent_fields: list[str] = ["label", "end_seq_cmd"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Sequence Processor window.

        Args:
            **kwargs: Additional keyword arguments forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        # Ensure compatibility default outputs exist
        defaults = {
            "CMD dict": IOTypes.CMD_DICT,
            "CMD dict 2": IOTypes.CMD_DICT,
            "Trigger": IOTypes.TRIGGER,
        }
        for k, v in defaults.items():
            if k not in self.outputs:
                self.outputs[k] = v
                if k not in self.connections:
                    self.connections[k] = []

        self.sequence_input_tag: str = f"seq_writer_input_{self.UUID}"
        self.table_tag: str = f"seq_writer_table_{self.UUID}"
        self.run_seq_tag: str = f"seq_run_{self.UUID}"
        self.kill_seq_tag: str = f"seq_kill_{self.UUID}"
        self.seq_remaining_tag: str = f"seq_remaining_{self.UUID}"
        self.seq_progress_bar_tag: str = f"seq_progress_bar_{self.UUID}"
        self.table_group_tag: str = f"seq_table_group_{self.UUID}"

        self.task: Optional[threading.Thread] = None
        self.cmd_list: list[str] = []
        self.total_duration: int | float = 0
        self.initial_total_duration: int | float = 0
        self.seq_running: bool = False
        self.seq_error: bool = False
        self._ready_event: threading.Event = threading.Event()
        self.check_delay: int = 5
        self.current_folder: Optional[str] = None
        self.raw_sequence: str = ""
        self.last_sequence: list[str] = []

        end_cmd_val = getattr(self, "end_seq_cmd", None)
        if end_cmd_val is None:
            self.end_seq_cmd = getattr(sequence_processor, "end_seq_cmd", "R0G0B0W0IR1000")
        else:
            self.end_seq_cmd = end_cmd_val

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """Construct DearPyGui widgets for sequence execution and feedback."""
        scale = display_scaling.scale
        with self.window():
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Run seq.",
                    tag=self.run_seq_tag,
                    callback=self.run_sequence,
                    width=scale(100),
                )
                dpg.add_button(
                    label="KILL seq.",
                    tag=self.kill_seq_tag,
                    callback=self.kill_sequence,
                    width=scale(100),
                )

            dpg.add_text("Remaining : 00:00:00", tag=self.seq_remaining_tag)
            dpg.add_progress_bar(tag=self.seq_progress_bar_tag, default_value=0.0, width=-1)

            with dpg.group(tag=self.table_group_tag):
                pass

    def update_permission(self) -> None:
        """
        Adapt UI based on app_state.mode.
        Window remains closable even in user mode, and Run sequence button remains available.
        """
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=False)

        # Keep Run button visible and enabled
        if dpg.does_item_exist(self.run_seq_tag):
            dpg.show_item(self.run_seq_tag)
            dpg.configure_item(self.run_seq_tag, enabled=True)

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

    def validate_sequence(self, cmd_list: list[str]) -> None:
        """
        Validate group declarations and prevent folder collisions before execution.

        Args:
            cmd_list: Ordered list of command strings.
        """
        self.seq_error = False

        g_commands = set()
        for cmd in cmd_list:
            clean_cmd = cmd
            if cmd.lower().startswith("out2 "):
                clean_cmd = cmd[5:].strip()
            match = re.match(r"(?i)^g(.+)", clean_cmd)
            if match:
                g_commands.add(match.group(1).strip())

        if not g_commands:
            logger.error(f"[{self.label}] No group defined (G command missing). Sequence must have at least one group.")
            self.seq_error = True

        conflict_groups = []
        if self.current_folder and os.path.exists(self.current_folder) and g_commands:
            existing_files = os.listdir(self.current_folder)
            for g_cmd in g_commands:
                for f in existing_files:
                    if f"_{g_cmd}_" in f or f.startswith(f"{g_cmd}_") or f"G{g_cmd}" in f:
                        conflict_groups.append(g_cmd)
                        break

        self._finalize_validation(cmd_list, conflict_groups, missing_group=not g_commands)

    def _finalize_validation(
        self,
        cmd_list: list[str],
        conflict_groups: list[str] | None = None,
        missing_group: bool = False,
        show_window: bool = True,
    ) -> None:
        """
        Reconstruct DearPyGui table displaying commands, translations, and durations.
        """
        if conflict_groups is None:
            conflict_groups = []

        if show_window and dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, show=True)
            dpg.focus_item(self.winID)

        if not dpg.does_item_exist(self.table_group_tag):
            return

        dpg.push_container_stack(self.table_group_tag)
        try:
            if dpg.does_item_exist(self.table_tag):
                dpg.delete_item(self.table_tag)

            # Generate a unique tag to avoid DearPyGUI internal state persistence
            self.table_tag = f"seq_writer_table_{self.UUID}_{uuid.uuid4().hex[:8]}"

            with dpg.table(
                header_row=True,
                resizable=True,
                tag=self.table_tag,
                borders_innerH=True,
                borders_innerV=True,
                borders_outerH=True,
                borders_outerV=True,
                policy=dpg.mvTable_SizingStretchProp,
            ):
                dpg.add_table_column(label="Cmd", width_stretch=True, init_width_or_weight=0.3)
                dpg.add_table_column(label="Descr", width_stretch=True, init_width_or_weight=0.7)
                self.total_duration = 0
                self.last_sequence = []

                if missing_group:
                    with dpg.table_row():
                        dpg.add_text("ERROR")
                        dpg.add_text("Missing Group (G command)")
                    dpg.highlight_table_row(self.table_tag, 0, [255, 0, 0, 50])
                    row_offset = 1
                else:
                    row_offset = 0

                for i, cmd in enumerate(cmd_list):
                    with dpg.table_row():
                        dpg.add_text(cmd)
                        cmd_result = sequence_processor.trad_cmd(cmd)

                        if not cmd_result["valid"]:
                            dpg.highlight_table_row(self.table_tag, i + row_offset, [255, 0, 0, 50])
                            self.seq_error = True
                            logger.error(f"[{self.label}] Invalid command at index {i}! Please correct: {cmd}")
                        else:
                            clean_cmd = cmd
                            if cmd.lower().startswith("out2 "):
                                clean_cmd = cmd[5:].strip()
                            match = re.match(r"(?i)^g(.+)", clean_cmd)
                            if match:
                                group_name = match.group(1).strip()
                                if group_name in conflict_groups:
                                    dpg.highlight_table_row(self.table_tag, i + row_offset, [255, 0, 0, 50])
                                    dpg.add_text("ERROR: Group already exists")
                                    self.seq_error = True
                                    logger.error(
                                        f"[{self.label}] Group '{group_name}' already exists in folder {self.current_folder}"
                                    )
                                    continue

                        self.total_duration += cmd_result["duration"]
                        dpg.add_text(str(cmd_result["description"]))

                if dpg.does_item_exist(self.seq_remaining_tag):
                    dpg.set_value(
                        self.seq_remaining_tag,
                        f"Remaining : {datetime.timedelta(seconds=int(self.total_duration))}",
                    )
                self.initial_total_duration = self.total_duration
                if dpg.does_item_exist(self.seq_progress_bar_tag):
                    dpg.set_value(self.seq_progress_bar_tag, 0.0)

                if not self.seq_error:
                    logger.info(
                        f"[{self.label}] Sequence validated with {len(cmd_list)} commands, total duration: {self.total_duration} seconds"
                    )
                    self.last_sequence = cmd_list
        finally:
            dpg.pop_container_stack()

    def run_sequence(self, *args: Any, **kwargs: Any) -> None:
        """
        Validate and execute the loaded command sequence in a background worker thread.
        """
        if self.seq_running:
            tutorial_manager.play_center_hint(
                "A sequence is already running. Please cancel the current sequence before starting a new one.",
                type="error_hint",
                icon_type="warning",
            )
            logger.warning(
                f"[{self.label}] A sequence is already running, please wait for it to finish or kill it before starting a new one."
            )
            return

        if hasattr(self, "last_sequence") and self.last_sequence:
            self.validate_sequence(self.last_sequence)

        if not self.seq_error:

            def task(final_group: str | None = None, manual: bool = False) -> None:
                def cleanup(status: str) -> None:
                    self.trigger_cb(
                        "CMD dict",
                        {"record": False, "save": False, "add_timestamp": False, "manual": False, "export": True},
                    )

                    # Send end of sequence command if configured
                    end_cmd = getattr(self, "end_seq_cmd", None)
                    if end_cmd is None:
                        end_cmd = getattr(sequence_processor, "end_seq_cmd", "R0G0B0W0IR1000")

                    if end_cmd:
                        logger.info(f"[{self.label}] Sending end_seq_cmd: {end_cmd}")
                        if isinstance(end_cmd, list):
                            for item in end_cmd:
                                item_str = str(item).strip()
                                if not item_str:
                                    continue
                                target = "CMD dict 2" if item_str.lower().startswith("out2 ") else "CMD dict"
                                raw_cmd = item_str[5:].strip() if item_str.lower().startswith("out2 ") else item_str
                                cmd_with_newline = raw_cmd if raw_cmd.endswith("\n") else f"{raw_cmd}\n"
                                self.trigger_cb(target, cmd_with_newline)
                                time.sleep(0.02)
                        elif isinstance(end_cmd, str):
                            for c in sequence_processor.get_commands(end_cmd):
                                if not c.strip():
                                    continue
                                parsed = sequence_processor.trad_cmd(c)
                                if parsed.get("valid") and parsed.get("cmd"):
                                    target = "CMD dict 2" if parsed.get("out2") else "CMD dict"
                                    self.trigger_cb(target, parsed["cmd"])
                                else:
                                    target = "CMD dict 2" if c.lower().startswith("out2 ") else "CMD dict"
                                    raw_c = c[5:].strip() if c.lower().startswith("out2 ") else c
                                    cmd_with_newline = raw_c if raw_c.endswith("\n") else f"{raw_c}\n"
                                    self.trigger_cb(target, cmd_with_newline)
                                time.sleep(0.02)

                    if dpg.does_item_exist(self.seq_remaining_tag):
                        dpg.set_value(self.seq_remaining_tag, f"Remaining : {status}")
                    if status == "DONE" and dpg.does_item_exist(self.seq_progress_bar_tag):
                        dpg.set_value(self.seq_progress_bar_tag, 1.0)
                    self.trigger_cb("Trigger", "stop")
                    self.seq_running = False

                    if status in ("ABORTED", "KILLED"):
                        self.trigger_cb("CMD dict", {"stop_video": True, "stop": True})

                    if status == "DONE" and not manual:
                        reload_cmd = {"scan": True}
                        if final_group:
                            reload_cmd["group"] = final_group
                        self.trigger_cb("CMD dict", reload_cmd)

                for i, cmd in enumerate(self.last_sequence):
                    if not self.seq_running:
                        logger.info(f"[{self.label}] Sequence aborted")
                        cleanup("ABORTED")
                        return

                    if dpg.does_item_exist(self.table_tag):
                        dpg.highlight_table_row(self.table_tag, i, [0, 0, 255, 50])

                    cmd_response = sequence_processor.trad_cmd(cmd)

                    wait_feedback = bool(cmd_response.get("wait_feedback", False))
                    duration = float(cmd_response.get("duration", 0.0))

                    if wait_feedback:
                        self._ready_event.clear()

                    if cmd_response.get("out2"):
                        self.trigger_cb("CMD dict 2", cmd_response["cmd"])
                    else:
                        self.trigger_cb("CMD dict", cmd_response["cmd"])

                    if wait_feedback:
                        start_wait = time.perf_counter()
                        # Timeout: wait until ready event OR auto-advance if duration + 2.0s expires (or timeout if duration is 0)
                        max_wait = duration + 2.0 if duration > 0 else float(cmd_response.get("timeout", 10.0))

                        while not self._ready_event.is_set():
                            if not self.seq_running:
                                logger.info(f"[{self.label}] Sequence aborted")
                                cleanup("KILLED")
                                return

                            time.sleep(0.01)
                            now = time.perf_counter()
                            elapsed = now - start_wait
                            if elapsed >= max_wait:
                                logger.info(
                                    f"[{self.label}] Feedback wait elapsed for command '{cmd}' ({elapsed:.1f}s), auto-advancing."
                                )
                                break

                            # Live update remaining time display while waiting (only if duration > 0)
                            if duration > 0:
                                time_spent = min(elapsed, duration)
                                rem = max(0.0, self.total_duration - time_spent)
                                if dpg.does_item_exist(self.seq_remaining_tag):
                                    dpg.set_value(
                                        self.seq_remaining_tag,
                                        f"Remaining : {datetime.timedelta(seconds=int(rem))}",
                                    )
                                if self.initial_total_duration > 0 and dpg.does_item_exist(self.seq_progress_bar_tag):
                                    progress = max(0.0, min(1.0, 1.0 - (rem / self.initial_total_duration)))
                                    dpg.set_value(self.seq_progress_bar_tag, progress)

                        if duration > 0:
                            self.total_duration -= duration
                            if self.total_duration < 0:
                                self.total_duration = 0

                        if dpg.does_item_exist(self.seq_remaining_tag):
                            dpg.set_value(
                                self.seq_remaining_tag,
                                f"Remaining : {datetime.timedelta(seconds=int(self.total_duration))}",
                            )
                        if self.initial_total_duration > 0 and dpg.does_item_exist(self.seq_progress_bar_tag):
                            progress = max(0.0, min(1.0, 1.0 - (self.total_duration / self.initial_total_duration)))
                            dpg.set_value(self.seq_progress_bar_tag, progress)

                    elif duration > 0:
                        remaining_time = duration
                        while remaining_time > 0:
                            sleep_time = min(1, remaining_time)
                            time.sleep(sleep_time)
                            remaining_time -= sleep_time

                            self.total_duration -= sleep_time
                            if self.total_duration < 0:
                                self.total_duration = 0

                            if dpg.does_item_exist(self.seq_remaining_tag):
                                dpg.set_value(
                                    self.seq_remaining_tag,
                                    f"Remaining : {datetime.timedelta(seconds=int(self.total_duration))}",
                                )
                            if self.initial_total_duration > 0 and dpg.does_item_exist(self.seq_progress_bar_tag):
                                progress = 1.0 - (self.total_duration / self.initial_total_duration)
                                dpg.set_value(self.seq_progress_bar_tag, progress)

                            if not self.seq_running:
                                logger.info(f"[{self.label}] Sequence aborted")
                                cleanup("KILLED")
                                return

                    if dpg.does_item_exist(self.table_tag):
                        dpg.highlight_table_row(self.table_tag, i, [0, 255, 0, 50])

                cleanup("DONE")

            if not self.seq_running:
                # Pre-scan sequence for Triggers and Groups using command registry definitions
                current_group = None
                current_A = 0
                names_to_send = []
                has_captures = False
                is_manual = False

                for cmd_str in self.last_sequence:
                    clean_cmd = cmd_str
                    if cmd_str.lower().startswith("out2 "):
                        clean_cmd = cmd_str[5:].strip()

                    g_match = re.match(r"(?i)^g(.+)", clean_cmd)
                    if g_match:
                        current_group = g_match.group(1).strip()

                    # Track A command
                    a_match = re.fullmatch(r"[aA](\d+)", clean_cmd)
                    if a_match:
                        current_A = int(a_match.group(1))

                    # Get translated command
                    cmd_result = sequence_processor.trad_cmd(cmd_str)
                    captures = cmd_result.get("captures", [])

                    if cmd_result.get("cmd", {}).get("manual", False):
                        is_manual = True

                    if captures:
                        has_captures = True
                        for cap_name in captures:
                            if current_group:
                                names_to_send.append(f"{len(names_to_send)}_{current_group}_A{current_A}_{cap_name}")
                            else:
                                names_to_send.append(f"{len(names_to_send)}_A{current_A}_{cap_name}")

                setup_cmd: dict[str, Any] = {}
                if has_captures:
                    setup_cmd["trigger_mode"] = "Trigger"

                if names_to_send:
                    setup_cmd["names"] = names_to_send
                    setup_cmd["record"] = True

                if not is_manual:
                    setup_cmd["record"] = True
                    setup_cmd["save"] = True
                    setup_cmd["add_timestamp"] = True

                setup_cmd["reset_ts"] = True
                setup_cmd["clear"] = True

                if current_group:
                    setup_cmd["group"] = current_group

                setup_cmd["manual"] = is_manual

                self.trigger_cb("CMD dict", setup_cmd)
                time.sleep(0.1)

                # Save sequence JSON
                logger.info(
                    f"[{self.label}] Attempting to save sequence JSON. current_group='{current_group}', current_folder='{self.current_folder}'"
                )
                if current_group and self.current_folder:
                    json_path = os.path.join(self.current_folder, f"{current_group}.json")
                    try:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "sequence": self.raw_sequence,
                                    "user": app_state.get("username", ""),
                                    "detector": app_state.get("detector", None),
                                },
                                f,
                                indent=4,
                            )
                        logger.info(f"[{self.label}] Sequence saved to {json_path}")
                    except Exception as e:
                        logger.error(f"[{self.label}] Failed to save sequence JSON: {e}")
                else:
                    logger.warning(
                        f"[{self.label}] Cannot save JSON because current_group or current_folder is missing (current_group={current_group}, current_folder={self.current_folder})"
                    )

                self.set_table_color((0, 0, 0, 0))
                self.trigger_cb("Trigger", "start")
                self.seq_running = True
                self.task = threading.Thread(
                    target=task,
                    args=(current_group, is_manual),
                    daemon=True,
                    name=f"SeqProcWorker_{self.UUID}",
                )
                self.task.start()
            else:
                tutorial_manager.play_center_hint(
                    "A sequence is already running. Please cancel the current sequence before starting a new one.",
                    type="error_hint",
                    icon_type="warning",
                )
                logger.warning(
                    f"[{self.label}] A sequence is already running, please wait for it to finish or kill it before starting a new one."
                )
        else:
            tutorial_manager.play_center_hint(
                "Sequence validation failed. Please check for errors (missing group or group already exists) before running.",
                type="error_hint",
                icon_type="error",
            )
            logger.error(f"[{self.label}] Cannot run a sequence that contains errors. Please validate the sequence first.")
            if dpg.does_item_exist(self.seq_remaining_tag):
                dpg.set_value(self.seq_remaining_tag, "Remaining : ERROR")

    def set_table_color(self, rgba: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
        """Set the background highlight color of all rows in the sequence table."""
        if hasattr(self, "table_tag") and self.table_tag and dpg.does_item_exist(self.table_tag):
            try:
                rows = dpg.get_item_children(self.table_tag, 1)
                if rows:
                    for i in range(len(rows)):
                        dpg.highlight_table_row(self.table_tag, i, rgba)
            except Exception as e:
                logger.error(f"[{self.label}] Failed to set table color: {e}")

    def kill_sequence(self, *args: Any, **kwargs: Any) -> None:
        """Abort the currently executing sequence immediately."""
        self.seq_running = False
        self._ready_event.set()
        self.set_table_color((255, 0, 0, 50))

    def clear_sequence(self, *args: Any, **kwargs: Any) -> None:
        """Clear the sequence processor state."""
        logger.info(f"[{self.label}] clear_sequence called (current_folder before clear: {self.current_folder})")
        self.seq_running = False
        self.seq_error = False
        self._ready_event.set()
        self.raw_sequence = ""
        self.last_sequence = []
        self.cmd_list = []
        self.task = None
        self.total_duration = 0
        self.initial_total_duration = 0
        self._finalize_validation([], show_window=False)
        logger.info(f"[{self.label}] Sequence processor cleared")

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Process incoming sequence definitions, folder paths, commands, or feedback triggers.
        """
        if data is None and "data" in kwargs:
            data = kwargs["data"]
        elif data is None and args:
            data = args[0]

        data_type = kwargs.get("data_type")
        data_content = data

        if data_type == IOTypes.FOLDER_PATH or (isinstance(data_content, str) and os.path.isdir(data_content)):
            self.current_folder = data_content
            logger.info(f"[{self.label}] Set current_folder: {self.current_folder}")
            return

        # 1. Handle READY / Trigger feedback signals from downstream modules
        trigger_signals = {"READY", "RECORD_START", "RECORD_STARTED", "RECORD_STOP", "DONE", "STOP", "FINISHED"}
        is_feedback = False
        if data_type == IOTypes.TRIGGER:
            if isinstance(data_content, str):
                if data_content.upper() in trigger_signals:
                    is_feedback = True
            else:
                is_feedback = True
        elif isinstance(data_content, str) and data_content.upper() in trigger_signals:
            is_feedback = True

        if is_feedback:
            logger.info(f"[{self.label}] Received feedback trigger: '{data_content}'")
            self._ready_event.set()
            return

        if data_type == IOTypes.CMD_DICT or isinstance(data_content, dict):
            if isinstance(data_content, dict):
                feedback_keys = ("ready", "done", "finished", "record_started", "record_start")
                if any(data_content.get(k) for k in feedback_keys):
                    logger.info(f"[{self.label}] Received feedback dict: {data_content}")
                    self._ready_event.set()
                    return
                if data_content.get("start_sequence"):
                    self.run_sequence()
                if data_content.get("clear", False):
                    self.clear_sequence()
            return

        # Block setting sequence if a sequence is already running
        if self.seq_running:
            if data_type == IOTypes.TEXT or isinstance(data_content, (str, list)):
                tutorial_manager.play_center_hint(
                    "A sequence is already running. Please cancel the current sequence before setting a new one.",
                    type="error_hint",
                    icon_type="warning",
                )
                logger.warning(f"[{self.label}] A sequence is already running, ignoring incoming sequence update.")
                return

        if data_type == IOTypes.TEXT or isinstance(data_content, str):
            self.raw_sequence = data_content
            cmd_list = sequence_processor.get_commands(self.raw_sequence)
            self.validate_sequence(cmd_list)
            return

        if isinstance(data_content, list):
            # Backward compatibility or direct list input
            self.raw_sequence = "\n".join(str(c) for c in data_content)
            self.validate_sequence(data_content)

    def trigger_cb(self, target: str | int = "CMD dict", cmd: dict[str, Any] | str | None = None) -> None:
        """
        Send command or trigger to downstream connected modules.

        Args:
            target: Target output pin name or index.
            cmd: Command payload (dict or string).
        """
        if cmd is None:
            cmd = {}

        if isinstance(target, int):
            if target == 0:
                output_key = "CMD dict"
            elif target == 1:
                output_key = "Trigger"
            else:
                output_key = "CMD dict 2"
        else:
            output_key = str(target)

        connected_modules = list(self.connections.get(output_key, []))
        for module in connected_modules:
            try:
                if output_key == "Trigger":
                    module.input_cb(cmd, data_type=IOTypes.TRIGGER)
                else:
                    module.input_cb(cmd)
            except Exception as e:
                logger.error(
                    f"[{self.label}] Error sending to {getattr(module, 'label', 'module')} ({output_key}): {e}"
                )

    def on_close(self) -> None:
        """Clean up background tasks before window closes."""
        self.kill_sequence()
        if self.task and self.task.is_alive() and threading.current_thread() != self.task:
            self.task.join(timeout=0.5)
        super().on_close()

    def close(self) -> None:
        """Clean up background tasks and unregister module."""
        self.kill_sequence()
        if self.task and self.task.is_alive() and threading.current_thread() != self.task:
            self.task.join(timeout=0.5)
        super().close()


EXPORTED_CLASS = SequenceProcessorWin
EXPORTED_NAME = "Sequence processor"
