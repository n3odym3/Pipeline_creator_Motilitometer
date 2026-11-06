from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.file_explorer import file_explorer
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from core.app_state import app_state
from .video_tools import Video_tools


class VideoReader_win(WindowBase):
    """
    Hierarchical Video Reader module window for the Motilitometer pipeline.

    Scans directories for video files (recursively or flat), provides interactive
    tree navigation, automated batch processing queues with color-coding
    (blue = processing, green = processed), media transport controls
    (Play, Pause, Stop, Seek Scrubber, FPS adjustment), and streams frames,
    triggers, and file paths downstream.
    """

    label = "Video Reader"
    description = "Hierarchical video directory browser, batch queue runner, and synchronized playback streamer."
    win_width = display_scaling.scale(350)
    win_height = display_scaling.scale(520)

    accepted_input_types = [
        IOTypes.FOLDER_PATH,
        IOTypes.FILE_PATH,
        IOTypes.CMD_DICT,
        IOTypes.TRIGGER,
    ]
    outputs = {
        "Frame": IOTypes.FRAME,
        "Trigger": IOTypes.TRIGGER,
        "File Path": IOTypes.FILE_PATH,
        "Playback": IOTypes.PLAYBACK,
    }
    persistent_fields = ["label", "recursive", "reprocess", "file_format", "folder"]

    recursive: bool = True
    reprocess: bool = True
    file_format: str = "All"
    folder: str = ""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        def _p(attr: str, default: Any) -> Any:
            val = getattr(self, attr, None)
            if val is None:
                val = kwargs.get(attr, default)
            return default if val is None else val

        # Defensive extraction of persistent fields
        self.recursive = bool(_p("recursive", True))
        self.reprocess = bool(_p("reprocess", True))
        self.file_format = str(_p("file_format", "All") or "All")
        folder_init = _p("folder", "") or kwargs.get("folder_path", "")
        self.folder = str(folder_init or "")

        # DPG UI Tags
        self.winID = f"video_reader_win_{self.UUID}"
        self.tree_group_tag = f"video_reader_tree_group_{self.UUID}"
        self.child_win_tree_tag = f"video_reader_child_tree_{self.UUID}"
        self.fps_tag = f"video_reader_fps_{self.UUID}"
        self.start_tag = f"video_reader_start_{self.UUID}"
        self.stop_tag = f"video_reader_stop_{self.UUID}"
        self.pause_tag = f"video_reader_pause_{self.UUID}"
        self.progress_tag = f"video_reader_progress_{self.UUID}"
        self.folder_label_tag = f"video_reader_folder_label_{self.UUID}"
        self.recursive_tag = f"video_reader_recursive_{self.UUID}"
        self.format_tag = f"video_reader_format_{self.UUID}"

        # Process All UI Tags
        self.process_all_tag = f"video_reader_process_all_{self.UUID}"
        self.stop_process_all_tag = f"video_reader_stop_process_{self.UUID}"
        self.reprocess_tag = f"video_reader_reprocess_{self.UUID}"
        self.process_status_tag = f"video_reader_process_status_{self.UUID}"

        self.last_selectable: Optional[Union[int, str]] = None
        self.last_video_selected: Optional[str] = None
        self.enabled: bool = False

        # Video playback state
        self.video_capturer = Video_tools()
        self.video_running: bool = False
        self.video_paused: bool = False
        self.video_thread: Optional[threading.Thread] = None
        self._seek_request: Optional[int] = None
        self._ignore_slider_cb: bool = False
        self._current_video_name: str = ""
        self._total_frames: int = 0
        self._is_closing: bool = False

        self.playback_fps: int = 30
        self.frame_interval: float = 1.0 / self.playback_fps

        # Folder tree and video listings
        self.folder_tree: Optional[Dict[str, Any]] = None
        self.folder_map: Dict[str, Dict[str, Any]] = {}
        self.video_list: List[Dict[str, Any]] = []
        self.video_ui_map: Dict[Union[int, str], Dict[str, Any]] = {}
        self.video_path_to_tag: Dict[str, Union[int, str]] = {}
        self._ui_tag_counter: int = 0

        # Process All / Queue state
        self.process_lock = threading.Lock()
        self.process_queue: List[Dict[str, Any]] = []
        self.process_queue_index: int = 0
        self.process_all_active: bool = False

        # Load local config.json (video_extensions + regex/color status_rules)
        self._config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.video_extensions: List[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
        self.status_rules: List[Dict[str, Any]] = []
        self._color_themes: Dict[Tuple[int, int, int, int], Union[int, str]] = {}
        self._load_config()

        # Themes for video processing state color-coding
        with dpg.theme() as self.theme_processing_blue:
            with dpg.theme_component(dpg.mvSelectable):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 210, 90, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (40, 80, 150, 255))

        with dpg.theme() as self.theme_processed_green:
            with dpg.theme_component(dpg.mvSelectable):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 255, 100, 255))

        # Build Interface
        scale = display_scaling.scale
        combo_items = ["All"] + list(self.video_extensions)
        if self.file_format not in combo_items:
            self.file_format = "All"

        with self.window():
            # Folder browser row
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Browse",
                    width=scale(100),
                    callback=self._on_browse_clicked,
                )
                dpg.add_combo(
                    items=combo_items,
                    default_value=self.file_format,
                    tag=self.format_tag,
                    width=scale(85),
                    callback=self._on_format_changed,
                )
                dpg.add_checkbox(
                    label="Recursive",
                    default_value=self.recursive,
                    callback=self._on_recursive_toggled,
                    tag=self.recursive_tag,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("Folder:", color=(150, 150, 150))
                init_display = self._format_folder_display(self.folder) if self.folder else "No folder selected"
                dpg.add_text(init_display, tag=self.folder_label_tag)

            dpg.add_separator()

            # Process All Controls
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Process ALL",
                    width=scale(150),
                    callback=self._on_process_all_clicked,
                    tag=self.process_all_tag,
                )
                dpg.add_button(
                    label="Stop Process",
                    width=scale(150),
                    callback=self._on_stop_process_clicked,
                    tag=self.stop_process_all_tag,
                    enabled=False,
                )
                dpg.add_checkbox(
                    label="Reprocess",
                    default_value=self.reprocess,
                    tag=self.reprocess_tag,
                    callback=self._on_reprocess_toggled,
                )

            dpg.add_text("Ready", tag=self.process_status_tag, color=(100, 200, 100))

            dpg.add_separator()

            # Playback controls row
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="\uf04b",
                    tag=self.start_tag,
                    width=scale(50),
                    callback=self.start_video,
                )
                dpg.add_button(
                    label="\uf04c",
                    tag=self.pause_tag,
                    width=scale(50),
                    callback=self.toggle_pause,
                )
                dpg.add_button(
                    label="\uf04d",
                    tag=self.stop_tag,
                    width=scale(50),
                    callback=self.stop_video,
                )
                dpg.add_text("FPS: ")
                dpg.add_input_int(
                    tag=self.fps_tag,
                    default_value=self.playback_fps,
                    min_value=1,
                    min_clamped=True,
                    step=-1,
                    width=-1,
                    callback=self.set_playback_fps,
                )

            with dpg.group(horizontal=True):
                dpg.add_slider_int(
                    tag=self.progress_tag,
                    width=-1,
                    default_value=0,
                    callback=self.on_slider_change,
                )

            dpg.add_separator()

            # Hierarchical Folder Tree container
            with dpg.group(tag=self.tree_group_tag):
                with dpg.child_window(tag=self.child_win_tree_tag):
                    dpg.add_text("Browse a folder to view videos.", color=(150, 150, 150))

        # Apply icon font for FontAwesome glyphs
        try:
            from config.theme_manager import theme_manager

            theme_manager.apply_icon_font(self.start_tag)
            theme_manager.apply_icon_font(self.pause_tag)
            theme_manager.apply_icon_font(self.stop_tag)
        except Exception as e:
            logger.debug(f"Could not apply icon font in {self.winID}: {e}")

        self.update_permission()

        if self.folder and os.path.isdir(self.folder):
            self.update_file_list(self.folder)

    def update_permission(self) -> None:
        """Update UI permissions based on application mode."""
        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def is_ready(self) -> bool:
        """Indicate whether the video reader module is ready."""
        return not self._is_closing

    def _format_folder_display(self, path: str, max_chars: int = 30) -> str:
        """Format folder path for display in header label with truncation."""
        if not path:
            return "No folder selected"
        return path if len(path) <= max_chars else f"...{path[-(max_chars - 3):]}"

    def set_folder(self, folder_path: str) -> bool:
        """
        Set active working folder, scan for video files, and update UI tree.

        Args:
            folder_path: Absolute or relative directory path.

        Returns:
            True if folder exists or was created and loaded successfully.
        """
        if not folder_path:
            return False
        folder_str = str(folder_path).strip("'\"")
        if not os.path.exists(folder_str):
            try:
                os.makedirs(folder_str, exist_ok=True)
            except Exception as e:
                logger.warning(f"VideoReader: Failed to create folder '{folder_str}': {e}")

        if os.path.isdir(folder_str):
            self.folder = os.path.abspath(folder_str)
            self.enabled = True
            self.video_running = False
            if dpg.does_item_exist(self.folder_label_tag):
                dpg.set_value(self.folder_label_tag, self._format_folder_display(self.folder))
            self.update_file_list(self.folder)
            logger.info(f"VideoReader working folder set to: {self.folder}")
            return True
        return False

    def _next_id(self) -> int:
        """Generate unique incremental identifier for dynamic DPG tags."""
        self._ui_tag_counter += 1
        return self._ui_tag_counter

    def _on_browse_clicked(self, *args: Any, **kwargs: Any) -> None:
        """Open native OS directory picker dialog."""
        file_explorer.select_folder(
            default_path=self.folder if self.folder else "",
            callback=self._on_folder_selected,
        )

    def _on_folder_selected(self, selected_dir: str, *args: Any, **kwargs: Any) -> None:
        """Process folder selected from OS dialog."""
        if selected_dir:
            self.set_folder(selected_dir)

    def _on_recursive_toggled(
        self,
        sender: Any,
        app_data: bool,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Toggle recursive directory search mode."""
        self.recursive = app_data
        if self.folder:
            self.update_file_list(self.folder)

    def _on_reprocess_toggled(
        self,
        sender: Any,
        app_data: bool,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Toggle reprocess option for already completed videos."""
        self.reprocess = app_data

    def _on_format_changed(
        self,
        sender: Any,
        app_data: str,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Update file extension filter and reload directory tree."""
        self.file_format = app_data
        if self.folder:
            self.update_file_list(self.folder)

    def _load_config(self) -> None:
        """Load `video_extensions` and `status_rules` from local `config.json`."""
        default_exts = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
        default_rules = [
            {
                "name": "processed",
                "regex": "^[filename]\\.processed$",
                "color": [100, 255, 100, 255],
                "skip_unless_reprocess": True,
            },
            {
                "name": "drift",
                "regex": "^[filename]_drift\\.csv$",
                "color": [100, 180, 255, 255],
                "skip_unless_reprocess": False,
            },
        ]
        if not os.path.isfile(self._config_path):
            self.video_extensions = list(default_exts)
            self.status_rules = list(default_rules)
            self._prepare_status_rules()
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            raw_exts = cfg.get("video_extensions", default_exts)
            if isinstance(raw_exts, list) and raw_exts:
                cleaned_exts: List[str] = []
                for ext in raw_exts:
                    if isinstance(ext, str) and ext.strip():
                        e = ext.strip().lower()
                        if not e.startswith("."):
                            e = "." + e
                        if e not in cleaned_exts:
                            cleaned_exts.append(e)
                self.video_extensions = cleaned_exts or list(default_exts)
            else:
                self.video_extensions = list(default_exts)

            raw_rules = cfg.get("status_rules", default_rules)
            if isinstance(raw_rules, list):
                self.status_rules = [dict(r) for r in raw_rules if isinstance(r, dict)]
            else:
                self.status_rules = list(default_rules)
        except Exception as e:
            logger.warning(f"VideoReader failed to load {self._config_path}: {e}")
            self.video_extensions = list(default_exts)
            self.status_rules = list(default_rules)
        self._prepare_status_rules()

    def _prepare_status_rules(self) -> None:
        """Pre-compile static regexes and placeholder flags on `self.status_rules`."""
        placeholder_tokens = ("[filename]", "{filename}", "[stem]", "{stem}", "[name]", "{name}")
        for rule in self.status_rules:
            raw_pat = str(rule.get("regex") or rule.get("pattern") or "").strip()
            has_ph = any(tok in raw_pat for tok in placeholder_tokens)
            rule["_raw_pat"] = raw_pat
            rule["_has_placeholder"] = has_ph
            if raw_pat and not has_ph:
                try:
                    rule["_compiled_rx"] = re.compile(raw_pat, re.IGNORECASE)
                except re.error:
                    rule["_compiled_rx"] = None
            else:
                rule["_compiled_rx"] = None

    @staticmethod
    def _normalize_rgba(color: Any) -> Optional[Tuple[int, int, int, int]]:
        """Normalize an RGB/RGBA sequence into a 4-tuple of ints `(R, G, B, A)`."""
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            return None
        try:
            r = max(0, min(255, int(color[0])))
            g = max(0, min(255, int(color[1])))
            b = max(0, min(255, int(color[2])))
            a = max(0, min(255, int(color[3]))) if len(color) >= 4 else 255
            return (r, g, b, a)
        except (ValueError, TypeError):
            return None

    def _get_color_theme(self, rgba: Optional[Tuple[int, int, int, int]]) -> Union[int, str]:
        """Return a cached DPG selectable text-color theme for `rgba` (or 0 if None)."""
        if rgba is None:
            return 0
        if rgba not in self._color_themes:
            with dpg.theme() as theme_id:
                with dpg.theme_component(dpg.mvSelectable):
                    dpg.add_theme_color(dpg.mvThemeCol_Text, rgba)
            self._color_themes[rgba] = theme_id
        return self._color_themes[rgba]

    @staticmethod
    def _rule_matches_video(
        rule: Dict[str, Any],
        video_stem: str,
        video_filename: str,
        dir_entries: List[str],
        dir_entries_lower_set: Set[str],
    ) -> bool:
        """
        Check whether a status rule from `config.json` matches any sibling metadata file
        for `video_filename` (stem `video_stem`) in `dir_entries`.

        Supports `[filename]`, `{filename}`, `[stem]`, `{stem}`, `[name]`, `{name}` placeholders
        as well as full regular expressions.
        """
        raw_pat = rule.get("_raw_pat")
        if raw_pat is None:
            raw_pat = str(rule.get("regex") or rule.get("pattern") or "").strip()
        if not raw_pat:
            return False

        has_placeholder = rule.get("_has_placeholder")
        if has_placeholder is None:
            has_placeholder = any(
                tok in raw_pat for tok in ("[filename]", "{filename}", "[stem]", "{stem}", "[name]", "{name}")
            )
        if has_placeholder:
            # 1. Fast literal check if the pattern is a simple template without regex anchors
            lit = raw_pat
            for tok in ("[filename]", "{filename}", "[stem]", "{stem}"):
                lit = lit.replace(tok, video_stem)
            for tok in ("[name]", "{name}"):
                lit = lit.replace(tok, video_filename)
            lit_unescaped = lit.replace("\\.", ".")
            if lit_unescaped.startswith("^") and lit_unescaped.endswith("$"):
                lit_unescaped = lit_unescaped[1:-1]
            lit_low = lit_unescaped.lower()
            if lit_low != video_filename.lower() and lit_low in dir_entries_lower_set:
                return True

            # 2. Regex substitution check
            esc_stem = re.escape(video_stem)
            esc_name = re.escape(video_filename)
            rx_str = raw_pat
            for tok in ("[filename]", "{filename}", "[stem]", "{stem}"):
                rx_str = rx_str.replace(tok, esc_stem)
            for tok in ("[name]", "{name}"):
                rx_str = rx_str.replace(tok, esc_name)
            try:
                rx = re.compile(rx_str, re.IGNORECASE)
                return any(entry != video_filename and bool(rx.search(entry)) for entry in dir_entries)
            except re.error:
                return False

        # No placeholder: match against sibling files sharing the video stem
        rx = rule.get("_compiled_rx")
        if rx is None:
            try:
                rx = re.compile(raw_pat, re.IGNORECASE)
            except re.error:
                return False

        stem_lower = video_stem.lower()
        name_lower = video_filename.lower()
        for entry in dir_entries:
            e_low = entry.lower()
            if e_low == name_lower or not e_low.startswith(stem_lower):
                continue
            suffix = entry[len(video_stem):]
            if rx.search(entry) or rx.search(suffix):
                return True
        return False

    def _evaluate_video_status(
        self,
        video_path: str,
        dir_entries: Optional[List[str]] = None,
        dir_entries_lower_set: Optional[Set[str]] = None,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], bool, Optional[str]]:
        """
        Evaluate `self.status_rules` for `video_path`.

        Returns:
            Tuple of `(matched_color_rgba, skip_unless_reprocess, matched_rule_name)`.
        """
        folder_dir = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        video_stem = os.path.splitext(video_filename)[0]
        if dir_entries is None:
            try:
                dir_entries = os.listdir(folder_dir)
            except Exception:
                dir_entries = []
        if dir_entries_lower_set is None:
            dir_entries_lower_set = {e.lower() for e in dir_entries if e != video_filename}

        matched_color: Optional[Tuple[int, int, int, int]] = None
        matched_rule_name: Optional[str] = None
        skip_unless_reprocess = False

        for rule in self.status_rules:
            if self._rule_matches_video(rule, video_stem, video_filename, dir_entries, dir_entries_lower_set):
                if matched_color is None:
                    matched_color = self._normalize_rgba(rule.get("color"))
                    matched_rule_name = str(rule.get("name", "")) or None
                rule_skip = bool(
                    rule.get(
                        "skip_unless_reprocess",
                        rule.get("mark_as_processed", rule.get("inhibit_process_all", False)),
                    )
                )
                if rule_skip:
                    skip_unless_reprocess = True

        return matched_color, skip_unless_reprocess, matched_rule_name

    def _refresh_video_info_and_theme(
        self,
        video_info: Dict[str, Any],
        tag: Optional[Union[int, str]] = None,
        dir_entries: Optional[List[str]] = None,
        dir_entries_lower_set: Optional[Set[str]] = None,
    ) -> None:
        """Re-evaluates disk status rules for `video_info` and updates its DPG selectable theme."""
        color, skip_reproc, rule_name = self._evaluate_video_status(
            video_info["path"],
            dir_entries=dir_entries,
            dir_entries_lower_set=dir_entries_lower_set,
        )
        video_info["color"] = color
        video_info["skip_unless_reprocess"] = skip_reproc
        video_info["processed"] = skip_reproc
        video_info["matched_rule"] = rule_name

        if tag is None:
            tag = self.video_path_to_tag.get(video_info["path"])
        if tag and dpg.does_item_exist(tag):
            try:
                dpg.bind_item_theme(tag, self._get_color_theme(color))
            except Exception:
                pass

    def scan_directory(self, root_dir: str, recursive: bool = True) -> Optional[Dict[str, Any]]:
        """
        Scan directory recursively and build nested folder structure with video files.

        Args:
            root_dir: Root directory path to scan.
            recursive: Whether to scan child directories.

        Returns:
            Root node dictionary containing subdirs and video entries.
        """
        self._load_config()
        if hasattr(self, "format_tag") and dpg.does_item_exist(self.format_tag):
            combo_items = ["All"] + list(self.video_extensions)
            try:
                dpg.configure_item(self.format_tag, items=combo_items)
            except Exception:
                pass

        self.video_list = []
        self.folder_map = {}

        fmt = getattr(self, "file_format", "All")
        if fmt == "All":
            valid_exts = tuple(self.video_extensions)
        elif fmt.startswith("."):
            valid_exts = (fmt.lower(),)
        else:
            valid_exts = (f".{fmt.lower()}",)

        def scan_node(current_path: str) -> Optional[Dict[str, Any]]:
            name = os.path.basename(current_path)
            if not name:
                name = current_path

            subdirs: List[Dict[str, Any]] = []
            videos: List[Dict[str, Any]] = []

            try:
                with os.scandir(current_path) as it:
                    scandir_entries = list(it)
            except Exception:
                return None

            entries: List[str] = [e.name for e in scandir_entries]
            entries_lower_set: Set[str] = {name_str.lower() for name_str in entries}

            for d_entry in scandir_entries:
                entry = d_entry.name
                full_path = d_entry.path
                try:
                    is_dir = d_entry.is_dir()
                except OSError:
                    is_dir = False
                if is_dir:
                    if recursive:
                        child_node = scan_node(full_path)
                        if child_node:
                            subdirs.append(child_node)
                else:
                    ext = os.path.splitext(entry)[1].lower()
                    if ext in valid_exts:
                        color, skip_reproc, rule_name = self._evaluate_video_status(
                            full_path,
                            dir_entries=entries,
                            dir_entries_lower_set=entries_lower_set,
                        )
                        video_info = {
                            "name": entry,
                            "path": full_path,
                            "basename": os.path.splitext(entry)[0],
                            "processed": skip_reproc,
                            "skip_unless_reprocess": skip_reproc,
                            "color": color,
                            "matched_rule": rule_name,
                        }
                        videos.append(video_info)

            subdirs.sort(key=lambda x: x["name"])
            videos.sort(key=lambda x: x["name"])

            node: Dict[str, Any] = {
                "type": "folder",
                "name": name,
                "path": current_path,
                "subdirs": subdirs,
                "videos": videos,
            }
            self.folder_map[current_path] = node
            return node

        root_node = scan_node(root_dir)

        # Filter out folders that do not contain any videos recursively
        if root_node and recursive:

            def filter_node(node: Dict[str, Any]) -> bool:
                node["subdirs"] = [s for s in node["subdirs"] if filter_node(s)]
                return len(node["videos"]) > 0 or len(node["subdirs"]) > 0

            filter_node(root_node)

        # Collect all videos from the filtered tree
        if root_node:

            def collect_all(node: Dict[str, Any]) -> List[Dict[str, Any]]:
                vids: List[Dict[str, Any]] = []
                vids.extend(node["videos"])
                for s in node["subdirs"]:
                    vids.extend(collect_all(s))
                return vids

            self.video_list = collect_all(root_node)
        else:
            self.video_list = []

        return root_node

    def build_tree_ui(self, node: Optional[Dict[str, Any]], parent_tag: Union[int, str]) -> None:
        """
        Recursively create DPG tree nodes and selectables for folder hierarchy.

        Args:
            node: Folder node dict containing subdirs and videos.
            parent_tag: DPG parent container tag.
        """
        if not node:
            return

        if node["type"] == "folder":
            if len(node["videos"]) > 0 or len(node["subdirs"]) > 0:
                node_tag = f"dir_node_{self.UUID}_{self._next_id()}"
                with dpg.tree_node(label=node["name"], tag=node_tag, parent=parent_tag, default_open=True):
                    with dpg.popup(parent=node_tag, mousebutton=dpg.mvMouseButton_Right):
                        dpg.add_menu_item(
                            label="\uf04b  Process this folder",
                            callback=self._on_process_folder_context_clicked,
                            user_data=node["path"],
                        )

                    for subdir in node["subdirs"]:
                        self.build_tree_ui(subdir, node_tag)

                    for video_info in node["videos"]:
                        self._add_video_selectable(video_info, node_tag)
        else:
            self._add_video_selectable(node, parent_tag)

    def _add_video_selectable(self, video_info: Dict[str, Any], parent_tag: Union[int, str]) -> None:
        """
        Add a selectable item for an individual video file in the tree.

        Args:
            video_info: Dict containing name, path, basename, and processed state.
            parent_tag: DPG parent tree node tag.
        """
        video_tag = f"vid_node_{self.UUID}_{self._next_id()}"
        self.video_ui_map[video_tag] = video_info
        self.video_path_to_tag[video_info["path"]] = video_tag

        dpg.add_selectable(label=video_info["name"], tag=video_tag, parent=parent_tag, callback=self.on_video_clicked)
        if video_info.get("color") is not None:
            try:
                dpg.bind_item_theme(video_tag, self._get_color_theme(video_info["color"]))
            except Exception:
                pass
        elif video_info.get("processed", False):
            try:
                dpg.bind_item_theme(video_tag, self.theme_processed_green)
            except Exception:
                pass

        with dpg.popup(parent=video_tag, mousebutton=dpg.mvMouseButton_Right):
            dpg.add_menu_item(
                label="\uf064  Send path",
                callback=self._on_send_video_path_context_clicked,
                user_data=video_tag,
            )

    def update_file_list(self, path: str) -> None:
        """
        Refresh folder scan and regenerate DPG tree view.

        Args:
            path: Directory path to scan.
        """
        self.last_selectable = None
        self.last_video_selected = None
        self.video_ui_map = {}
        self.video_path_to_tag = {}
        self._ui_tag_counter = 0

        self.stop_process_queue()

        self.folder_tree = self.scan_directory(path, recursive=self.recursive)

        if dpg.does_item_exist(self.child_win_tree_tag):
            dpg.delete_item(self.child_win_tree_tag)
            if dpg.does_item_exist(self.tree_group_tag):
                dpg.push_container_stack(self.tree_group_tag)
                with dpg.child_window(tag=self.child_win_tree_tag):
                    if self.folder_tree and (
                        len(self.folder_tree["videos"]) > 0 or len(self.folder_tree["subdirs"]) > 0
                    ):
                        self.build_tree_ui(self.folder_tree, self.child_win_tree_tag)
                    else:
                        dpg.add_text("No videos found in this folder.", color=(200, 100, 100))
                dpg.pop_container_stack()

    def on_video_clicked(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """
        Handle user selection of a video from the tree.

        Args:
            sender: DPG selectable tag.
            app_data: Event value.
            user_data: User data attached to item.
        """
        if self.last_selectable is not None and dpg.does_item_exist(self.last_selectable):
            try:
                dpg.set_value(self.last_selectable, False)
            except Exception:
                pass

        self.last_selectable = sender
        if dpg.does_item_exist(sender):
            dpg.set_value(sender, True)

        video_info = self.video_ui_map.get(sender)
        if not video_info:
            return

        self.last_video_selected = video_info["path"]
        self._current_video_name = video_info["basename"]

        self.stop_video()

        self.video_capturer.set_video(self.last_video_selected)
        total = self.video_capturer.get_total_frames()
        self._total_frames = total
        if dpg.does_item_exist(self.progress_tag):
            dpg.configure_item(self.progress_tag, max_value=total if total > 0 else 100)
        self._ignore_slider_cb = True
        if dpg.does_item_exist(self.progress_tag):
            dpg.set_value(self.progress_tag, 0)
        self._ignore_slider_cb = False

        ret, frame = self.video_capturer.read()
        if ret and frame is not None:
            self.frame_cb(frame=frame, current_frame=1)

    def _on_send_video_path_context_clicked(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Context menu callback to transmit selected video path downstream."""
        video_tag = user_data
        video_info = self.video_ui_map.get(video_tag)
        if video_info:
            self._send_path(video_info["path"], video_info["basename"])

    def _send_path(self, path: str, basename: str) -> None:
        """
        Send video file path downstream via File Path output connection.

        Args:
            path: Absolute file path.
            basename: Filename without extension.
        """
        if not self.process_all_active:
            tag = self.video_path_to_tag.get(path)
            if tag and dpg.does_item_exist(tag):
                try:
                    dpg.bind_item_theme(tag, self.theme_processing_blue)
                except Exception:
                    pass

        payload = (path, basename)
        for module in list(self.connections.get("File Path", [])):
            if hasattr(module, "input_cb") and callable(module.input_cb):
                try:
                    module.input_cb(data=payload, data_type=IOTypes.FILE_PATH)
                except Exception as e:
                    logger.error(f"VideoReader send_path error to {getattr(module, 'winID', module)}: {e}")

    def _on_process_all_clicked(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Start batch processing queue for all videos in current tree."""
        self.start_process_queue(self.video_list)

    def _on_process_folder_context_clicked(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Context menu callback to start queue for a specific folder subtree."""
        folder_path = user_data
        folder_node = self.folder_map.get(folder_path)
        if folder_node:

            def collect_all(node: Dict[str, Any]) -> List[Dict[str, Any]]:
                vids: List[Dict[str, Any]] = []
                vids.extend(node["videos"])
                for s in node["subdirs"]:
                    vids.extend(collect_all(s))
                return vids

            folder_videos = collect_all(folder_node)
            self.start_process_queue(folder_videos)

    def _on_stop_process_clicked(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Stop active batch processing queue."""
        self.stop_process_queue()

    def start_process_queue(self, queue: List[Dict[str, Any]]) -> None:
        """
        Initialize and run sequential batch processing of videos.

        Args:
            queue: List of video info dicts to process.
        """
        self._load_config()
        dir_cache: Dict[str, Tuple[List[str], Set[str]]] = {}
        for tag, info in self.video_ui_map.items():
            f_dir = os.path.dirname(info["path"])
            cached_dir = dir_cache.get(f_dir)
            if cached_dir is None:
                try:
                    entries = os.listdir(f_dir)
                except Exception:
                    entries = []
                cached_dir = (entries, {e.lower() for e in entries})
                dir_cache[f_dir] = cached_dir
            self._refresh_video_info_and_theme(
                info,
                tag,
                dir_entries=cached_dir[0],
                dir_entries_lower_set=cached_dir[1],
            )

        reprocess_val = dpg.get_value(self.reprocess_tag) if dpg.does_item_exist(self.reprocess_tag) else self.reprocess
        if not reprocess_val:
            queue = [v for v in queue if not v.get("skip_unless_reprocess", v.get("processed", False))]

        if not queue:
            if dpg.does_item_exist(self.process_status_tag):
                dpg.set_value(self.process_status_tag, "All videos already processed. Skipping.")
            return

        self.stop_video()

        with self.process_lock:
            self.process_queue = queue
            self.process_queue_index = 0
            self.process_all_active = True

            if dpg.does_item_exist(self.stop_process_all_tag):
                dpg.configure_item(self.stop_process_all_tag, enabled=True)
            if dpg.does_item_exist(self.process_all_tag):
                dpg.configure_item(self.process_all_tag, enabled=False)

            first_video = self.process_queue[0]
            path = first_video["path"]
            basename = first_video["basename"]

            status_text = f"Processing 1/{len(self.process_queue)}: {first_video['name']}"
            if dpg.does_item_exist(self.process_status_tag):
                dpg.set_value(self.process_status_tag, status_text)

            tag = self.video_path_to_tag.get(path)
            if tag and dpg.does_item_exist(tag):
                if self.last_selectable and dpg.does_item_exist(self.last_selectable):
                    try:
                        dpg.set_value(self.last_selectable, False)
                    except Exception:
                        pass
                try:
                    dpg.set_value(tag, True)
                    self.last_selectable = tag
                    dpg.bind_item_theme(tag, self.theme_processing_blue)
                except Exception:
                    pass

            self._send_path(path, basename)

    def _process_next_in_queue(self) -> None:
        """Advance queue after downstream processing of current video completes."""
        with self.process_lock:
            if not self.process_all_active:
                return

            if self.process_queue_index < len(self.process_queue):
                current_video = self.process_queue[self.process_queue_index]
                current_tag = self.video_path_to_tag.get(current_video["path"])
                if current_tag and dpg.does_item_exist(current_tag):
                    try:
                        dpg.set_value(current_tag, False)
                    except Exception:
                        pass
                self._refresh_video_info_and_theme(current_video, current_tag)

            self.process_queue_index += 1
            if self.process_queue_index < len(self.process_queue):
                next_video = self.process_queue[self.process_queue_index]
                path = next_video["path"]
                basename = next_video["basename"]

                status_text = f"Processing {self.process_queue_index + 1}/{len(self.process_queue)}: {next_video['name']}"
                if dpg.does_item_exist(self.process_status_tag):
                    dpg.set_value(self.process_status_tag, status_text)

                tag = self.video_path_to_tag.get(path)
                if tag and dpg.does_item_exist(tag):
                    if self.last_selectable and dpg.does_item_exist(self.last_selectable):
                        try:
                            dpg.set_value(self.last_selectable, False)
                        except Exception:
                            pass
                    try:
                        dpg.set_value(tag, True)
                        self.last_selectable = tag
                        dpg.bind_item_theme(tag, self.theme_processing_blue)
                    except Exception:
                        pass

                self._send_path(path, basename)
            else:
                self.process_all_active = False
                if dpg.does_item_exist(self.process_status_tag):
                    dpg.set_value(
                        self.process_status_tag,
                        f"Finished processing all {len(self.process_queue)} videos.",
                    )
                if dpg.does_item_exist(self.stop_process_all_tag):
                    dpg.configure_item(self.stop_process_all_tag, enabled=False)
                if dpg.does_item_exist(self.process_all_tag):
                    dpg.configure_item(self.process_all_tag, enabled=True)

    def stop_process_queue(self) -> None:
        """Halt active sequential queue processing."""
        with self.process_lock:
            if self.process_all_active:
                self.process_all_active = False
                if dpg.does_item_exist(self.process_status_tag):
                    dpg.set_value(self.process_status_tag, "Processing stopped by user.")
                if dpg.does_item_exist(self.stop_process_all_tag):
                    dpg.configure_item(self.stop_process_all_tag, enabled=False)
                if dpg.does_item_exist(self.process_all_tag):
                    dpg.configure_item(self.process_all_tag, enabled=True)

    def play_loop(self) -> None:
        """Background thread loop streaming video frames at target FPS."""
        next_frame_time = time.perf_counter()

        while self.video_running and not self._is_closing:
            if not self.is_outputs_ready():
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            if now < next_frame_time:
                time.sleep(min(0.005, next_frame_time - now))
                continue

            if self._seek_request is not None:
                self.video_capturer.set_current_frame(self._seek_request)
                self._seek_request = None
                if self.video_paused:
                    ret, frame = self.video_capturer.read()
                    if ret and frame is not None:
                        current_frame = self.video_capturer.get_current_frame()
                        self.frame_cb(frame=frame, current_frame=current_frame)
                        self._ignore_slider_cb = True
                        if dpg.does_item_exist(self.progress_tag):
                            dpg.set_value(self.progress_tag, current_frame)
                        self._ignore_slider_cb = False

            if self.video_paused:
                time.sleep(0.01)
                next_frame_time = time.perf_counter()
                continue

            ret, frame = self.video_capturer.read()
            if not ret:
                break
            if frame is not None:
                current_frame = self.video_capturer.get_current_frame()
                self.frame_cb(frame=frame, current_frame=current_frame)

                if self._seek_request is None:
                    self._ignore_slider_cb = True
                    if dpg.does_item_exist(self.progress_tag):
                        dpg.set_value(self.progress_tag, current_frame)
                    self._ignore_slider_cb = False

            next_frame_time += self.frame_interval
            if next_frame_time < time.perf_counter():
                next_frame_time = time.perf_counter()

        self.trigger_cb(event="STOP")
        self.video_running = False
        self.video_thread = None

    def start_video(self, *args: Any, **kwargs: Any) -> None:
        """Start or restart video playback from beginning."""
        if not self.last_video_selected:
            logger.warning(f"{self.winID} No video selected to play.")
            return

        self.trigger_cb(event="START")
        self.video_capturer.set_video(self.last_video_selected)
        self._total_frames = self.video_capturer.get_total_frames()
        self.video_capturer.set_current_frame(0)
        self.video_running = True
        self.video_paused = False
        if dpg.does_item_exist(self.pause_tag):
            dpg.set_item_label(self.pause_tag, "\uf04c")

        if self.video_thread is None or not self.video_thread.is_alive():
            self.video_thread = threading.Thread(target=self.play_loop, daemon=True)
            self.video_thread.start()

    def toggle_pause(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Toggle playback pause / resume state."""
        if not self.last_video_selected or not self.video_running:
            return

        self.video_paused = not self.video_paused
        icon = "\uf04b" if self.video_paused else "\uf04c"
        if dpg.does_item_exist(self.pause_tag):
            dpg.set_item_label(self.pause_tag, icon)

    def stop_video(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Stop video playback and reset transport controls."""
        self.video_running = False
        self.video_paused = False
        if dpg.does_item_exist(self.pause_tag):
            dpg.set_item_label(self.pause_tag, "\uf04c")

    def set_playback_fps(
        self,
        sender: Any,
        app_data: int,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        """Update playback target frame rate."""
        self.playback_fps = max(1, int(app_data))
        self.frame_interval = 1.0 / self.playback_fps

    def on_slider_change(
        self,
        sender: Any,
        app_data: int,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Seek video playback position to slider value."""
        if getattr(self, "_ignore_slider_cb", False):
            return
        self._seek_request = app_data

        if self.last_video_selected and not self.video_running:
            if self.video_capturer.video is None:
                self.video_capturer.set_video(self.last_video_selected)
                self._total_frames = self.video_capturer.get_total_frames()
            self.video_capturer.set_current_frame(self._seek_request)
            ret, frame = self.video_capturer.read()
            if ret and frame is not None:
                self.frame_cb(frame=frame)
            self._seek_request = None

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming data from upstream nodes or external commands.

        Accepts:
            - Completion triggers (IOTypes.TRIGGER)
            - Command dictionary (IOTypes.CMD_DICT)
            - Folder path string/tuple (IOTypes.FOLDER_PATH)
            - File path string/tuple (IOTypes.FILE_PATH)
        """
        data = kwargs.get("data") if kwargs.get("data") is not None else (args[0] if args else None)
        data_type = kwargs.get("data_type")

        direct_folder = kwargs.get("folder_path") or kwargs.get("folder") or kwargs.get("path")
        if direct_folder and isinstance(direct_folder, str):
            if self.set_folder(direct_folder):
                return

        # Downstream completion trigger -> advances batch queue
        if data_type == IOTypes.TRIGGER:
            logger.info(f"VideoReader received completion trigger: {data}")

            # Re-evaluate status rules on disk for the completed video (e.g. .processed -> green, _drift.csv -> blue)
            if isinstance(data, str):
                tag = self.video_path_to_tag.get(data)
                if tag and tag in self.video_ui_map:
                    self._refresh_video_info_and_theme(self.video_ui_map[tag], tag)

            if self.process_all_active:
                self._process_next_in_queue()
            return

        # Command dictionary
        cmd = None
        if isinstance(data, dict):
            cmd = data
        elif data_type == IOTypes.CMD_DICT:
            cmd = data

        if cmd and isinstance(cmd, dict):
            if cmd.get("browse", False):
                self._on_browse_clicked()

            if cmd.get("start", False):
                self.start_video()

            if cmd.get("stop", False):
                self.stop_video()

            if cmd.get("pause", False):
                self.toggle_pause(None, None)

            if "fps" in cmd:
                try:
                    val = int(cmd["fps"])
                    if dpg.does_item_exist(self.fps_tag):
                        dpg.set_value(self.fps_tag, val)
                    self.set_playback_fps(None, val)
                except (ValueError, TypeError):
                    pass

            cmd_folder = (
                cmd.get("folder")
                or cmd.get("folder_path")
                or cmd.get("path")
                or cmd.get("working_dir")
                or cmd.get("output_dir")
            )
            if cmd_folder:
                self.set_folder(cmd_folder)
            return

        # Folder or file paths
        if data_type == IOTypes.FOLDER_PATH:
            target_dir = None
            if isinstance(data, str):
                target_dir = data
            elif isinstance(data, (tuple, list)) and len(data) > 0 and isinstance(data[0], str):
                target_dir = data[0]
            elif isinstance(data, dict):
                target_dir = data.get("folder_path") or data.get("folder") or data.get("path")
            if target_dir:
                if self.set_folder(target_dir):
                    return

        if isinstance(data, str):
            if os.path.isdir(data) or (data_type == IOTypes.FOLDER_PATH):
                if self.set_folder(data):
                    return
            elif os.path.isfile(data):
                file_dir = os.path.dirname(data)
                self.set_folder(file_dir)
                return

        if isinstance(data, (tuple, list)) and len(data) > 0 and isinstance(data[0], str):
            first_elem = data[0]
            if os.path.isdir(first_elem):
                if self.set_folder(first_elem):
                    return
            elif os.path.isfile(first_elem):
                file_dir = os.path.dirname(first_elem)
                self.set_folder(file_dir)
                return

    def frame_cb(self, frame: np.ndarray, current_frame: Optional[int] = None) -> None:
        """
        Transmit decoded video frame downstream with video metadata.

        Args:
            frame: Video image frame array.
            current_frame: Optional pre-queried 1-based frame position after read().
        """
        cur = current_frame if current_frame is not None else self.video_capturer.get_current_frame()
        frame_idx = max(0, cur - 1)
        self._playback_cb(frame_idx)
        payload = (frame, self._current_video_name, frame_idx)
        for module in list(self.connections.get("Frame", [])):
            if hasattr(module, "input_cb") and callable(module.input_cb):
                try:
                    module.input_cb(data=payload, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.error(f"VideoReader frame dispatch error to {getattr(module, 'winID', module)}: {e}")

    def _playback_cb(self, frame_idx: int) -> None:
        """
        Emit the playback state (video path + displayed frame number) on the Playback output.

        Sent right before the matching Frame so downstream modules (e.g. Motility Reprocessor)
        can stay synchronised with the video, including on seek and single-frame display.

        Args:
            frame_idx: Index of the frame being displayed.
        """
        modules = list(self.connections.get("Playback", []))
        if not modules:
            return
        total = self._total_frames if self._total_frames > 0 else self.video_capturer.get_total_frames()
        payload = {
            "path": self.last_video_selected,
            "name": self._current_video_name,
            "frame_idx": frame_idx,
            "total_frames": total,
            "playing": bool(self.video_running and not self.video_paused),
        }
        for module in modules:
            if hasattr(module, "input_cb") and callable(module.input_cb):
                try:
                    module.input_cb(data=dict(payload), data_type=IOTypes.PLAYBACK)
                except Exception as e:
                    logger.error(f"VideoReader playback dispatch error to {getattr(module, 'winID', module)}: {e}")

    def trigger_cb(self, event: Optional[str] = None) -> None:
        """
        Emit playback lifecycle event trigger downstream.

        Args:
            event: Event string ('START' or 'STOP').
        """
        for module in list(self.connections.get("Trigger", [])):
            if hasattr(module, "input_cb") and callable(module.input_cb):
                try:
                    module.input_cb(data=event, data_type=IOTypes.TRIGGER)
                except Exception as e:
                    logger.error(f"VideoReader trigger dispatch error to {getattr(module, 'winID', module)}: {e}")

    def on_close(self) -> None:
        """Stop queue processing, stop playback loop, join thread and release video capture."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        self.stop_process_queue()
        self.video_running = False
        if self.video_thread is not None and self.video_thread.is_alive():
            try:
                self.video_thread.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error joining video thread in {self.winID}: {e}")
        try:
            self.video_capturer.release()
        except Exception as e:
            logger.warning(f"Error releasing video capture in {self.winID}: {e}")
        super().on_close()

    def close(self) -> None:
        """Close window and unregister."""
        self.on_close()
        super().close()

    def destroy(self) -> None:
        """Destroy window and release resources."""
        self.close()
        super().destroy()


VideoReaderWin = VideoReader_win
EXPORTED_CLASS = VideoReader_win
EXPORTED_NAME = "Video Reader"
