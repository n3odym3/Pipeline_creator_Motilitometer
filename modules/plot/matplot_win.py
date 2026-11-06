from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
from loguru import logger

from config.display_scaling import display_scaling
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.app_state import app_state
from core.tools.clipboard_injector import clipboardinjector


class Matplot_win(WindowBase):
    """
    High-resolution Matplotlib & Seaborn rendering module with scriptable DataFrame filtering.

    Receives a long-format DataFrame (columns `serie`, `x`, `y`, optional `point`) through the
    `DATAFRAME` input (e.g. emitted from `Lineplot`). Plot metadata (title, xlabel, ylabel, x_lim,
    y_lim, log_x, colors) is read from `df.attrs`.

    Features a scriptable Data Explorer allowing:
      - Row-level filtering with pandas expressions (e.g. `x % 2 == 0 and y > 0`).
      - Intra-group selection for identical X values (e.g. `[-1]` for the last point of each light step).
      - Group aggregation (Mean ± SD, Median, Min, Max, Raw points).
      - Optional full Python/Pandas script mode (`df = ...`).
      - High-performance, lightweight preview table (no individual checkbox lag).
    """

    description: str = "High-resolution Matplotlib and Seaborn plot renderer with Scriptable DataFrame filtering."

    # 1. Window Configuration
    label = "Matplotlib Viewer"
    win_width = display_scaling.scale(800)
    win_height = display_scaling.scale(600)

    # 2. Input/Output definitions
    accepted_input_types = [IOTypes.DATAFRAME]
    outputs = {
        "Frame": IOTypes.FRAME,
    }

    # 3. Persistent fields
    persistent_fields = [
        "label", "plot_title", "xlabel", "ylabel", "dpi",
        "canvas_width", "canvas_height", "show_legend", "use_seaborn", "use_shading",
        "filter_expr", "group_slice", "agg_mode", "use_custom_script", "custom_script",
    ]

    # Persistent field defaults
    plot_title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    dpi: int = 100
    canvas_width: int = 800
    canvas_height: int = 600
    show_legend: bool = True
    use_seaborn: bool = True
    use_shading: bool = False
    filter_expr: str = ""
    group_slice: str = ""
    agg_mode: str = "Mean ± SD"
    use_custom_script: bool = False
    custom_script: str = "# Custom script (assign result to 'df')\n# df = df[df['x'] % 2 == 0]\n"

    agg_modes: List[str] = ["Mean ± SD", "Mean", "Median", "First", "Last", "Min", "Max", "Raw points"]
    _applied_seaborn_state: Optional[bool] = None

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Matplotlib Viewer window module.
        """
        super().__init__(**kwargs)
        self._is_closing: bool = False

        # Ensure correct types for configuration state
        self.plot_title = str(self.plot_title or "")
        self.xlabel = str(self.xlabel or "")
        self.ylabel = str(self.ylabel or "")
        self.dpi = int(self.dpi if self.dpi is not None else 100)
        self.canvas_width = int(self.canvas_width if self.canvas_width is not None else 800)
        self.canvas_height = int(self.canvas_height if self.canvas_height is not None else 600)
        self.show_legend = bool(self.show_legend if self.show_legend is not None else True)
        self.use_seaborn = bool(self.use_seaborn if self.use_seaborn is not None else True)
        self.use_shading = bool(self.use_shading if self.use_shading is not None else False)

        # Scriptable Filter & Slicing State
        self.filter_expr = str(self.filter_expr or "")
        self.group_slice = str(self.group_slice or "")
        self.agg_mode = self.agg_mode if self.agg_mode in self.agg_modes else self.agg_modes[0]
        self.use_custom_script = bool(self.use_custom_script if self.use_custom_script is not None else False)
        self.custom_script = self.custom_script or "# Custom script (assign result to 'df')\n# df = df[df['x'] % 2 == 0]\n"

        # Data state & UI Tags
        self.df: Optional[pd.DataFrame] = None
        self._last_processed_df: Optional[pd.DataFrame] = None
        self.series_enabled: Dict[str, bool] = {}
        self.last_status_message: str = ""
        self.last_status_is_error: bool = False

        self.config_win_tag = f"matplot_config_{self.UUID}"
        self.data_win_tag = f"matplot_data_{self.UUID}"
        self.data_series_tag = f"matplot_data_series_{self.UUID}"
        self.data_status_tag = f"matplot_data_status_{self.UUID}"
        self.data_preview_tag = f"matplot_data_preview_{self.UUID}"
        self.data_std_group_tag = f"matplot_std_group_{self.UUID}"
        self.data_script_group_tag = f"matplot_script_group_{self.UUID}"

        # Texture Handling
        self.texture_registry_tag = f"matplot_registry_{self.UUID}"
        self.texture_tag = f"matplot_texture_{self.UUID}"
        self.texture_width = self.canvas_width
        self.texture_height = self.canvas_height
        self._tex_buffer = np.zeros((self.texture_height, self.texture_width, 3), dtype=np.float32)
        self.last_bgr_frame = None
        self.last_plot_data = None

        with dpg.texture_registry(tag=self.texture_registry_tag, show=False):
            dpg.add_raw_texture(
                width=self.texture_width,
                height=self.texture_height,
                default_value=self._tex_buffer.ravel(),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb,
            )

        with self.window():
            with dpg.group(horizontal=True):
                dpg.add_button(label="Clipboard", callback=self.copy_to_clipboard)
                dpg.add_button(label="Advanced Options", callback=self._show_advanced_options)
                dpg.add_button(label="Data Filter", callback=self._show_data_explorer)

            dpg.add_separator()
            with dpg.group(tag=f"matplot_wrapper_{self.UUID}"):
                self._build_plot()

        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on the application mode.
        """
        mode = app_state.mode
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=(mode == "user"))

    def _show_advanced_options(self, *args: Any, **kwargs: Any) -> None:
        """
        Display configuration modal for adjusting dimensions, DPI, labels, and rendering styles.
        """
        config_win = self.config_win_tag
        if dpg.does_item_exist(config_win):
            dpg.show_item(config_win)
            dpg.focus_item(config_win)
            return

        with dpg.window(
            label=f"Config - {self.label}",
            tag=config_win,
            width=300,
            height=350,
            pos=(self.pos[0] + 50, self.pos[1] + 50),
        ):
            dpg.add_input_text(label="Title", default_value=self.plot_title, callback=lambda s, a: setattr(self, "plot_title", a))
            dpg.add_input_text(label="X Label", default_value=self.xlabel, callback=lambda s, a: setattr(self, "xlabel", a))
            dpg.add_input_text(label="Y Label", default_value=self.ylabel, callback=lambda s, a: setattr(self, "ylabel", a))
            dpg.add_separator()
            dpg.add_input_int(label="Width", default_value=self.canvas_width, callback=lambda s, a: setattr(self, "canvas_width", a))
            dpg.add_input_int(label="Height", default_value=self.canvas_height, callback=lambda s, a: setattr(self, "canvas_height", a))
            dpg.add_input_int(label="DPI", default_value=self.dpi, callback=lambda s, a: setattr(self, "dpi", a))
            dpg.add_checkbox(label="Show Legend", default_value=self.show_legend, callback=lambda s, a: setattr(self, "show_legend", a))
            dpg.add_checkbox(label="Seaborn Look", default_value=self.use_seaborn, callback=lambda s, a: setattr(self, "use_seaborn", a))
            dpg.add_checkbox(label="Shading", default_value=self.use_shading, callback=lambda s, a: setattr(self, "use_shading", a))
            dpg.add_separator()
            dpg.add_button(label="Apply and Refresh", width=-1, callback=self._apply_config)

    def _build_plot(self) -> None:
        """
        Construct Dear PyGui image plot canvas matching current texture dimensions.
        """
        with dpg.plot(height=-1, width=-1, no_menus=True, tag=f"matplot_plot_{self.UUID}", parent=f"matplot_wrapper_{self.UUID}"):
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True)
            with dpg.plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True) as yaxis:
                dpg.add_image_series(self.texture_tag, [0, 0], [self.texture_width, self.texture_height])
                dpg.fit_axis_data(xaxis)
                dpg.fit_axis_data(yaxis)

    def _apply_config(self, *args: Any, **kwargs: Any) -> None:
        """
        Apply dimension and style updates, reallocating texture memory if canvas size changed.
        """
        if self.canvas_width != self.texture_width or self.canvas_height != self.texture_height:
            self.texture_width = self.canvas_width
            self.texture_height = self.canvas_height
            self._tex_buffer = np.zeros((self.texture_height, self.texture_width, 3), dtype=np.float32)

            if dpg.does_item_exist(self.texture_tag):
                dpg.delete_item(self.texture_tag)
            if dpg.does_item_exist(f"matplot_plot_{self.UUID}"):
                dpg.delete_item(f"matplot_plot_{self.UUID}")

            dpg.add_raw_texture(
                width=self.texture_width,
                height=self.texture_height,
                default_value=self._tex_buffer.ravel(),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb,
                parent=self.texture_registry_tag,
            )
            self._build_plot()

        self._rerender()

    def copy_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """
        Copy rendered figure image directly to the system clipboard.
        """
        if self.last_bgr_frame is not None:
            clipboardinjector.send_image(self.last_bgr_frame)
            logger.success("Plot copied to clipboard.")
        else:
            logger.warning("No plot available to copy.")

    def is_ready(self) -> bool:
        return True

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle an incoming plot DataFrame (plot metadata is read from `df.attrs`).
        """
        if isinstance(data, pd.DataFrame):
            self._load_df(data)
            self.last_plot_data = dict(data.attrs)
            dpg.show_item(self.winID)
            dpg.focus_item(self.winID)
            self.render_plot(self.last_plot_data)

    # ------------------------------------------------------------------
    # Data Processing Pipeline (Filter -> Slice -> Aggregate / Script)
    # ------------------------------------------------------------------
    def _load_df(self, df: pd.DataFrame) -> None:
        """Store new DataFrame and synchronize series selection."""
        self.df = df.reset_index(drop=True)
        names = [str(s) for s in self.df["serie"].unique()]
        self.series_enabled = {s: self.series_enabled.get(s, True) for s in names}
        if dpg.does_item_exist(self.data_win_tag):
            self._refresh_series_checkboxes()

    @staticmethod
    def _apply_row_filter(df: pd.DataFrame, expr: str) -> pd.DataFrame:
        """
        Filter DataFrame rows using a boolean expression (pandas query / Python eval).
        Supports: `x % 2 == 0`, `y > 0 and x <= 500`, `point >= 2`, `serie != 'Blank'`.
        """
        expr = expr.strip()
        if not expr:
            return df
        try:
            return df.query(expr, engine="python")
        except Exception:
            local_dict = {col: df[col] for col in df.columns}
            local_dict.update({"df": df, "np": np, "pd": pd})
            res = eval(expr, {"__builtins__": {}}, local_dict)
            if isinstance(res, (pd.Series, np.ndarray)):
                return df[np.asarray(res, dtype=bool)]
            raise

    @staticmethod
    def _apply_group_slice(df: pd.DataFrame, slice_str: str) -> pd.DataFrame:
        """
        Apply intra-group indexing / slicing for each (serie, x) group.
        Supports:
          - '[-1]' or '-1' : last measurement of each X
          - '[0]' or '0'   : first measurement of each X
          - '[-2:]'        : last 2 measurements of each X
          - '[1:]'         : ignore first measurement of each X
          - '[:2]'         : first 2 measurements of each X
          - '[::2]'        : every second measurement
          - 'all' or ''    : keep all measurements
        """
        s = slice_str.strip()
        if not s or s.lower() in ("all", "*", "[:]", "none"):
            return df

        clean = s
        if clean.startswith("[") and clean.endswith("]"):
            clean = clean[1:-1].strip()

        # Single integer index, e.g. -1, 0, 2
        try:
            idx = int(clean)
            return df.groupby(["serie", "x"], sort=False, as_index=False).nth(idx)
        except ValueError:
            pass

        # Python slice, e.g. -2:, 1:, :2, ::2
        if ":" in clean:
            parts = clean.split(":")
            start = int(parts[0].strip()) if parts[0].strip() else None
            stop = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else None
            step = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else None
            sl = slice(start, stop, step)

            g = df.groupby(["serie", "x"], sort=False)
            sizes = g["y"].transform("size").to_numpy()
            ranks = g.cumcount().to_numpy()

            if step is None or step > 0:
                if start is None:
                    low = np.zeros_like(sizes)
                else:
                    low = np.where(start >= 0, np.minimum(sizes, start), np.maximum(0, sizes + start))
                if stop is None:
                    high = sizes
                else:
                    high = np.where(stop >= 0, np.minimum(sizes, stop), np.maximum(0, sizes + stop))
                mask = (ranks >= low) & (ranks < high)
                if step is not None and step > 1:
                    mask &= ((ranks - low) % step == 0)
            else:
                unique_sizes = np.unique(sizes)
                slice_cache = {sz: set(range(sz)[sl]) for sz in unique_sizes}
                mask = np.fromiter((r in slice_cache[sz] for r, sz in zip(ranks, sizes)), dtype=bool, count=len(ranks))
            return df[mask]

        # Comma-separated indices, e.g. "0, -1"
        if "," in clean:
            indices = [int(p.strip()) for p in clean.split(",") if p.strip()]
            return df.groupby(["serie", "x"], sort=False, as_index=False).nth(indices)

        return df

    @staticmethod
    def _apply_custom_script(df: pd.DataFrame, script: str) -> pd.DataFrame:
        """
        Execute arbitrary Python / Pandas code on df and return resulting df.
        """
        script = script.strip()
        if not script:
            return df
        loc = {"df": df.copy(), "pd": pd, "np": np}
        exec(script, {"__builtins__": __builtins__}, loc)
        res_df = loc.get("df")
        if isinstance(res_df, pd.DataFrame):
            return res_df
        raise ValueError("Script must assign the result to variable 'df' (e.g. df = df[...]).")

    def _processed_df(self) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Execute the full filtering pipeline.
        Returns:
            Tuple[pd.DataFrame, Optional[str]]: (processed_df with columns ['serie', 'x', 'y', 'std'], error_msg)
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame(columns=["serie", "x", "y", "std"]), None

        df = self.df

        # 1. Active series filter
        active_series = [s for s, on in self.series_enabled.items() if on]
        if len(active_series) < len(self.series_enabled):
            df = df[df["serie"].astype(str).isin(active_series)]

        if df.empty:
            return pd.DataFrame(columns=["serie", "x", "y", "std"]), "All series are unchecked."

        try:
            # 2. Custom script OR Standard pipeline
            if self.use_custom_script and self.custom_script.strip():
                df = self._apply_custom_script(df, self.custom_script)
            else:
                # Step A: Row filter expression
                if self.filter_expr.strip():
                    df = self._apply_row_filter(df, self.filter_expr)

                # Step B: Per-X group slice
                if self.group_slice.strip():
                    df = self._apply_group_slice(df, self.group_slice)

            if df.empty:
                return pd.DataFrame(columns=["serie", "x", "y", "std"]), "0 points matching the filter criteria."

            # Ensure minimal required columns exist
            for req in ("serie", "x", "y"):
                if req not in df.columns:
                    raise KeyError(f"Missing required column '{req}' after processing.")

            # Step C: Aggregation
            if self.agg_mode == "Raw points":
                out = df[["serie", "x", "y"]].assign(std=np.nan)
            else:
                g = df.groupby(["serie", "x"], sort=False)["y"]
                funcs = {
                    "Mean ± SD": "mean",
                    "Mean": "mean",
                    "Median": "median",
                    "First": "first",
                    "Last": "last",
                    "Min": "min",
                    "Max": "max",
                }
                func = funcs.get(self.agg_mode, "mean")
                out = g.agg(func).rename("y").reset_index()
                out["std"] = g.std(ddof=0).to_numpy() if self.agg_mode == "Mean ± SD" else np.nan

            return out, None

        except Exception as e:
            return pd.DataFrame(columns=["serie", "x", "y", "std"]), f"{type(e).__name__}: {e}"

    def _rerender(self) -> None:
        """Trigger re-render of the plot canvas with current settings."""
        if self.df is not None:
            self.render_plot(self.last_plot_data or {})

    # ------------------------------------------------------------------
    # Data Explorer Window & Fast Preview
    # ------------------------------------------------------------------
    def _show_data_explorer(self, *args: Any, **kwargs: Any) -> None:
        """Open or focus the scriptable Data Explorer configuration window."""
        if dpg.does_item_exist(self.data_win_tag):
            dpg.show_item(self.data_win_tag)
            dpg.focus_item(self.data_win_tag)
            if dpg.does_item_exist(self.data_status_tag):
                dpg.set_value(self.data_status_tag, self.last_status_message or "Ready")
                dpg.configure_item(
                    self.data_status_tag,
                    color=(255, 100, 100) if self.last_status_is_error else (100, 220, 100),
                )
            self._update_preview_table(self._last_processed_df)
            return

        with dpg.window(
            label=f"Data Filter & Selection - {self.label}",
            tag=self.data_win_tag,
            width=580,
            height=540,
            pos=(self.pos[0] + 60, self.pos[1] + 60),
        ):
            dpg.add_text("Active Series:", color=(150, 200, 255))
            dpg.add_group(tag=self.data_series_tag, horizontal=True)
            dpg.add_separator()

            # Mode toggle
            dpg.add_checkbox(
                label="Advanced Python/Pandas Script Mode",
                default_value=self.use_custom_script,
                callback=self._on_script_mode_toggle,
            )
            dpg.add_spacer(height=4)

            # Standard pipeline group
            with dpg.group(tag=self.data_std_group_tag, show=not self.use_custom_script):
                dpg.add_text("1. Filter expression (pandas query):", color=(200, 220, 255))
                dpg.add_input_text(
                    default_value=self.filter_expr,
                    callback=self._on_filter_expr_change,
                    on_enter=True,
                    width=-1,
                )
                dpg.add_text("   Ex: x % 2 == 0, y > 0, point >= 1, serie != 'Blank'", color=(140, 140, 140))
                dpg.add_spacer(height=4)

                dpg.add_text("2. Per-X Selector (intra-group slice):", color=(200, 220, 255))
                dpg.add_input_text(
                    default_value=self.group_slice,
                    callback=self._on_group_slice_change,
                    on_enter=True,
                    width=-1,
                )
                dpg.add_text("   Ex: [-1] (last point per X), [0] (first), [-2:] (last 2), all", color=(140, 140, 140))
                dpg.add_spacer(height=4)

                with dpg.group(horizontal=True):
                    dpg.add_text("3. Aggregation:", color=(200, 220, 255))
                    dpg.add_combo(
                        items=self.agg_modes,
                        default_value=self.agg_mode,
                        width=180,
                        callback=self._on_agg_mode_change,
                    )

            # Custom script group
            with dpg.group(tag=self.data_script_group_tag, show=self.use_custom_script):
                dpg.add_text("Custom Python/Pandas Script (Input & Output: 'df'):", color=(200, 220, 255))
                dpg.add_input_text(
                    default_value=self.custom_script,
                    multiline=True,
                    height=110,
                    width=-1,
                    callback=self._on_custom_script_change,
                )
                dpg.add_text("   Ex: df = df[df['x'] % 2 == 0].groupby(['serie', 'x']).tail(1)", color=(140, 140, 140))

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Apply / Refresh", callback=lambda *_: self._rerender())
                dpg.add_button(label="Copy Processed Data (TSV)", callback=self._copy_processed_tsv)

            dpg.add_text(
                self.last_status_message or "Ready",
                tag=self.data_status_tag,
                color=(255, 100, 100) if self.last_status_is_error else (100, 220, 100),
            )
            dpg.add_separator()

            dpg.add_text("Data Preview (Sample rows):", color=(150, 200, 255))
            with dpg.child_window(tag=self.data_preview_tag, height=-1):
                pass

        self._refresh_series_checkboxes()
        self._update_preview_table(self._last_processed_df)

    def _on_script_mode_toggle(self, sender: Any, app_data: Any) -> None:
        self.use_custom_script = bool(app_data)
        if dpg.does_item_exist(self.data_std_group_tag):
            (dpg.hide_item if self.use_custom_script else dpg.show_item)(self.data_std_group_tag)
        if dpg.does_item_exist(self.data_script_group_tag):
            (dpg.show_item if self.use_custom_script else dpg.hide_item)(self.data_script_group_tag)
        self._rerender()

    def _on_filter_expr_change(self, sender: Any, app_data: Any) -> None:
        self.filter_expr = str(app_data)
        self._rerender()

    def _on_group_slice_change(self, sender: Any, app_data: Any) -> None:
        self.group_slice = str(app_data)
        self._rerender()

    def _on_agg_mode_change(self, sender: Any, app_data: Any) -> None:
        self.agg_mode = str(app_data)
        self._rerender()

    def _on_custom_script_change(self, sender: Any, app_data: Any) -> None:
        self.custom_script = str(app_data)

    def _on_series_toggle(self, sender: Any, app_data: Any, user_data: str) -> None:
        self.series_enabled[user_data] = bool(app_data)
        self._rerender()

    def _refresh_series_checkboxes(self) -> None:
        if not dpg.does_item_exist(self.data_series_tag):
            return
        dpg.delete_item(self.data_series_tag, children_only=True)
        if not self.series_enabled:
            dpg.add_text("No data received yet.", parent=self.data_series_tag, color=(140, 140, 140))
            return
        for s, on in self.series_enabled.items():
            dpg.add_checkbox(
                label=str(s),
                default_value=on,
                user_data=s,
                callback=self._on_series_toggle,
                parent=self.data_series_tag,
            )

    def _update_preview_table(self, processed_df: Optional[pd.DataFrame] = None) -> None:
        """Update fast preview table with head & tail rows."""
        if not dpg.does_item_exist(self.data_preview_tag):
            return
        dpg.delete_item(self.data_preview_tag, children_only=True)

        if processed_df is None or processed_df.empty:
            dpg.add_text("No data to display.", parent=self.data_preview_tag, color=(140, 140, 140))
            return

        # Show head and tail without creating thousands of widgets
        if len(processed_df) <= 24:
            sample = processed_df
        else:
            sample = pd.concat([processed_df.head(12), processed_df.tail(12)]).drop_duplicates()

        with dpg.table(
            parent=self.data_preview_tag,
            header_row=True,
            borders_innerH=True,
            borders_innerV=True,
            policy=dpg.mvTable_SizingFixedFit,
        ):
            dpg.add_table_column(label="Serie")
            dpg.add_table_column(label="X")
            dpg.add_table_column(label="Y")
            dpg.add_table_column(label="Std (±)")

            for r in sample.itertuples():
                with dpg.table_row():
                    dpg.add_text(str(r.serie))
                    dpg.add_text(f"{r.x:.6g}" if isinstance(r.x, (int, float, np.number)) else str(r.x))
                    dpg.add_text(f"{r.y:.6g}" if isinstance(r.y, (int, float, np.number)) else str(r.y))
                    has_std = hasattr(r, "std") and not pd.isna(r.std)
                    dpg.add_text(f"{r.std:.4g}" if has_std else "-")

    def _copy_processed_tsv(self, *args: Any) -> None:
        """Copy the processed plot dataset directly to clipboard as TSV."""
        proc_df, err = self._processed_df()
        if err or proc_df.empty:
            logger.warning(f"Cannot copy data: {err or 'empty dataset'}")
            return
        clipboardinjector.send_text(proc_df.to_csv(sep="\t", index=False))
        logger.success("Processed plot dataset copied to clipboard (TSV).")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_plot(self, plot_data: Dict[str, Any]) -> None:
        """
        Render filtered and aggregated dataset to Matplotlib canvas and DPG texture.
        """
        proc_df, err = self._processed_df()
        self._last_processed_df = proc_df

        # Update status feedback in Data window if open
        if err:
            self.last_status_message = f"⚠ {err}"
            self.last_status_is_error = True
        else:
            total_pts = len(proc_df)
            total_raw = len(self.df) if self.df is not None else 0
            n_series = proc_df["serie"].nunique() if not proc_df.empty else 0
            self.last_status_message = f"✓ {total_pts} points retained (from {total_raw} raw across {n_series} series)"
            self.last_status_is_error = False

        if dpg.does_item_exist(self.data_win_tag) and dpg.is_item_shown(self.data_win_tag):
            if dpg.does_item_exist(self.data_status_tag):
                dpg.set_value(self.data_status_tag, self.last_status_message)
                dpg.configure_item(
                    self.data_status_tag,
                    color=(255, 100, 100) if self.last_status_is_error else (100, 220, 100),
                )
            self._update_preview_table(proc_df)

        try:
            if Matplot_win._applied_seaborn_state != self.use_seaborn:
                if self.use_seaborn:
                    sns.set_theme(style="darkgrid", palette="muted")
                else:
                    plt.rcParams.update(plt.rcParamsDefault)
                Matplot_win._applied_seaborn_state = self.use_seaborn

            dpi = self.dpi
            fig = Figure(figsize=(self.texture_width / dpi, self.texture_height / dpi), dpi=dpi)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            if not self.use_seaborn:
                ax.grid(True)

            if not proc_df.empty:
                raw = self.agg_mode == "Raw points"
                colors_dict = plot_data.get("colors", {}) if isinstance(plot_data, dict) else {}

                for serie, sdf in proc_df.groupby("serie", sort=False):
                    if not raw:
                        sdf = sdf.sort_values("x")
                    x = sdf["x"].to_numpy()
                    y = sdf["y"].to_numpy()
                    std = sdf["std"].to_numpy() if "std" in sdf.columns else np.full(len(x), np.nan)

                    line_color = colors_dict.get(str(serie))
                    line, = ax.plot(
                        x, y, label=str(serie),
                        marker="o" if raw else None, markersize=3,
                        color=line_color,
                    )
                    c = line.get_color()
                    if not np.isnan(std).all():
                        ax.fill_between(x, y - std, y + std, color=c, alpha=0.2)
                    if self.use_shading:
                        ax.fill_between(x, y, color=c, alpha=0.15)

            ax.set_title(self.plot_title if self.plot_title else plot_data.get("title", "Plot"))
            ax.set_xlabel(self.xlabel if self.xlabel else plot_data.get("xlabel", "X"))
            ax.set_ylabel(self.ylabel if self.ylabel else plot_data.get("ylabel", "Y"))

            x_lim = plot_data.get("x_lim")
            y_lim = plot_data.get("y_lim")
            if x_lim:
                ax.set_xlim(x_lim)
            if y_lim:
                ax.set_ylim(y_lim)

            if plot_data.get("log_x"):
                ax.set_xscale("log")

            if self.show_legend and ax.has_data():
                ax.legend()
            ax.grid(True)

            canvas.draw()
            rgba_buffer = canvas.buffer_rgba()
            img_rgba = np.asarray(rgba_buffer)

            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            self.last_bgr_frame = img_bgr

            self._emit_frame(img_bgr)

            np.multiply(img_rgba[:, :, :3], 1.0 / 255.0, out=self._tex_buffer, casting="unsafe")
            dpg.set_value(self.texture_tag, self._tex_buffer.ravel())

            logger.success("Matplotlib plot rendered & emitted (BGR).")

        except Exception as e:
            logger.error(f"Failed to render Matplotlib plot to DPG: {e}")

    def _emit_frame(self, frame: np.ndarray) -> None:
        """
        Emit rendered plot frame to connected downstream modules.
        """
        for module in self.connections.get("Frame", []):
            if hasattr(module, "input_cb"):
                try:
                    module.input_cb(data=frame, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.debug(f"Matplot_win frame emission failed: {e}")

    def on_close(self) -> None:
        """
        Release DPG auxiliary windows and GPU texture resources on module close.
        """
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        for tag in (self.config_win_tag, self.data_win_tag):
            if dpg.does_item_exist(tag):
                try:
                    dpg.delete_item(tag)
                except Exception:
                    pass
        if dpg.does_item_exist(self.texture_registry_tag):
            try:
                dpg.delete_item(self.texture_registry_tag)
            except Exception:
                pass
        super().on_close()

    def close(self) -> None:
        """
        Programmatically close the Matplotlib Viewer window.
        """
        self.on_close()
        if dpg.does_item_exist(self.winID):
            dpg.delete_item(self.winID)


EXPORTED_CLASS = Matplot_win
EXPORTED_NAME = "Matplotlib Viewer"
