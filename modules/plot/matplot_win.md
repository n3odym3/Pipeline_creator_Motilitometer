# Matplotlib Viewer Module

The **Matplotlib Viewer** module renders high-resolution publication-quality 2D plots using Matplotlib and Seaborn directly inside the application canvas. It receives DataFrames emitted from upstream nodes (such as `Lineplot`, `Histogram`, or analysis modules) via the `DATAFRAME` type.

---

## Technical Specifications

- **Accepted Inputs**:
  - `IOTypes.DATAFRAME`: long-format pandas DataFrame with columns `serie`, `x`, `y`, optional `point`. Plot metadata is read from `df.attrs`: `title`, `xlabel`, `ylabel`, `x_lim`, `y_lim`, `log_x`.
- **Outputs**:
  - `Frame`: `IOTypes.FRAME` (Emits rendered BGR video frames of the plot canvas).

---

## Key Features & Controls

- **Seaborn Styling**: Automatically formats plots using Seaborn (`darkgrid` style, `muted` palette).
- **Scriptable Data Filter (`Data Filter` button)**: secondary window offering advanced pandas-based data selection without UI lag:
  - **Active Series**: One-click toggles per series.
  - **Row Filter Expression**: Pandas/Python expression (e.g. `x % 2 == 0`, `y > 0`, `point >= 2`, `serie != 'Blank'`).
  - **Per-X Selector (Group Slice)**: Python index/slice within each $X$ group (e.g. `[-1]` for the last measurement of each light step in an rETR sequence, `[0]` for the first, `[-2:]` for the last two, `[1:]` to ignore adaptation spikes, `all` for all).
  - **Aggregation**: Combines remaining measurements per $X$ (`Mean ± SD` with shaded error bands, `Mean`, `Median`, `First`, `Last`, `Min`, `Max`, `Raw points`).
  - **Custom Script Mode**: Multi-line Python/pandas editor for custom transformations (`df = ...`).
  - **Live Status & Preview**: Displays retained points vs raw count in real-time with error notifications and a read-only table preview.
  - **TSV Export**: `Copy Processed Data (TSV)` button for immediate clipboard export.
- **Logarithmic Scaling**: Respects and applies logarithmic scale transformations when passed from upstream modules.
- **Configurable Canvas & DPI**: Adjust resolution (e.g. 800x600, 1920x1080) and dots-per-inch (DPI) on the fly via the `Advanced Options` modal.
- **Clipboard Export**: One-click copying of the high-resolution rendered graphic directly to the system clipboard.
- **Frame Emission**: Broadcasts the rendered graphic as a native `IOTypes.FRAME` (BGR) to downstream video recorders or image viewports.
