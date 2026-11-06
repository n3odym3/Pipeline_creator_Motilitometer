# Polar Plot Module

The **Polar Plot** module visualizes global vector orientations, angular distributions, and directional fields (such as optical flow vectors from object tracking) using circular histograms (Rose diagrams) and circular statistics rendered via an asynchronous background multiprocessing worker.

---

## Technical Specifications

- **Accepted Inputs**:
  - `IOTypes.VECTORS`: Vector dictionary containing `angles` (radians) and `magnitudes` (speed/length).
  - `IOTypes.SAMPLE` / `IOTypes.DATALIST`: Directional sample distributions.
  - `IOTypes.CMD_DICT`: Remote commands (`"clear"`, `"copy"`, `"export_tsv"`, `"update_params"`).
  - `IOTypes.TRIGGER`: Triggers immediate re-rendering of accumulated vectors.
- **Outputs**:
  - `Frame`: `IOTypes.FRAME` (Emits rendered BGR image of the polar plot to image viewers or video recorders).
  - `CMD`: `IOTypes.CMD_DICT` (Exports plot parameters and vector statistics).

---

## Key Features & Controls

- **Multiprocessing Worker**: Rendering is executed off the main UI thread via `PolarPlotProcessor` (`ProcessingBase`), ensuring zero GUI lag during heavy vector rendering.
- **Directional Histogram (Rose Diagram)**: Configurable number of angular sectors (bins from 8 to 72).
- **Accumulation Modes**:
  - **Batch Mode (Default)**: Collects a batch of $N$ frames before rendering a discrete summary.
  - **Rolling Window**: Continuously updates on every single frame over a sliding window of the last $N$ frames.
- **Speed & Magnitude Weighting**: Weights angular sector heights by vector velocity / speed rather than just vector count.
- **Circular Statistics**: Real-time computation of:
  - Mean circular angle ($\bar{\theta}$) in degrees and radians.
  - Directionality index ($R \in [0, 1]$), representing angular coherence.
  - Mean resultant vector arrow overlay.
  - Mean particle speed.
- **Color Mapping Options**:
  - `Angle (HSV Flow)`: Identical color mapping to tracking modules (Right = Red, Down = Green, Left = Cyan, Up = Blue).
  - `Angle (Colormap)`: Turbo, Viridis, Twilight, Plasma, Magma, etc.
  - `Magnitude / Intensity`: Color intensity proportional to sector density.
- **Coordinate System Adjustment**:
  - `Invert Y`: Inverts the vertical coordinate to accurately map camera/image space (where $+Y$ is down) to screen directions.
  - `Zero Ref`: Reference zero angle orientation (East or North).
  - `Direction`: Angular progression direction (Clockwise or Counter-Clockwise).
- **Export Capabilities**:
  - One-click copy of the rendered chart to the system clipboard (`Copy Image`).
  - Tab-Separated Values export of sector counts and statistics to clipboard (`Export TSV`).
