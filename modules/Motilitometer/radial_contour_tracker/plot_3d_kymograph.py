"""
Solid 3D Chemotaxis Kymograph Volume Plotter (Plotly 3D Surface Block)
=======================================================================
Inspired directly by the Polymager Phototaxis Processor (plotly_manager.py), 
this script parses radial detections (*_radial_detections.csv) and constructs 
a continuous, solid 3D volume block with sharp vertical drop walls down to Z=0.

Axes:
-----
- X-axis: Radial Distance from Attractant Pellet (px or µm)
- Y-axis: Time (seconds or minutes)
- Z-axis: Worm Density / Population Count

Features:
---------
- 2D Gaussian Kernel Smoothing across Time (Y) and Distance (X).
- Single-trace Plotly go.Surface with extended boundary padding (z_base=0.0) 
  producing sharp, seamless vertical walls without multi-surface artifacts or dark grey shadows.
- Interactive 3D rotation, zoom, pan, hover tooltips, and standalone HTML export (*_3d_kymograph.html).

Usage:
------
1. Interactive File Selection:
   python plot_3d_kymograph.py

2. Command-Line Options:
   python plot_3d_kymograph.py "path/to/my_video_radial_detections.csv" --bins 50 --time-steps 40 --smooth-time 3.0 --smooth-space 2.0 --colorscale Viridis
"""

import sys
import os
import argparse
import numpy as np

def select_csv_file() -> str:
    """Opens a native OS file dialog to select a CSV file if no path is provided."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select Radial Detections CSV File",
        filetypes=[("CSV Files", "*_radial_detections.csv *.csv"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def parse_detections_matrix(file_path: str, num_bins: int = 50, n_time_steps: int = 40, smooth_time: float = 3.0, smooth_space: float = 2.0):
    """
    Parses detection CSV and constructs a smooth, regularly sampled (Y_time x X_dist) 
    density matrix Z with 2D Gaussian Kernel smoothing.
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception:
        import csv
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader)]
            rows = list(reader)
        data = []
        for r in rows:
            if r:
                try:
                    data.append([float(x) if x != '' else 0.0 for x in r])
                except ValueError:
                    pass
        data_arr = np.array(data)
        df = {h: data_arr[:, i] for i, h in enumerate(headers)}

    cols = {str(col).lower().strip(): col for col in df.keys()}
    time_col = next((cols[c] for c in cols if 'time' in c), None)
    if time_col is None:
        time_col = next((cols[c] for c in cols if 'frame' in c), None)

    dist_col = next((cols[c] for c in cols if 'distance_µm' in c or 'distance_um' in c or 'distance_px' in c or 'distance' in c), None)

    if dist_col is None:
        raise ValueError("Could not find Distance column in CSV.")

    unit_name = "px"
    if "µm" in str(dist_col) or "um" in str(dist_col):
        unit_name = "µm"

    times = df[time_col].to_numpy() if hasattr(df[time_col], 'to_numpy') else np.array(df[time_col])
    dists = df[dist_col].to_numpy() if hasattr(df[dist_col], 'to_numpy') else np.array(df[dist_col])

    max_dist = max(1.0, float(np.nanmax(dists)))
    min_time = float(np.nanmin(times))
    max_time = float(np.nanmax(times))

    # Build regular time intervals and spatial bins
    sampled_times = np.linspace(min_time, max_time, n_time_steps)
    dist_bins = np.linspace(0, max_dist, num_bins + 1)
    x_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])

    # 2D Histogram calculation across sampled time windows
    time_bin_edges = np.linspace(min_time, max_time, n_time_steps + 1)
    Z_raw, _, _ = np.histogram2d(times, dists, bins=[time_bin_edges, dist_bins])

    # Normalize by the number of frames contained in each time window to get Mean Worm Count per Frame
    frame_col_name = next((cols[c] for c in cols if 'frame' in c), None)
    if frame_col_name is not None:
        total_unique_frames = len(np.unique(df[frame_col_name]))
    else:
        total_unique_frames = len(np.unique(times))
    
    frames_per_window = max(1.0, float(total_unique_frames) / float(n_time_steps))
    Z_norm = Z_raw / frames_per_window

    # 2D Gaussian Kernel Smoothing across Time (Y) and Space (X)
    Z_smooth = Z_norm
    if smooth_time > 0 or smooth_space > 0:
        try:
            from scipy.ndimage import gaussian_filter
            Z_smooth = gaussian_filter(Z_norm, sigma=(smooth_time, smooth_space))
        except ImportError:
            pass

    return sampled_times, x_centers, max_dist, unit_name, Z_smooth

def plot_solid_3d_block_plotly(file_path: str, num_bins: int = 50, n_time_steps: int = 40, 
                               smooth_time: float = 3.0, smooth_space: float = 2.0, colorscale: str = "Viridis"):
    """
    Generates a solid, continuous 3D volume block (filled down to Z=0) 
    using the Polymager Phototaxis extended boundary surface algorithm.
    """
    sampled_times, x_centers, max_dist, unit_name, Z = parse_detections_matrix(
        file_path, num_bins=num_bins, n_time_steps=n_time_steps, 
        smooth_time=smooth_time, smooth_space=smooth_space
    )

    base_name = os.path.basename(file_path)

    try:
        import plotly.graph_objects as go
        print(f"Generating Phototaxis-Style Solid 3D Volume Block for {base_name}...")

        n_y, n_x = Z.shape
        z_base = 0.0

        # Duplicate edge coordinates to create sharp vertical drop walls down to Z=0 (Polymager Phototaxis Method)
        x_ext = np.concatenate([[x_centers[0]], x_centers, [x_centers[-1]]])
        y_ext = np.concatenate([[sampled_times[0]], sampled_times, [sampled_times[-1]]])

        # Build extended Z matrix with 0.0 base value on all outer boundaries
        Z_ext = np.full((n_y + 2, n_x + 2), z_base)
        Z_ext[1:-1, 1:-1] = Z

        # Single continuous surface trace creating top surface AND solid vertical side walls
        surf_trace = go.Surface(
            z=Z_ext,
            x=x_ext,
            y=y_ext,
            colorscale=colorscale,
            name="Chemotaxis 3D Volume",
            colorbar=dict(title=dict(text="Worm Count", font=dict(size=12))),
            hovertemplate=f"Distance (X): %{{x:.1f}} {unit_name}<br>Time (Y): %{{y:.1f}} s<br>Worm Count (Z): %{{z:.1f}} worms<extra></extra>",
            lighting=dict(ambient=0.85, diffuse=0.9, specular=0.0, roughness=1.0, fresnel=0.0),
            contours=dict(z=dict(show=False))
        )

        fig = go.Figure(data=[surf_trace])

        fig.update_layout(
            title=dict(
                text=f"Solid 3D Chemotaxis Population Volume Block (Filled to Z=0)<br><sup>{base_name}</sup>",
                x=0.5,
                font=dict(size=16, family="Arial", color="black")
            ),
            scene=dict(
                xaxis=dict(title=f"Distance from Pellet ({unit_name})", backgroundcolor="rgb(245, 245, 245)", gridcolor="white"),
                yaxis=dict(title="Time (seconds)", backgroundcolor="rgb(245, 245, 245)", gridcolor="white"),
                zaxis=dict(title="Worm Count (Z)", backgroundcolor="rgb(235, 235, 235)", gridcolor="white"),
                camera=dict(
                    eye=dict(x=1.8, y=-1.8, z=1.4)
                )
            ),
            autosize=True,
            margin=dict(l=20, r=20, b=20, t=60)
        )

        out_html = os.path.splitext(file_path)[0] + "_solid_3d_volume.html"
        fig.write_html(out_html)
        print(f"Successfully saved Solid 3D HTML Volume Block to: {out_html}")

        fig.show()

    except ImportError:
        print("Plotly is not installed. Falling back to Matplotlib Solid 3D Surface...")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        X_grid, Y_grid = np.meshgrid(x_centers, sampled_times)
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap=colorscale.lower(), edgecolor='none', alpha=0.95)
        ax.set_title(f"Solid 3D Chemotaxis Population Volume\n{base_name}", fontsize=13, fontweight='bold')
        ax.set_xlabel(f"Distance from Pellet ({unit_name})", fontsize=10, fontweight='bold')
        ax.set_ylabel("Time (seconds)", fontsize=10, fontweight='bold')
        ax.set_zlabel("Worm Accumulation Density", fontsize=10, fontweight='bold')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Worm Density")
        ax.view_init(elev=35, azim=225)

        out_png = os.path.splitext(file_path)[0] + "_solid_3d_volume.png"
        plt.savefig(out_png, dpi=300)
        print(f"Saved Matplotlib 3D solid surface plot to: {out_png}")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solid 3D Chemotaxis Population Volume Generator")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to *_radial_detections.csv file")
    parser.add_argument("--bins", type=int, default=50, help="Spatial distance bins (default: 50)")
    parser.add_argument("--time-steps", type=int, default=40, help="Number of sampled time intervals (default: 40)")
    parser.add_argument("--smooth-time", type=float, default=3.0, help="Temporal Gaussian smoothing sigma (default: 3.0)")
    parser.add_argument("--smooth-space", type=float, default=2.0, help="Spatial Gaussian smoothing sigma (default: 2.0)")
    parser.add_argument("--colorscale", type=str, default="Viridis", help="Plotly colorscale: Viridis, Plasma, Inferno, Magma, Turbo (default: Viridis)")
    args = parser.parse_args()

    target_path = args.csv_path or select_csv_file()

    if target_path and os.path.exists(target_path):
        plot_solid_3d_block_plotly(
            target_path, num_bins=args.bins, n_time_steps=args.time_steps, 
            smooth_time=args.smooth_time, smooth_space=args.smooth_space, colorscale=args.colorscale
        )
    else:
        print("No valid CSV file selected.")
