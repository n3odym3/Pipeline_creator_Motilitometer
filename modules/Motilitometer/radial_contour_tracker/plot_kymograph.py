"""
Ultra-Fast Continuous Chemotaxis Kymograph Heatmap Plotter
===========================================================
Uses C-accelerated Pandas + NumPy (np.histogram2d) with 2D Gaussian Kernel temporal-spatial 
smoothing (matching the 3D Phototaxis surface style) to generate silky-smooth 2D heatmaps.

Axes:
-----
- X-axis: Radial Distance from Attractant Pellet (px or µm)
- Y-axis: Time (seconds), starting at t=0 at the BOTTOM
- Color intensity: Continuous Worm Accumulation Density

Usage:
------
1. Interactive File Selection:
   python plot_kymograph.py

2. Command-Line Options:
   python plot_kymograph.py "path/to/my_video_radial_detections.csv" --bins 50 --smooth-time 3.0 --smooth-space 2.0 --interp bicubic
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

def parse_detections_csv_fast(file_path: str, num_bins: int = 50, static_thresh: float = 0.5, smooth_time: float = 3.0, smooth_space: float = 2.0):
    """
    Ultra-fast vectorized CSV parser using Pandas / NumPy + np.histogram2d 
    with Phototaxis-style 2D Gaussian Kernel temporal-spatial smoothing.
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
    speed_col = next((cols[c] for c in cols if 'speed' in c), None)
    static_col = next((cols[c] for c in cols if 'is_static' in c or 'static' in c), None)

    if dist_col is None:
        raise ValueError("Could not find Distance column in CSV.")

    unit_name = "px"
    if "µm" in str(dist_col) or "um" in str(dist_col):
        unit_name = "µm"

    times = df[time_col].to_numpy() if hasattr(df[time_col], 'to_numpy') else np.array(df[time_col])
    dists = df[dist_col].to_numpy() if hasattr(df[dist_col], 'to_numpy') else np.array(df[dist_col])
    
    if speed_col is not None:
        speeds = df[speed_col].to_numpy() if hasattr(df[speed_col], 'to_numpy') else np.array(df[speed_col])
        is_static = (speeds < static_thresh)
    elif static_col is not None:
        is_static = (df[static_col].to_numpy() == 1) if hasattr(df[static_col], 'to_numpy') else (np.array(df[static_col]) == 1)
    else:
        is_static = np.zeros_like(dists, dtype=bool)

    max_dist = max(1.0, float(np.nanmax(dists)))
    unique_times, time_indices = np.unique(times, return_inverse=True)
    num_times = len(unique_times)

    # Fast 2D Histogram binning in C
    time_bins = np.arange(num_times + 1) - 0.5
    dist_bins = np.linspace(0, max_dist, num_bins + 1)

    tot_matrix, _, _ = np.histogram2d(time_indices, dists, bins=[time_bins, dist_bins])
    mov_matrix, _, _ = np.histogram2d(time_indices[~is_static], dists[~is_static], bins=[time_bins, dist_bins])
    stat_matrix, _, _ = np.histogram2d(time_indices[is_static], dists[is_static], bins=[time_bins, dist_bins])

    # Phototaxis-style 2D Gaussian Kernel Smoothing across Time (Y) and Space (X)
    if smooth_time > 0 or smooth_space > 0:
        try:
            from scipy.ndimage import gaussian_filter
            tot_matrix = gaussian_filter(tot_matrix, sigma=(smooth_time, smooth_space))
            mov_matrix = gaussian_filter(mov_matrix, sigma=(smooth_time, smooth_space))
            stat_matrix = gaussian_filter(stat_matrix, sigma=(smooth_time, smooth_space))
        except ImportError:
            pass

    return unique_times, max_dist, unit_name, tot_matrix, mov_matrix, stat_matrix

def plot_kymograph(file_path: str, num_bins: int = 50, static_thresh: float = 0.5, 
                   smooth_time: float = 3.0, smooth_space: float = 2.0, interpolation: str = "bicubic"):
    """Generates and displays a continuous, silky-smooth 2D spatial-temporal Kymograph Heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required. Install via: pip install matplotlib")
        sys.exit(1)

    print(f"Processing CSV for Phototaxis-Style Smooth Heatmap: {file_path}")
    times, max_dist, unit_name, tot_matrix, mov_matrix, stat_matrix = parse_detections_csv_fast(
        file_path, num_bins=num_bins, static_thresh=static_thresh,
        smooth_time=smooth_time, smooth_space=smooth_space
    )

    has_subtypes = (mov_matrix is not None and stat_matrix is not None)
    n_panels = 3 if has_subtypes else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 7.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    base_name = os.path.basename(file_path)
    fig.suptitle(f"Continuous Chemotaxis Kymograph (Spatial-Temporal Heatmap)\n{base_name}", fontsize=13, fontweight='bold', y=0.98)

    max_t = times[-1] if len(times) > 0 else tot_matrix.shape[0]
    min_t = times[0] if len(times) > 0 else 0
    extent = [0, max_dist, min_t, max_t]

    # Panel 1: Total Density Heatmap
    im0 = axes[0].imshow(tot_matrix, aspect='auto', cmap='inferno', extent=extent, origin='lower', interpolation=interpolation)
    axes[0].set_title("Total Worm Density Field", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Time (seconds)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel(f"Distance from Pellet ({unit_name})", fontsize=11, fontweight='bold')
    cbar0 = fig.colorbar(im0, ax=axes[0], pad=0.02)
    cbar0.set_label("Worm Density", fontsize=9)

    if has_subtypes:
        # Panel 2: Moving Worm Density Heatmap
        im1 = axes[1].imshow(mov_matrix, aspect='auto', cmap='plasma', extent=extent, origin='lower', interpolation=interpolation)
        axes[1].set_title(f"Moving Worms (Speed >= {static_thresh})", fontsize=11, fontweight='bold')
        axes[1].set_xlabel(f"Distance from Pellet ({unit_name})", fontsize=11, fontweight='bold')
        cbar1 = fig.colorbar(im1, ax=axes[1], pad=0.02)
        cbar1.set_label("Worm Density", fontsize=9)

        # Panel 3: Static Worm Density Heatmap
        im2 = axes[2].imshow(stat_matrix, aspect='auto', cmap='cividis', extent=extent, origin='lower', interpolation=interpolation)
        axes[2].set_title(f"Static Worms (Speed < {static_thresh})", fontsize=11, fontweight='bold')
        axes[2].set_xlabel(f"Distance from Pellet ({unit_name})", fontsize=11, fontweight='bold')
        cbar2 = fig.colorbar(im2, ax=axes[2], pad=0.02)
        cbar2.set_label("Worm Density", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_png = os.path.splitext(file_path)[0] + "_continuous_kymograph.png"
    plt.savefig(out_png, dpi=300)
    print(f"Successfully generated and saved Continuous Heatmap Kymograph to: {out_png}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Chemotaxis Kymograph Generator")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to *_radial_detections.csv file")
    parser.add_argument("--bins", type=int, default=50, help="Spatial distance bins (default: 50)")
    parser.add_argument("--static-thresh", type=float, default=0.5, help="Speed threshold for static worms (default: 0.5)")
    parser.add_argument("--smooth-time", type=float, default=3.0, help="Temporal Gaussian smoothing sigma (default: 3.0)")
    parser.add_argument("--smooth-space", type=float, default=2.0, help="Spatial Gaussian smoothing sigma (default: 2.0)")
    parser.add_argument("--interp", type=str, default="bicubic", help="Interpolation mode: bicubic, gaussian, bilinear, nearest (default: bicubic)")
    args = parser.parse_args()

    target_path = args.csv_path or select_csv_file()

    if target_path and os.path.exists(target_path):
        plot_kymograph(
            target_path, num_bins=args.bins, static_thresh=args.static_thresh, 
            smooth_time=args.smooth_time, smooth_space=args.smooth_space, interpolation=args.interp
        )
    else:
        print("No valid CSV file selected.")
