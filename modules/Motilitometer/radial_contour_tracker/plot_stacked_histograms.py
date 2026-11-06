"""
Stacked Ridgeline / Waterfall Chemotaxis Population Plotter
============================================================
Reads raw detection data (*_radial_detections.csv) from the Radial Contour Tracker 
and generates 3D Ridgeline (Waterfall) plots where each timepoint's population distribution 
histogram is vertically offset upwards to create a 3D depth effect.

Usage:
------
1. Interactive File Selection:
   python plot_stacked_histograms.py

2. Command-Line Options:
   python plot_stacked_histograms.py "path/to/my_video_radial_detections.csv" --bins 20 --time-steps 12 --overlap 1.5
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

def parse_detections(file_path: str, num_bins: int = 30):
    """Fast parsing and 2D spatial-temporal matrix generation."""
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
    unique_times, time_indices = np.unique(times, return_inverse=True)
    num_times = len(unique_times)

    time_bins = np.arange(num_times + 1) - 0.5
    dist_bins = np.linspace(0, max_dist, num_bins + 1)

    tot_matrix, _, _ = np.histogram2d(time_indices, dists, bins=[time_bins, dist_bins])

    return unique_times, dist_bins, unit_name, tot_matrix

def plot_ridgeline_population(file_path: str, num_bins: int = 30, time_steps: int = 12, overlap: float = 1.5):
    """
    Generates a 3D Ridgeline / Waterfall depth plot where histograms at successive timepoints 
    are offset vertically upwards with 3D depth overlapping shading.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("Error: matplotlib is required. Install via: pip install matplotlib")
        sys.exit(1)

    print(f"Parsing detection data for Ridgeline Waterfall plot: {file_path}")
    times, dist_bins, unit_name, tot_matrix = parse_detections(file_path, num_bins=num_bins)

    # Smooth spatial profile for clean Ridgeline curves
    try:
        from scipy.ndimage import gaussian_filter1d, gaussian_filter
        tot_matrix_smooth = gaussian_filter(tot_matrix, sigma=(2.0, 2.2))
    except ImportError:
        tot_matrix_smooth = tot_matrix

    num_frames = len(times)
    sample_indices = np.linspace(0, num_frames - 1, time_steps, dtype=int)
    bin_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), gridspec_kw={'height_ratios': [1.6, 1]})
    base_name = os.path.basename(file_path)
    fig.suptitle(f"Radial Chemotaxis Ridgeline Waterfall & Stacked Distribution\n{base_name}", fontsize=13, fontweight='bold', y=0.98)

    # -------------------------------------------------------------------------
    # Panel 1: 3D Ridgeline / Waterfall Depth Plot (Histograms Shifted Upwards)
    # -------------------------------------------------------------------------
    colors = cm.get_cmap('viridis', time_steps)(np.linspace(0, 1, time_steps))

    # Calculate vertical spacing offset
    max_count = np.max([tot_matrix_smooth[idx, :] for idx in sample_indices]) or 1.0
    vertical_offset = (max_count * 0.6) / overlap

    y_tick_positions = []
    y_tick_labels = []

    for i, idx in enumerate(sample_indices):
        t_sec = times[idx]
        profile = tot_matrix_smooth[idx, :]
        
        y_base = i * vertical_offset
        y_top = y_base + profile

        y_tick_positions.append(y_base)
        if t_sec >= 60:
            y_tick_labels.append(f"{t_sec / 60:.1f} min")
        else:
            y_tick_labels.append(f"{int(t_sec)} s")

        # Solid white background fill for complete occlusion of back curves
        ax1.fill_between(bin_centers, y_base, y_top, color='white', alpha=1.0, zorder=i * 2)
        # Rich solid fill inside the histogram curve
        ax1.fill_between(bin_centers, y_base, y_top, color=colors[i], alpha=0.90, zorder=i * 2 + 1)
        # Crisp stroke line on top
        ax1.plot(bin_centers, y_top, color='white', linewidth=1.0, zorder=i * 2 + 2)
        ax1.plot(bin_centers, y_top, color=colors[i], linewidth=2.2, zorder=i * 2 + 3)
        # Baseline reference line
        ax1.plot([bin_centers[0], bin_centers[-1]], [y_base, y_base], color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=i * 2)

    ax1.set_title("1. 3D Ridgeline Waterfall Plot (Population Histograms Shifted Upwards over Time)", fontsize=11, fontweight='bold')
    ax1.set_xlabel(f"Distance from Pellet ({unit_name})", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Time Evolution", fontsize=10, fontweight='bold')
    ax1.set_yticks(y_tick_positions)
    ax1.set_yticklabels(y_tick_labels, fontsize=9)
    ax1.set_xlim(0, dist_bins[-1])
    ax1.grid(True, axis='x', linestyle='--', alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 2: Stacked Population Area Evolution over Time
    # -------------------------------------------------------------------------
    n_stack_bins = 8
    stack_dist_bins = np.linspace(0, dist_bins[-1], n_stack_bins + 1)
    stack_time_bins = np.arange(num_frames + 1) - 0.5
    
    # Calculate matrix for stacked area plot
    dists_raw = np.linspace(0, dist_bins[-1], tot_matrix.shape[1])
    tot_stack_matrix = np.zeros((num_frames, n_stack_bins))
    for f in range(num_frames):
        tot_stack_matrix[f, :], _ = np.histogram(dists_raw, weights=tot_matrix[f, :], bins=stack_dist_bins)

    stack_colors = cm.get_cmap('plasma_r', n_stack_bins)(np.linspace(0, 1, n_stack_bins))
    fmt = lambda v: f"{int(v)}" if v == int(v) else f"{v:.1f}"
    stack_labels = [f"{fmt(stack_dist_bins[b])}-{fmt(stack_dist_bins[b+1])}{unit_name}" for b in range(n_stack_bins)]

    ax2.stackplot(times, tot_stack_matrix.T, labels=stack_labels, colors=stack_colors, alpha=0.85)
    ax2.set_title("2. Stacked Population Zone Distribution over Time", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Worm Count in Zone", fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', title=f"Distance Zones ({unit_name})", fontsize=7.5, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xlim(times[0], times[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = os.path.splitext(file_path)[0] + "_ridgeline_waterfall.png"
    plt.savefig(out_png, dpi=300)
    print(f"Successfully generated and saved 3D Ridgeline Waterfall figure to: {out_png}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacked Ridgeline / Waterfall Population Plotter")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to *_radial_detections.csv file")
    parser.add_argument("--bins", type=int, default=30, help="Number of spatial distance bins (default: 30)")
    parser.add_argument("--time-steps", type=int, default=12, help="Number of superimposed time profiles (default: 12)")
    parser.add_argument("--overlap", type=float, default=1.5, help="3D depth vertical overlap factor (default: 1.5)")
    args = parser.parse_args()

    target_path = args.csv_path or select_csv_file()

    if target_path and os.path.exists(target_path):
        plot_ridgeline_population(target_path, num_bins=args.bins, time_steps=args.time_steps, overlap=args.overlap)
    else:
        print("No valid CSV file selected.")
