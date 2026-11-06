# Boxplot Module

## Description
The **Boxplot & Statistical Time Distribution** module calculates and visualizes statistical metrics over time (or any grouped independent variable X). For each unique time step or time bin, it computes:
- **Mean** ($\mu$) & **Standard Deviation** ($\sigma$)
- **Standard Error of the Mean (SEM)**
- **Median** ($Q_2$), **25th Percentile** ($Q_1$), and **75th Percentile** ($Q_3$) (Interquartile Range - IQR)
- **Minimum & Maximum** (Whiskers)
- **Sample Count** ($N$)

## Features
- **Multiple Visualization Modes**:
  - `Mean +/- Std Ribbon`: Central mean curve with translucent $\pm 1\sigma$ deviation envelope.
  - `Boxplot (IQR + Whiskers)`: Standard box-and-whisker plot (Q25 to Q75 shaded box, median line, and min/max whiskers).
  - `Min/Max Envelope`: Shaded band between extreme minimum and maximum observed values.
  - `Scatter + Mean`: All raw individual points displayed with overlaid mean trend line.
- **Configurable Options**:
  - `Mean`, `+/-Std`, `Min/Max`, `Scatter` individual toggle checkboxes.
  - `Time Binning / Downsample`: Groups dense frame sequences into a user-defined number of bins (e.g. 50 bins).
  - `Autoscale` and `Clear Plot`.
  - `Export CSV`: Exports a complete tabular summary of all statistical metrics across time.

## Inputs
- `IOTypes.SAMPLE` (Dictionary with `x` [time] and `y` [values, e.g. speed of all worms]).
- `IOTypes.DATALIST`, `IOTypes.CMD_DICT`.

## Outputs
- **Summary Sample**: `IOTypes.SAMPLE` (Emits computed statistics dictionary for downstream logging or aggregators).
- **Datalist**: `IOTypes.DATALIST`.
- **CMD**: `IOTypes.CMD_DICT`.
