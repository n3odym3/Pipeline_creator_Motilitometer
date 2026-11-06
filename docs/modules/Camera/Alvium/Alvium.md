# Alvium Camera Module

This module controls **Allied Vision Alvium** industrial cameras via the Vimba/Veye API. It supports exposure control, gain adjustments, white balance tuning, and flat-field corrections.

---

## Technical Specifications

- **Inputs**: `In` (Type: `CMD_DICT` for remote adjustments)
- **Outputs**: `Frame` (Type: `FRAME` 8-bit image)

---

## Key Features & Controls

- **Exposure & Gain**: Adjust sensor integration time and gain levels dynamically.
- **White Balance**: Tune Red and Blue channels for correct color rendering.
- **ROI & Binning**: Define pixel binning factors and a Region of Interest (ROI) width and height to optimize frame rates.
- **Flat-Field Correction (FFC)**: Capture and apply mono or RGB calibration matrices to correct sensor illumination inequalities.
