# IDS Camera Module

This module controls **IDS Imaging Development Systems** industrial cameras. It is designed for high-resolution imaging and supports advanced trigger modes, pixel clocks, and dark reference correction.

---

## Technical Specifications

- **Inputs**: `In` (Type: `CMD_DICT` for remote adjustments)
- **Outputs**: `12bit frame` (Type: `FRAME12` 12-bit raw image)

---

## Key Features & Controls

- **Pixel Clock**: Configure the clock rate of the sensor to balance frame rates and noise levels.
- **Exposure & Shutter**: Manually set exposure times or adjust shutter levels.
- **Trigger Modes**: Toggle between continuous streaming and hardware/software trigger modes.
- **Dark Reference Correction**: Accumulate and subtract dark reference frames in real time to correct sensor hot pixels and thermal noise.
