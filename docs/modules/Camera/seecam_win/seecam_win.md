# seeCam Camera Module

This module controls **e-con Systems see3CAM** series cameras, including **See3CAM_20CUG (CU20G)** and **See3CAM_CU55M (CU55M)**. It is designed to capture and stream dual-mode outputs, delivering standard 8-bit RGB/BGR frames as well as high-depth raw grayscale frames (Y8, Y12, Y16).

---

## Technical Specifications

- **Inputs**: `In` (Type: `CMD_DICT` for remote trigger commands `{"trigger_cam": True}`)
- **Outputs**:
  - `8bit RGB` (Type: `FRAME` 8-bit color/grayscale image)
  - `16bit gray` (Type: `FRAME16` raw high-depth image, e.g. 12-bit / 16-bit / 8-bit)

---

## Key Features & Controls

- **Supported Models**:
  - **See3CAM_20CUG (CU20G)**: 2MP global shutter monochrome camera (OV2311). Supports **Y8** (8-bit) and **Y16** (10-bit raw in 16-bit word).
  - **See3CAM_CU55M (CU55M)**: 5MP monochrome camera (AR0521). Supports **Y8** (8-bit) and **Y12** (packed 12-bit, unpacked to 16-bit [0..4095]).
  - **Generic See3CAM**: Dynamically discovers all stream capabilities via DirectShow.
- **Resolution & Pixel Format Selection**:
  - Filter streams by format: `All`, `Y8`, `Y12`, `Y16`.
  - Resolution dropdown lists matching capabilities with theoretical frame rates (e.g. `2592x1944 @ 30 FPS (Y12)` or `1920x1080 @ 60 FPS (Y16)`).
  - Seamless on-the-fly stream reconfiguration.
- **Adaptive 8-bit Conversion**:
  - `Auto`: Automatically adapts bit-shifting based on the detected camera model and active format:
    - `Y8`: Pass-through.
    - `CU55M Y12`: Bit-shift `>> 4` (12-bit to 8-bit).
    - `CU20G Y16`: Bit-shift `>> 2` (10-bit raw in 16-bit container to 8-bit).
  - `Normalize`: Dynamic min-max contrast stretching.
  - Manual shifts: `8-bit`, `10-bit`, `12-bit`, `16-bit`.
- **Acquisition Modes**:
  - `Continuous`: Real-time streaming with live FPS counter.
  - `Trigger`: On-demand capture via the `Trigger Frame` UI button or external `CMD_DICT` trigger input.
- **Hardware & UVC Controls**:
  - `Open Camera Settings`: Launches the native hardware/driver DirectShow property page.
  - Gain and Exposure sliders with dynamic ranges queried directly from the device.
  - Auto-exposure loop with live peak intensity feedback.
