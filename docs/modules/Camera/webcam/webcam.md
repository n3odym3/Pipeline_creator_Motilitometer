# Webcam Module

This module controls **generic USB Webcams** via OpenCV's grabber interface. It is ideal for rapid testing, demo setups, or general BGR video acquisitions.

---

## Technical Specifications

- **Inputs**: `In` (Type: `CMD_DICT` for remote adjustments)
- **Outputs**: `Color Frame (BGR)` (Type: `FRAME` 8-bit color image)

---

## Key Features & Controls

- **Auto-Discovery**: Click **Refresh** to query all connected USB video grabbers and select your camera from the dropdown.
- **Dynamic Resolution & Capability Scanning**: Automatically probes the camera hardware via DirectShow to list all physically supported resolutions and their maximum theoretical frame rates (e.g. `1920x1080 (max 60 FPS)`, `1280x720 (max 60 FPS)`, `640x480 (max 90 FPS)`).
- **Theoretical & Live FPS Monitoring**: Displays both the camera driver's theoretical FPS and the actual live measured acquisition framerate.
- **Standard Controls**: Open native hardware UVC settings dialog.
- **Trigger Modes**: Grab continuous video frames or wait for downstream software triggers.
