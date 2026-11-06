# Video Writer Module

The **Video Writer** module records incoming image streams into high-performance video files (`.mp4`, `.mkv`, `.avi`, `.mov`) with **smart real-time frame rate synchronization (Drop & Duplicate)**, multi-backend encoding (VidGear `WriteGear`, direct FFmpeg subprocess pipe, OpenCV fallback), and native support for both standard and **exotic image formats (Mono 8-bit, 12-bit / 16-bit Mono/HDR `uint16`, BGRA)**.

---

## Key Features

- **Smart Real-Time Synchronization (Drop / Duplicate)**:
  - Eliminates video distortion caused by mismatched frame rates.
  - If a camera outputs **90 FPS** and the recording is set to **30 FPS**, the module intelligently drops excess frames per time-slot ($\Delta t = 1/30$s), guaranteeing that 5 minutes of real-world recording produces a video of **exactly 5 minutes** (not 15 minutes).
  - If incoming frames are slower (e.g. 24 FPS) or delayed by heavy upstream processing, missing frames are duplicated to maintain 1:1 real-world playback speed.
  - **Pass-through mode** is also available for sequential frame-by-frame offline rendering.
- **Zero UI Freeze / High Performance**:
  - Uses an asynchronous, non-blocking bounded queue (`< 0.05 ms` per frame ingestion).
  - Background recording worker thread isolates video compression and disk I/O from the DearPyGui render loop.
- **Exotic Formats & High Bit-Depth**:
  - Standard 8-bit BGR / RGB / BGRA.
  - 8-bit Grayscale / Mono.
  - **12-bit Mono / HDR** (`uint16`, 0–4095) & **16-bit Mono / HDR** (`uint16`, 0–65535).
  - Choose between **Fast SIMD 8-bit Tone Mapping** (for universal MP4 playback) and **Native Full Bit-Depth Preservation** (using FFV1, ProRes, or HEVC Main10/Main12).
- **Multi-Backend & Hardware Acceleration**:
  - **VidGear `WriteGear`**: High-performance video compression engine.
  - **Direct FFmpeg Subprocess**: Dedicated process pipe with GPU encoding (**NVIDIA NVENC**, **Intel QuickSync**, **AMD AMF**).
  - **OpenCV VideoWriter**: Automatic fallback if FFmpeg is not installed.
- **Live Telemetry & Statistics**:
  - Real-time display of ingestion FPS vs output FPS, recorded duration, frames written, dropped frames, duplicated frames, and current file size.

---

## Technical Specifications

### Inputs

| Input Type | Description |
|---|---|
| `IOTypes.FRAME` | Standard 8-bit image frame (single `np.ndarray` or `(frame, name)`). |
| `IOTypes.FRAME12` | 12-bit HDR/Mono image frame (`uint16` array). |
| `IOTypes.FRAME16` | 16-bit HDR/Mono image frame (`uint16` array). |
| `IOTypes.MASK` | 8-bit binary/grayscale mask. |
| `IOTypes.FOLDER_PATH` | Sets the output destination directory for saved videos. |
| `IOTypes.TRIGGER` | Trigger event signal. |
| `IOTypes.CMD_DICT` | Command dictionary: `{"start": True}`, `{"record": True, "duration": 30.0}`, `{"stop": True}`, `{"pause": True}`, `{"fps": int}`, `{"folder": str}`, `{"prefix": str}`. |

### Outputs

| Output Name | Type | Description |
|---|---|---|
| `Frame` | `IOTypes.FRAME` | **Pass-through**: Re-emits the input frame/metadata downstream without blocking. |
| `File Path` | `IOTypes.FILE_PATH` | Emits a tuple `(file_path, basename)` upon recording completion. |
| `Trigger` | `IOTypes.TRIGGER` | Emits lifecycle event strings (`"RECORD_START"`, `"RECORD_PAUSE"`, `"RECORD_RESUME"`, `"RECORD_STOP"`, and `"READY"`). |
| `Status` | `IOTypes.STATUS_DICT` | Emits live metrics dictionary (FPS, duration, frames written/dropped/duplicated, file size). |

---

## UI Controls

- **REC / PAUSE / STOP** (Buttons): Controls recording lifecycle.
- **Output Dir** (Button & Label): Opens native OS folder picker to choose destination directory.
- **Prefix** (Input Text): Base filename prefix (appended with timestamp `_YYYYMMDD_HHMMSS`).
- **Target FPS** (Input Int): Target output video framerate (1 to 240 FPS).
- **Sync Mode** (Combo): Choose between `Realtime (Drop / Duplicate)` and `Pass-through (Sequential 1:1)`.
- **Codec** (Combo): Select video encoder (H.264, H.264 NVENC, H.264 QuickSync, H.265, H.265 NVENC, MJPEG, ProRes, FFV1, OpenCV MP4V).
- **Container** (Combo): Video container format (`.mp4`, `.mkv`, `.avi`, `.mov`).
- **Bit-Depth Mode** (Combo): Choose between `8-bit Tone-mapped` and `Native Full Bit-Depth`.
- **Bitrate (Mbps)** (Slider Float): Video encoding target bitrate.
- **Live Telemetry** (Collapsing Header): Live monitors showing Ingest FPS, Output FPS, Duration, Frames, Drops, Duplications, and File Size.
