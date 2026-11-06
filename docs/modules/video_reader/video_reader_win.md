# Video Reader Module

The **Video Reader** module scans local directories for video files (`.mp4`), provides an interactive playback and frame scrubbing interface, and streams frames and events downstream for image processing, computer vision, or recording pipelines.

---

## Features

- **Folder Scanning & File Table**: Browse and list all `.mp4` video files in a selected directory with table view.
- **Playback Controls**: Standard media playback with **Play** (`▶`), **Pause** (`⏸`), and **Stop** (`⏹`) buttons powered by OpenCV and multi-threaded playback.
- **Frame Scrubber / Seek Bar**: Interactive timeline slider allowing scrubbing and random access seeking to any frame, with instant visual feedback when paused.
- **Adjustable FPS**: Configurable playback rate (FPS) to speed up analysis or slow down video streams.
- **Context Menu File Path Dispatch**: Right-click on any video in the list to send its full path and base name directly through the `File Path` output.
- **Trigger Event Signals**: Automatically emits `"START"` and `"STOP"` event pulses on playback start and end/stop.
- **Remote / Automated Control**: Accept command dictionaries via `CMD_DICT` for automated batch processing (`start`, `stop`, `pause`, `browse`, `fps`).

---

## Technical Specifications

### Inputs

| Input Type | Description |
|---|---|
| `IOTypes.FOLDER_PATH` / `IOTypes.FILE_PATH` | Directory path string to scan for `.mp4` video files. |
| `IOTypes.CMD_DICT` | Command dictionary supporting actions: `{"start": True}`, `{"stop": True}`, `{"pause": True}`, `{"browse": True}`, `{"fps": int}`. |

### Outputs

| Output Name | Type | Description |
|---|---|---|
| `Frame` | `IOTypes.FRAME` | Emits a tuple `(frame, video_name)` containing the raw OpenCV image array and source video basename. |
| `Trigger` | `IOTypes.TRIGGER` | Emits lifecycle event strings (`"START"` when playback begins, `"STOP"` when finished or stopped). |
| `File Path` | `IOTypes.FILE_PATH` | Emits a tuple `(file_path, basename)` on demand via right-click context menu. |

---

## UI Controls

- **Browse** (Button): Opens native OS folder picker to load a directory of video files.
- **Folder Label**: Displays current loaded folder path.
- **Play / Pause / Stop** (Buttons): Controls video playback loop.
- **FPS** (Input Int): Sets playback target frame rate.
- **Timeline Slider**: Draggable seek bar reflecting current frame position and allowing scrubbing.
- **Video Table**: Displays listed videos. Left-click loads and selects the video for playback; right-click opens the context menu to send the file path downstream.
