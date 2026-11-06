# OBS Controller Module

The **OBS Controller** module provides an exhaustive, real-time control center and telemetry monitor for **OBS Studio** via the official **OBS WebSocket Protocol v5** (natively included in OBS Studio 28+).

It allows Pipeline Creator workflows to automate scene transitions, stream and record sessions, trigger instant replay buffers, toggle virtual cameras, adjust audio mixer channels, inspect and toggle scene sources, and capture live screenshots directly into the image processing pipeline (`IOTypes.FRAME`).

---

## Key Features

- **Official OBS WebSocket v5 Protocol**:
  - Full support for secure challenge/salt SHA-256 authentication.
  - Automatic reconnection and non-blocking asynchronous event handling.
- **Master Output Operations**:
  - **Streaming**: Start, Stop, Toggle, with live stream duration and bitrate monitors.
  - **Recording**: Start, Pause, Resume, Stop, Toggle, with live timecode counters.
  - **Replay Buffer**: Start, Stop, Save Replay Buffer to disk with one click or automated trigger.
  - **Virtual Camera**: Start, Stop, Toggle OBS Virtual Camera.
  - **Studio Mode**: Toggle Studio Mode (Preview vs Program) and execute transitions.
- **Dynamic Scenes & Studio Transitions**:
  - One-click scene switcher with active program scene indicator.
  - Studio Mode transition trigger with configurable transition types (Cut, Fade, Slide, Stinger, etc.) and duration (ms).
- **Scene Sources & Inspector**:
  - Lists all sources in the active scene with live visibility (enable/disable) toggles.
  - **Instant Snapshot**: Captures full-resolution frames from any source or scene and emits them downstream as `IOTypes.FRAME`.
- **Audio Mixer**:
  - Real-time volume sliders (0.0 to 1.0) and instant mute/unmute buttons for every detected audio track (Mic/Aux, Desktop Audio, Media Sources).
- **Live System Telemetry & Performance**:
  - Real-time monitors for Render FPS, CPU Usage (%), RAM (MB), Free Disk Space (GB), Frame Render Time (ms), and Render / Output dropped/skipped frame rates.
- **Profiles & Collections**:
  - Dynamic switching between OBS Profiles and Scene Collections.
  - Raw JSON RPC Request executor for custom advanced commands.

---

## Technical Specifications

### Inputs

| Input Type | Description |
|---|---|
| `IOTypes.CMD_DICT` | Command dictionary for automation: <br>• `{"command": "set_scene", "scene": "Game"}`<br>• `{"command": "start_record"}`, `{"command": "stop_record"}`<br>• `{"command": "start_stream"}`, `{"command": "stop_stream"}`<br>• `{"command": "save_replay_buffer"}`<br>• `{"command": "toggle_virtual_cam"}`<br>• `{"command": "transition"}`<br>• `{"command": "toggle_mute", "source": "Mic/Aux"}`<br>• `{"command": "set_volume", "source": "Desktop Audio", "volume": 0.8}`<br>• `{"command": "set_visibility", "scene": "Main", "item_id": 1, "visible": True}`<br>• `{"command": "take_screenshot", "source": "Main"}`<br>• `{"command": "custom_request", "type": "...", "data": {...}}` |
| `IOTypes.TEXT` | Direct scene name to switch to (e.g. `"Intro"`, `"Main"`, `"Ending"`), or action verbs (`"START_REC"`, `"STOP_REC"`, `"STREAM"`, `"REPLAY"`, `"SNAPSHOT"`, `"MUTE"`). |
| `IOTypes.TRIGGER` | Trigger signals (`"START"`, `"STOP"`, `"SNAPSHOT"`, `"RECORD"`). |

### Outputs

| Output Name | Type | Description |
|---|---|---|
| `Status` | `IOTypes.STATUS_DICT` | Full telemetry dictionary: `connected`, `obs_version`, `current_scene`, `preview_scene`, `is_streaming`, `is_recording`, `is_replay_buffer_active`, `stats` (FPS, CPU, RAM, disk space, drops), and `audio_inputs`. |
| `Trigger` | `IOTypes.TRIGGER` | Emits lifecycle trigger strings (`"OBS_CONNECTED"`, `"OBS_DISCONNECTED"`, `"SCENE_CHANGED:<name>"`, `"RecordStateChanged"`, `"StreamStateChanged"`, `"REPLAY_BUFFER_SAVED"`, `"SCREENSHOT_TAKEN:<name>"`). |
| `Frame` | `IOTypes.FRAME` | Emits captured screenshot frames `(np.ndarray, source_name)` into the visual pipeline. |
| `Text` | `IOTypes.TEXT` | Emits current active program scene name on scene changes. |

---

## Configuration & Setup

1. In OBS Studio, open **Tools > WebSocket Server Settings**.
2. Check **Enable WebSocket server**.
3. Default Server Port is `4455`.
4. (Optional) Set a password and enter the same password in the module.
5. Click **Connect** in Pipeline Creator.
