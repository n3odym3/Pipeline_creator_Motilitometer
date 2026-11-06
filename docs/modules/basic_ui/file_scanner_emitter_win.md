# File Scanner Emitter

A batch file scanner and sequential path emitter module for DearPyGui Pipeline Creator.

## Features
- **Directory Browsing**: Select a target folder via native OS file explorer or path input.
- **Pattern Filtering**: Filter files using wildcard patterns (`*.csv`), substrings (`_radial_detections.csv`), or exact extensions.
- **Recursive Scanning**: Toggle recursive subfolder scanning with interactive collapsible folder trees.
- **Right-Click Context Menus**:
  - Right-click on a **File**: `Send Path` to emit its path to downstream processing modules.
  - Right-click on a **Folder**: `Send Folder Files` to queue and sequentially process all files in that directory.
- **Send ALL**: Sequential processing of all matched files.
- **Trigger-Synchronized Emission**: Emits the next file **only when a completion `IOTypes.TRIGGER`** is received from downstream modules.
- **Visual State Themes**:
  - 🔵 **Blue**: Currently emitting / in processing.
  - 🟢 **Green**: Processing complete.
