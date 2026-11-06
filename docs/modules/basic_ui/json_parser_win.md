# JSON Parser Module

A text processing utility that converts raw text streams into valid Python dictionary commands.

---

## Technical Specifications

- **Inputs**: `In` (Type: `TEXT`)
- **Outputs**: `Dict` (Type: `CMD_DICT`)

---

## Usage

Connect an incoming text line (e.g. from an automation stream or scanner). The node parses valid JSON notation, formats the output, and reports syntax errors in the status line.
