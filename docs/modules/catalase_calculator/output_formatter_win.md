# Output Formatter Module

The **Output Formatter** is a 96-well plate viewer designed to compile, display, and export computed slopes into standard spreadsheet structures.

---

## Technical Specifications

- **Inputs**: `In` (Type: `CMD_DICT` containing slope mappings)
- **Outputs**: None.

---

## Key Features & Controls

- **Interactive 96-Well Matrix**: Displays computed slopes in a structured grid matching the experimental microplate layout.
- **Dynamic Highlights**: Click individual cells to inspect values or apply custom styles.
- **Grid Management**:
  - **Clear Selected**: Resets the value of the highlighted cell.
  - **Clear All**: Resets the entire grid.
  - **Save CSV**: Opens a local save dialog to write the grid values into a standard `.csv` spreadsheet table.
