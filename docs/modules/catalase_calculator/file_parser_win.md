# File Parser Module

The **File Parser** module loads 96-well microplate reader raw absorbance datasets, parses kinetics profiles, and computes slopes via linear regression on a selected range of time points.

---

## Technical Specifications

- **Inputs**: `In` (Type: `FILE_PATH` for receiving automatic triggers or file locations)
- **Outputs**:
  - `Raw Data` (Type: `SAMPLE` for plotting the kinetics curves of the selected well)
  - `Regression` (Type: `SAMPLE` for plotting the fitted regression line)
  - `Slopes` (Type: `CMD_DICT` mapping well-coordinates to computed slopes)

---

## Key Features & Controls

- **Interactive 96-Well Grid**: Select any cell (A1 to H12) to visualize and analyze its specific absorbance profile.
- **Manual Regression Range**: Set start and end indices using the numeric input fields to manually fit slopes.
- **Auto-Selection Mode**: Finds the contiguous sequence of points yielding the highest coefficient of determination ($R^2$) given a minimum number of points.
- **Real-Time Statistics**: Displays computed slope, intercept, $R^2$, and status messages for the active well.
