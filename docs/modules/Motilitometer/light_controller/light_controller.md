# Light Controller

The **Light Controller** module controls multi-channel LED illumination hardware (Red, Green, Blue, White, and Infra-Red) over a serial interface.

## Overview

- **Serial Protocol**: Communicates via standard PySerial at 115200 baud (configurable).
- **JSON Formatting**: Emits structured JSON commands after calibration (`{"r": ..., "g": ..., "b": ..., "w": ..., "ir": ...}`).
- **Two-Segment Calibration**: Converts input setpoints in $\mu\text{E}$ ($\mu\text{mol photons m}^{-2} \text{s}^{-1}$) into hardware PWM integer levels using low-intensity correction and high-intensity linear equations.
- **Visual Feedback (Pastille)**: Real-time RGB + White mixed color preview badge displaying current illumination hue and computed total intensity.
- **Pulse Mode**: Fast hardware flash/pulse trigger with configurable millisecond duration.

---

## Inputs & Outputs

### Inputs

| Type | Format / Payload | Description |
| :--- | :--- | :--- |
| `CMD_DICT` | `{"r": 100, "g": 50, "b": 0, "w": 0, "ir": 0}` | Sets individual channel intensities and transmits calibrated command. |
| `CMD_DICT` | `{"light": {"r": 100, ...}}` | Alternative dictionary payload format from sequence processors. |
| `CMD_DICT` | `{"pulse": {"pulse": 150, "r": 200, ...}}` | Triggers a timed pulse with channel intensities. |
| `TEXT` / `STRING` | `"R100G50"`, `"R100"`, `"r 100 g 50 b 20 w 10 ir 500"` | Compact sequence command string. Unspecified channels default to 0. |
| `TRIGGER` | Trigger event | Triggers a pulse based on the active pulse mode configuration. |

### Outputs

| Output Name | Type | Description |
| :--- | :--- | :--- |
| `Status` | `STATUS_DICT` | Emits current state, active channel $\mu\text{E}$ levels, and calculated PWM dict. |
| `received_data` | `TEXT` | Raw incoming serial response text from the hardware controller. |

---

## Calibration Model

The $\mu\text{E}$ to PWM conversion uses a piecewise linear model:

$$\text{PWM}(\mu\text{E}) = \begin{cases} 
0 & \text{if } \mu\text{E} \le 0 \\
\frac{\mu\text{E} - p_{\text{low}}}{m_{\text{low}}} & \text{if } 0 < \mu\text{E} \le \text{threshold}_{\text{low}} \\
\frac{\mu\text{E} - p}{m} & \text{if } \mu\text{E} > \text{threshold}_{\text{low}}
\end{cases}$$

Default parameters:
- `max`: $500\ \mu\text{E}$
- `low_corr`: $15\ \mu\text{E}$
- `low_m`: $0.2$, `low_p`: $-5.0$
- `m`: $0.4975$, `p`: $-48.897$

---

## Sequence Processor Integration

Commands can be written in Sequence Processor workflows using compact channel notation:

```text
R100, 5s, G50, 5s, R100G50B20, 2s, R0G0B0W0IR0
```

- Writing `R100` sets Red to 100 and turns off all other channels (Green, Blue, White, IR $\to$ 0).
- Writing `R100G50` sets Red to 100, Green to 50, and turns off Blue, White, and IR.
- Writing `IR500` sets Infra-Red to 500 and turns off RGBW.
