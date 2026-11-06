from __future__ import annotations
import re
import inspect
from collections.abc import Callable

command_registry: list[tuple[re.Pattern[str], Callable[[str], dict[str, object] | None], str]] = []

def register_command(pattern: str) -> Callable[[Callable], Callable]:
	def decorator(func: Callable) -> Callable:
		regex = re.compile(pattern)
		sig = inspect.signature(func)
		n_args = len(sig.parameters)

		if n_args == 1:
			def wrapper(cmd: str) -> dict[str, object] | None:
				match = regex.fullmatch(cmd)
				return func(cmd) if match else None
			
			try:
				doc = func("").get("description", "No description")
			except Exception:
				doc = "No description"

		elif n_args == 2:
			def wrapper(cmd: str) -> dict[str, object] | None:
				match = regex.fullmatch(cmd)
				return func(cmd, match) if match else None
			
			try:
				doc = func("", None).get("description", "No description")
			except Exception:
				doc = "No description"
		else:
			raise TypeError(f"Function '{func.__name__}' must take 1 or 2 parameters.")

		command_registry.append((regex, wrapper, doc))
		return func
	return decorator

@register_command(r"(\d+)(ms|s|m|h)")
def cmd_delay(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Wait [value][unit]"}

	duration = int(match.group(1))
	unit = match.group(2)
	duration_s = {
		"ms": duration / 1000,
		"s": duration,
		"m": duration * 60,
		"h": duration * 3600,
	}[unit]
	
	return {
		"valid": True,
		"description": f"Wait {duration}{unit}",
		"duration": duration_s,
		"cmd": {
            "wait": duration_s
        }
	}

@register_command(r"volt(\d+(?:\.\d+)?)")
def cmd_voltage(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	min_val = 0
	max_val = 60

	if match is None:
		return {"description": f"Set PSU to [{min_val}-{max_val}]V"}
	
	volt = float(match.group(1))

	if not (min_val <= volt <= max_val):
		return {
			"valid": False,
			"description": f"Voltage {volt}V out of bounds ({min_val}-{max_val})V",
			"duration": 0,
			"error": "Value must be between 0 and 60",
			"cmd": {}
		}
	
	return {
		"valid": True,
		"description": f"Set PSU to {volt}V",
		"duration": 1,
		"cmd": {
			"volt" : volt
        },
	}

@register_command(r"amp(\d+(?:\.\d+)?)")
def cmd_current(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	min_val = 0
	max_val = 5

	if match is None:
		return {"description": f"Set PSU to [{min_val}-{max_val}]A"}
	
	amp = float(match.group(1))

	if not (min_val <= amp <= max_val):
		return {
			"valid": False,
			"description": f"Current {amp}A out of bounds ({min_val}-{max_val})",
			"duration": 0,
			"error": f"Value must be between {min_val} and {max_val}",
			"cmd": {}
		}
	
	return {
		"valid": True,
		"description": f"Set PSU to {amp}A",
		"duration": 1,
		"cmd": {
			"amp" : amp
        }
	}

@register_command(r"[aA](\d+)")
def cmd_act(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	min_val = 0
	max_val = 2500

	if match is None:
		return {"description": f"Set act to [{min_val}-{max_val}]"}

	intensity = int(match.group(1))
	if not (min_val <= intensity <= max_val):
		return {
			"valid": False,
			"description": f"Act intensity {intensity} out of bounds ({min_val}-{max_val})",
			"duration": 0,
			"error": f"Value must be between {min_val} and {max_val}",
			"cmd": {}
		}
	
	return {
		"valid": True,
		"description": f"Set act to {intensity}",
		"duration": 1,
		"cmd": {
			"act" : intensity
        }
	}

@register_command(r"(?i)det(\d+)")
def cmd_det(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Set det to [0-512]"}
	min_val = 0
	max_val = 512
	pwm = int(match.group(1))
	if not (min_val <= pwm <= max_val):
		return {
			"valid": False,
			"description": f"Det PWM {pwm} out of bounds ({min_val}-{max_val})",
			"duration": 0,
			"error": f"Value must be between {min_val} and {max_val}",
			"cmd": {}
		}
	
	return {
		"valid": True,
		"description": f"Set det to {pwm}",
		"duration": 1,
		"cmd": {
			"send" : f"det {pwm}"
        }
	}

@register_command(r"(?i)fr(\d+)")
def cmd_far_red(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Set Far Red to [0-512]"}
	
	intensity = int(match.group(1))
	return {
		"valid": True,
		"description": f"Set Far Red to {intensity}",
		"duration": 1,
		"cmd": {
			"send" : f"fr {intensity}"
        }
	}

@register_command(r"(?i)od(\d+)\s+(\d+)")
def cmd_offset_detection(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "ODet of Xµs after Yµs flash"}

	det_time = int(match.group(1))
	flash_time  = int(match.group(2))

	return {
		"valid": True,
		"description": f"ODet of {det_time}µs after {flash_time}µs flash",
		"duration": 1,
		"captures": [f"OD_{flash_time}"],
		"cmd": {
			"send": f"odet {det_time} {flash_time}"
		}
	}

@register_command(r"(?i)^v(\d+)(ms|s|m|h)$")
def cmd_video(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Record video for [time][unit] (e.g. V30s, V1m, V100ms)"}

	duration_val = int(match.group(1))
	unit = match.group(2).lower()
	duration_s = {
		"ms": duration_val / 1000.0,
		"s": float(duration_val),
		"m": duration_val * 60.0,
		"h": duration_val * 3600.0,
	}[unit]

	return {
		"valid": True,
		"description": f"Record Video for {duration_val}{unit}",
		"duration": 0.0,
		"timeout": 10.0,
		"wait_feedback": True,
		"cmd": {
			"record_video": True,
			"start": True,
			"duration": duration_s,
		}
	}

@register_command(r"(?i)^v$")
def cmd_fvfm(cmd: str) -> dict[str, object]:
	return {
		"valid": True,
		"description": "Fluo measurement",
		"duration": 1,
		"captures": ["F0", "Fm"],
		"cmd": {
			"fvfm": True
		}
	}

@register_command(r"(?i)^d(?:\s*(\d+)(?:\s+(\d+))?)?\s*$")
def cmd_d(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Detection: D or D [pdur] [ddim]"}

	from core.app_state import app_state

	pdur_val = match.group(1)
	ddim_val = match.group(2)

	det_state = app_state.get("detector")
	default_pdur = int(det_state) if det_state is not None and int(det_state) > 0 else 250
	pdur = int(pdur_val) if pdur_val is not None else default_pdur

	ddim_state = app_state.get("ddim")
	default_ddim = int(ddim_state) if ddim_state is not None and int(ddim_state) >= 0 else 512
	ddim = int(ddim_val) if ddim_val is not None else default_ddim

	return {
		"valid": True,
		"description": f"Detection Dur:{pdur} Dim:{ddim}",
		"duration": 1,
		"captures": ["F0"],
		"cmd": {
			"send": f"detection {pdur} {ddim}",
			"detection": True,
			"pdur": pdur,
			"ddim": ddim,
		}
	}

@register_command(r"(?i)t(\d+)?")
def cmd_trigger(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Trigger cam with optional expos T[time]"}
	
	exposure = match.group(1)
	description = "Trigger camera"
	cmd_dict = {"trigger_cam": True}
	
	if exposure is not None:
		cmd_dict["exposure"] = int(exposure)
		description += f" (Expos: {exposure})"
		
	return {
		"valid": True,
		"description": description,
		"duration": 1,
		"captures": ["Photo"],
		"cmd": cmd_dict
	}

@register_command(r"(?i)^g(.+)$")
def cmd_group(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Set group name GNAME"}
	
	from .sequence_processor import sequence_processor
	raw_cmd = sequence_processor.current_raw_cmd
	raw_match = re.match(r'(?i)^g\s*(.+)$', raw_cmd.strip()) if raw_cmd else None
	if raw_match:
		group_name = raw_match.group(1).strip()
	else:
		group_name = match.group(1).strip()

	return {
		"valid": True,
		"description": f"Set group to: {group_name}",
		"duration": 0,
		"cmd": {
			"group": group_name,
			"prefix": group_name,
		}
	}

@register_command(r"(?i)ojip\s+(\d+)\s+(\d+)\s+(\d+)")
def cmd_ojip(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "OJIP [det_time] [det_int] [pulse_time]"}
	
	det_time = int(match.group(1))
	det_int = int(match.group(2))
	pulse_time = int(match.group(3))

	return {
		"valid": True,
		"description": f"OJIP DTime:{det_time} DInt:{det_int} PTime:{pulse_time}",
		"duration": pulse_time / 1000000.0,
		"captures": [f"OJIP_{pulse_time}"],
		"cmd": {
			"send": f"ojip {det_time} {det_int} {pulse_time}"
		}
	}

@register_command(r"(?i)pdet\s+(\d+)\s+(\d+)\s+(\d+)")
def cmd_pdet(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Pulsed Detection: pdet [det_t] [rep] [int_dur]"}
	
	det_t = int(match.group(1))
	rep = int(match.group(2))
	int_dur = int(match.group(3))

	return {
		"valid": True,
		"description": f"PulseDet DTime:{det_t} Rep:{rep} IDur:{int_dur}",
		"duration": (rep * int_dur) / 1000000.0,
		"captures": [f"OJIP_{(i + 1) * int_dur}" for i in range(rep)],
		"cmd": {
			"send": f"pulsedet {det_t} {rep} {int_dur}"
		}
	}

@register_command(r"(?i)^\s*(?:ref\s*(\d+)?|r)\s*$")
def cmd_ref(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Capture Dark Reference: R, ref, or ref [exposure]"}
	
	exposure = match.group(1)
	description = "Capture Dark Reference"
	cmd_dict = {"fluo_ref": True}
	
	if exposure is not None:
		cmd_dict["exposure"] = int(exposure)
		description += f" (Expos: {exposure})"
		
	return {
		"valid": True,
		"description": description,
		"duration": 5.0, 
		"cmd": cmd_dict
	}

@register_command(r"(?i)w(?:hite)?2\s*(\d+)")
def cmd_white2(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Set White Light 2 to [0-512]"}
	
	intensity = int(match.group(1))
	min_val = 0
	max_val = 512
	if not (min_val <= intensity <= max_val):
		return {
			"valid": False,
			"description": f"White 2 intensity {intensity} out of bounds ({min_val}-{max_val})",
			"duration": 0,
			"error": f"Value must be between {min_val} and {max_val}",
			"cmd": {}
		}
	
	return {
		"valid": True,
		"description": f"Set White Light 2 to {intensity}",
		"duration": 1,
		"cmd": {
			"white2" : intensity
        }
	}

@register_command(r"(?i)w(?:hite)?\s*(\d+)")
def cmd_white(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Set White Light to [0-512]"}
	
	intensity = int(match.group(1))
	min_val = 0
	max_val = 512
	if not (min_val <= intensity <= max_val):
		return {
			"valid": False,
			"description": f"White intensity {intensity} out of bounds ({min_val}-{max_val})",
			"duration": 0,
			"error": f"Value must be between {min_val} and {max_val}",
			"cmd": {}
		}
	
	return {
		"valid": True,
		"description": f"Set White Light to {intensity}",
		"duration": 1,
		"cmd": {
			"white" : intensity
        }
	}


@register_command(r"(?i)manu[ea]l")
def cmd_manual(cmd: str) -> dict[str, object]:
	return {
		"valid": True,
		"description": "Manual Sequence (No Container Auto-Setup)",
		"duration": 0,
		"cmd": {
			"manual": True
		}
	}


@register_command(r"(?i)^(?:(?:r|g|b|w|ir)\s*\d+\s*)+$")
def cmd_light_channels(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	min_val = 0
	max_val = 1000

	if match is None or not cmd.strip():
		return {"description": "Set Light channels R[val]G[val]B[val]W[val]IR[val] (e.g. R100, R100G50)"}

	pairs = re.findall(r'(?i)(r|g|b|w|ir)\s*(\d+)', cmd)
	if not pairs:
		return {
			"valid": False,
			"description": "Invalid light command syntax. Use R[val]G[val]B[val]W[val]IR[val].",
			"duration": 0,
			"error": "No valid channel and intensity pairs found",
			"cmd": {}
		}

	payload = {"r": 0, "g": 0, "b": 0, "w": 0, "ir": 0}
	descriptions = []
	for channel_name, intensity_str in pairs:
		channel = channel_name.lower()
		intensity = int(intensity_str)

		if not (min_val <= intensity <= max_val):
			return {
				"valid": False,
				"description": f"Light {channel.upper()} intensity {intensity} out of bounds ({min_val}-{max_val})",
				"duration": 0,
				"error": f"Value for {channel} must be between {min_val} and {max_val}",
				"cmd": {}
			}
		payload[channel] = intensity
		descriptions.append(f"{channel.upper()}:{intensity}")

	import json
	return {
		"valid": True,
		"description": f"Set Light " + ", ".join(descriptions),
		"duration": 1,
		"cmd": {
			"light": payload,
			"r": payload["r"],
			"g": payload["g"],
			"b": payload["b"],
			"w": payload["w"],
			"ir": payload["ir"],
			"send": json.dumps(payload),
		}
	}


@register_command(r"(?i)rgbw\s*(.*)")
def cmd_rgbw(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	min_val = 0
	max_val = 1000

	if match is None or not match.group(1).strip():
		return {"description": f"Set RGBW channels (e.g., 'rgbw r 500 g 0') to [{min_val}-{max_val}]"}
	
	args_str = match.group(1)
	pairs = re.findall(r'\b(r|g|b|w|ir)\s*(\d+)', args_str, re.IGNORECASE)
	
	if not pairs:
		return {
			"valid": False,
			"description": "Invalid RGBW command syntax. Use 'rgbw [channel] [value]'.",
			"duration": 0,
			"error": "No valid channel and intensity pairs found",
			"cmd": {}
		}
	
	payload = {"r": 0, "g": 0, "b": 0, "w": 0, "ir": 0}
	descriptions = []
	for channel_name, intensity_str in pairs:
		channel = channel_name.lower()
		intensity = int(intensity_str)
		
		if not (min_val <= intensity <= max_val):
			return {
				"valid": False,
				"description": f"RGBW {channel} intensity {intensity} out of bounds ({min_val}-{max_val})",
				"duration": 0,
				"error": f"Value for {channel} must be between {min_val} and {max_val}",
				"cmd": {}
			}
		payload[channel] = intensity
		descriptions.append(f"{channel}:{intensity}")
	
	import json
	return {
		"valid": True,
		"description": f"Set RGBW " + ", ".join(descriptions),
		"duration": 1,
		"cmd": {
			"light": payload,
			"r": payload["r"],
			"g": payload["g"],
			"b": payload["b"],
			"w": payload["w"],
			"ir": payload["ir"],
			"send": json.dumps(payload),
		}
	}


@register_command(r"(?i)out2\s+(.+)")
def cmd_out2(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Send command to output 2 (e.g. out2 det 100)"}

	inner_cmd = match.group(1).strip()
	
	from .sequence_processor import sequence_processor
	prev_raw = sequence_processor.current_raw_cmd
	if prev_raw and len(prev_raw) == len(cmd):
		sequence_processor.current_raw_cmd = prev_raw[match.start(1):match.end(1)].strip()
	
	try:
		# Find and run translation for the inner command
		for regex, func, desc in command_registry:
			# Avoid infinite recursion by skipping out2 pattern itself
			if "out2" in regex.pattern:
				continue
			
			result = func(inner_cmd)
			if result:
				result = result.copy()
				result["out2"] = True
				if "description" in result:
					result["description"] = f"{result['description']} (Out2)"
				return result
	finally:
		sequence_processor.current_raw_cmd = prev_raw

	return {
		"valid": False,
		"description": f"Unknown command after out2: '{inner_cmd}'",
		"duration": 0.0,
		"error": "unknown"
	}


@register_command(r"(?i)send\s+(.+)")
def cmd_send(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Send a raw string command"}

	from .sequence_processor import sequence_processor
	raw_cmd = sequence_processor.current_raw_cmd
	if raw_cmd and len(raw_cmd) == len(cmd):
		payload = raw_cmd[match.start(1):match.end(1)].strip()
	else:
		payload = match.group(1).strip()

	return {
		"valid": True,
		"description": f"Send raw command: '{payload}'",
		"duration": 1,
		"cmd": {
			"send": payload
		}
	}


@register_command(r"(?i)ox\s*(0|1)")
def cmd_ox(cmd: str, match: re.Match[str] | None) -> dict[str, object]:
	if match is None:
		return {"description": "Oxygen recording: ox [0|1]"}

	state = int(match.group(1))
	active = (state == 1)
	return {
		"valid": True,
		"description": f"Oxygen recording: {'ON' if active else 'OFF'}",
		"duration": 1,
		"cmd": {
			"ox": active,
			"polling": active,
			"save_sequence": active,
			"record": active,
		}
	}



