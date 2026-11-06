from __future__ import annotations
import re
from .command_registry import command_registry


class SequenceProcessor:
	sequence: str
	current_raw_cmd: str
	_end_seq_cmd: str | None

	def __init__(self) -> None:
		self.sequence = ""
		self.current_raw_cmd = ""
		self._end_seq_cmd = None

	@property
	def end_seq_cmd(self) -> str | list[str]:
		if self._end_seq_cmd is not None:
			return self._end_seq_cmd
		try:
			from core.config_manager import config
			val = config.get_nested("SequenceProcessor", "end_seq_cmd", default="R0G0B0W0IR1000")
			if isinstance(val, list):
				return [str(x) for x in val]
			return str(val) if val is not None else "R0G0B0W0IR1000"
		except Exception:
			return "R0G0B0W0IR1000"

	@end_seq_cmd.setter
	def end_seq_cmd(self, value: str | list[str] | None) -> None:
		self._end_seq_cmd = value

	def _split_sequence(self, sequence: str) -> list[str]:
		"""Split the sequence into individual commands."""
		return sequence.replace('\n', '').strip().split(',')

	def _substitute_variables(self, sequence: str) -> str:
		"""
		Extract and substitute global variable declarations.

		Syntax:  var NAME=VALUE
		  - NAME is one or more word characters (letters, digits, underscore)
		  - VALUE is a number (integer or decimal)
		  - Multiple declarations are separated by commas like normal commands
		  - Declarations are removed from the sequence after extraction
		  - Every remaining occurrence of NAME (whole word) is replaced by VALUE

		Examples:
			var Y=100,var Z=5,[X=0,X+100,5](OJIP Y 512 X,Zs)
			  -> [X=0,X+100,5](OJIP 100 512 X,5s)
			  -> OJIP 100 512 0,5s,OJIP 100 512 100,5s, ...

		Args:
			sequence (str): The raw input sequence.

		Returns:
			str: Sequence with variable declarations removed and values substituted.
		"""
		# Match pattern: var NAME=VALUE (case-insensitive for 'var')
		# VALUE is everything up to the next comma (text or numbers)
		var_pattern = re.compile(r'(?i)var\s+(\w+)\s*=\s*([^,]+)')

		# Collect all variable declarations
		variables = {}
		for m in var_pattern.finditer(sequence):
			name = m.group(1)
			variables[name] = m.group(2).strip()

		if not variables:
			return sequence

		# Remove the var declarations (and any trailing comma/whitespace)
		sequence = re.sub(r'(?i)var\s+\w+\s*=\s*[^,]+\s*,?\s*', '', sequence)

		# Clean up any leading/trailing commas left behind
		sequence = re.sub(r',{2,}', ',', sequence).strip(',')

		# Substitute each variable (simple replace, same as loop variable expansion)
		for name, value in variables.items():
			sequence = sequence.replace(name, value)

		return sequence
	
	def _expand_var_loops(self, sequence: str) -> str:
		"""
		Expands variable loop patterns of the form [VAR=start,VAR<op>step,count](CMD).

		Supported operators: +  -  *  /
		Replaces VAR in CMD with the current value at each iteration.

		Examples:
			[X=100,X+100,5](DX)   -> D100,D200,D300,D400,D500
			[V=60,V-10,4](voltV)  -> volt60,volt50,volt40,volt30
			[X=1,X*2,5](aX)       -> a1,a2,a4,a8,a16
			[X=1024,X/2,4](aX)    -> a1024,a512,a256,a128
			[T=10,T+10,3](pdet T 5 100) -> pdet 10 5 100,pdet 20 5 100,pdet 30 5 100

		Args:
			sequence (str): The input sequence string.

		Returns:
			str: The expanded sequence string.
		"""
		# Pattern: [VAR=start, VAR<op>operand, count](CMD)  — op is one of + - * /
		pattern = re.compile(
			r'\[(\w+)=(\d+(?:\.\d+)?),\1([+\-*/])(\d+(?:\.\d+)?),(\d+)\]\(([^)]+)\)'
		)

		ops = {
			'+': lambda a, b: a + b,
			'-': lambda a, b: a - b,
			'*': lambda a, b: a * b,
			'/': lambda a, b: a / b,
		}

		def expand_match(m: re.Match[str]) -> str:
			var     = m.group(1)
			start   = float(m.group(2))
			op      = m.group(3)
			operand = float(m.group(4))
			count   = int(m.group(5))
			cmd     = m.group(6)
			apply   = ops[op]

			parts = []
			value = start
			for _ in range(count):
				# Format as int when value has no fractional part
				formatted = str(int(value)) if value == int(value) else str(value)
				parts.append(cmd.replace(var, formatted))
				value = apply(value, operand)

			return ','.join(parts)

		# Keep expanding until no more matches (supports multiple loops in one sequence)
		while pattern.search(sequence):
			sequence = pattern.sub(expand_match, sequence)

		return sequence

	def _expand_groups(self, sequence: str) -> str:
		"""
		Recursively expands a sequence with nested parentheses.

		Args:
			sequence (str): The input sequence containing patterns like 2(10s,2(T100)).

		Returns:
			str: The expanded sequence.
		"""
		# Regex to find the innermost pattern of the form N(...), where N is a number
		pattern = re.compile(r'(\d+)\(([^()]+)\)')
		
		# While there are still expandable patterns in the sequence
		while (match := pattern.search(sequence)):
			# Extract the multiplier and the content inside the parentheses
			multiplier = int(match.group(1))
			content = match.group(2)
			
			# Expand the content by repeating it 'multiplier' times
			expanded_content = ','.join([content] * multiplier)
			
			# Replace the current match with the expanded content
			sequence = sequence[:match.start()] + expanded_content + sequence[match.end():]

		return sequence
	
	def get_commands(self, sequence: str = "") -> list[str]:
		return self._split_sequence(self._expand_groups(self._expand_var_loops(self._substitute_variables(sequence))))

	def trad_cmd(self, cmd: str) -> dict[str, object]:
		self.current_raw_cmd = cmd
		cmd_lower = cmd.lower()
		for regex, func, desc in command_registry:
			result = func(cmd_lower)
			if result:
				return result

		return {
			"valid": False,
			"description": "Unknown command",
			"duration": 0.0,
			"error": "unknown"
		}


sequence_processor = SequenceProcessor()