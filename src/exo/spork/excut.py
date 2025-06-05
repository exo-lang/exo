"""
excut: Exo eXecuted cuda test

Utility for logging actions on the CPU or CUDA device

The executed cuda code generates a trace JSON file,
and the pytest case generates a reference JSON file.
The two files get checked for concordance.

Syntax: List of actions, with actions being the 7-tuple:

    [action name: str, args: List[str],
    device name: str, blockIdx.x: int, threadIdx.x: int,
    source file: str, line: int]

where device name is "cpu" or "cuda" (blockIdx.x = threadIdx.x = 0 for cpu)
and where each arg is a string holding:

    integer: "int:{variable-length integer, decimal or hex}"
    string: "str:{string}"
    variable: "var:{varname}[{idxs}] + {offset}"
    sink: "_"

Variables and sinks must appear only in reference files.

* varname is an alphanumeric string (plus underscores)
  and must be in varnames_set (to avoid accidentally passing
  tests due to typos)

* idxs is a comma-separated list of integers.

* offset is an integer.

* Both "[{idxs}]" and "{offset}" are optional; assumed [] and 0 respectively.

* Within a single CUDA kernel launch, stable-sort all actions
  lexicographically by (blockIdx.x, threadIdx.x).

Concordance:

Take the set of all action names in the reference file,
and erase all actions in the trace file with an action name not in the set.
(Extensibility/backwards compatibility: allows us to trace new actions
without impacting old tests that don't care about them)

Expect the remaining list of actions to match, i.e.
that all attributes (except source file and line) match,
and arg matching is defined as:

* strings match with identical strings, or sink

* integers match with identical integers, sink, or variables,
  with the latter case deducing a value for the variable.

Deduction Rules:

* Each (varname, idxs) pair identifies a variable

* When we match int:{N} with {varname}[{idxs}] + {offset},
  deduce the variable (varname, idxs) to be N - offset.

* All deductions for a given variable must deduce the same value.

* Variables with the same varname but different idxs must be deduced
  to be different values.

The main purpose of deduction is to match pointer values, without knowing
the concrete value of the pointer.

"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Set, Tuple, Union


@dataclass(slots=True, frozen=True, eq=True)
class ExcutVariableID:
    varname: str
    idxs: Tuple[int]

    def encode(self):
        idx_str = ""
        if self.idxs:
            idx_str = "[" + ",".join(str(n) for n in self.idxs) + "]"
        return f"{self.varname}{idx_str}"


@dataclass(slots=True)
class ExcutVariableArg:
    id: ExcutVariableID
    offset: int

    def encode(self):
        return f"{self.id.encode()} + {self.offset}"


@dataclass(slots=True)
class ExcutSink:
    def encode(self):
        return "_"


sink = ExcutSink()


@dataclass
class ExcutDeduction:
    value: int
    id: ExcutVariableID
    src_file: str
    src_line: int
    json_file: str
    json_line: int

    def srcinfo(self):
        return f"{self.src_file}:{self.src_line} {self.json_file}:{self.json_line}?"


@dataclass(slots=True)
class ExcutAction:
    action_name: str
    args: List[Union[str, int, ExcutSink, ExcutVariableArg]]
    device_name: str
    blockIdx: int
    threadIdx: int
    src_file: str
    src_line: int
    json_file: str
    json_line: int
    # JSON parser doesn't actually know line numbers; we just guess
    # based on the standard format that we print in: opening and
    # closing [] on the first/last line, and then one line per action.

    def as_json(self):
        return json.dumps(
            [
                self.action_name,
                [self.encode_arg(a) for a in self.args],
                self.device_name,
                self.blockIdx,
                self.threadIdx,
                self.src_file,
                self.src_line,
            ]
        )

    def encode_arg(self, a):
        if isinstance(a, int):
            return f"int:{hex(a)}"
        if isinstance(a, str):
            return f"str:{a}"
        return a.encode()

    def srcinfo(self):
        return f"{self.src_file}:{self.src_line} {self.json_file}:{self.json_line}?"

    def match_trace(
        self,
        trace: ExcutAction,
        deductions: Dict[ExcutVariableID, ExcutDeduction],
        varnames_set: Set[str],
    ):
        """Match self (from the reference actions) with trace's ExcutAction

        Store a deduced variable value, if appropriate."""

        def fail(reason):
            raise ValueError(
                f"""excut concordance failed: {reason}
{self.as_json()} @ {self.srcinfo()}
{trace.as_json()} @ {trace.srcinfo()}"""
            )

        if self.action_name != trace.action_name:
            fail(f"{self.action_name!r} != {trace.action_name!r}")
        if self.device_name != trace.device_name:
            fail(f"{self.device_name!r} != {trace.device_name!r}")
        if (self.blockIdx, self.threadIdx) != (trace.blockIdx, trace.threadIdx):
            fail(
                f"({self.blockIdx}, {self.threadIdx}) != ({trace.blockIdx}, {trace.threadIdx})"
            )
        if len(self.args) != len(trace.args):
            fail("len(args) mismatch")

        for i, ref_arg, trace_arg in zip(range(len(self.args)), self.args, trace.args):
            if isinstance(trace_arg, str):
                if isinstance(ref_arg, ExcutSink):
                    continue
                elif ref_arg == trace_arg:
                    continue
                else:
                    fail(f"args[{i}] mismatch: {ref_arg!r} != {trace_arg!r}")
            elif isinstance(trace_arg, int):
                if isinstance(ref_arg, ExcutSink):
                    continue
                elif ref_arg == trace_arg:
                    continue
                elif isinstance(ref_arg, ExcutVariableArg):
                    deduced = trace_arg - ref_arg.offset
                    var_id = ref_arg.id
                    old_deduction = deductions.get(var_id)
                    if old_deduction is None:
                        if var_id.varname not in varnames_set:
                            fail(f"{var_id.varname!r} not in varnames_set")
                        deductions[var_id] = ExcutDeduction(
                            deduced,
                            var_id,
                            trace.src_file,
                            trace.src_line,
                            trace.json_file,
                            trace.json_line,
                        )
                    elif old_deduction.value != deduced:
                        fail(
                            f"""args[{i}] mismatch: {ref_arg.encode()} != {hex(trace_arg)}
mismatches {var_id.encode()} = {hex(old_deduction.value)} deduced at {old_deduction.srcinfo()}"""
                        )
                else:
                    ref_arg_text = (
                        hex(ref_arg) if isinstance(ref_arg, int) else repr(ref_arg)
                    )
                    fail(f"args[{i}] mismatch: {ref_arg_text} != {hex(trace_arg)}")
            else:
                fail(
                    f"Invalid trace (internal error?); args[{i}] has unknown type {type(trace_arg)}"
                )

        # Note: uniqueness property for deduced variables not diagnosed here


def decode_arg(encoded):
    if not isinstance(encoded, str):
        raise TypeError("Expected str")

    if encoded == "_":
        return sink

    if encoded.startswith("int:"):
        return int(encoded[4:], 0)

    if encoded.startswith("str:"):
        return encoded[4:]

    if encoded.startswith("var:"):
        var_text = encoded[4:]

        # Parse and remove offset to the right of +, if present
        offset = 0
        plus_split = var_text.split("+")
        if len(plus_split) == 2:
            var_text, offset_text = plus_split
            offset = int(offset_text)
        elif len(plus_split) != 1:
            raise ValueError("Too many +")

        # Parse and remove idxs, if present
        idxs = ()
        lbracket_split = var_text.split("[")
        if len(lbracket_split) == 2:
            var_text, tmp = lbracket_split
            rbracket_split = tmp.split("]")
            if len(rbracket_split) != 2:
                raise ValueError("Expected one ]")
            idx_strs, cruft = rbracket_split
            if cruft.strip():
                raise ValueError(f"Unexpected cruft {cruft!r} after ]")
            idxs = tuple(int(n) for n in idx_strs.split(","))
        elif len(lbracket_split) != 1:
            raise ValueError("Too many [")

        # Remove whitespace and check valid varname
        varname = var_text.strip()
        for c in varname:
            if c != "_" and not c.isalnum():
                raise ValueError("Expected alphanumeric varname")

        return ExcutVariableArg(ExcutVariableID(varname, idxs), offset)

    raise ValueError("Expected _, int:, str:, or var:")


def parse_json_file(filename) -> List[ExcutAction]:
    f = open(filename)
    try:
        j = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse {filename!r}") from e

    if not isinstance(j, list):
        raise ValueError(f"{filename!r} needs to contain a list")

    actions = []

    for i, encoded_action in enumerate(j):
        if not isinstance(encoded_action, list):
            raise ValueError(f"{filename!r}: action #{i} needs to be a list")

        if len(encoded_action) != 7:
            raise ValueError(f"{filename!r}: action #{i} needs to be length-7 list")

        def as_non_negative_int(value, name):
            try:
                n = int(value)
                if n != value or n < 0:
                    raise ValueError("Not a positive integer")
                return n
            except Exception as e:
                raise ValueError(f"{filename!r}: action #{i}, invalid {name}") from e

        def as_str(value, name):
            if not isinstance(value, str):
                raise ValueError(f"{filename!r}: action #{i}, expect str {name}")
            return value

        action_name = as_str(encoded_action[0], "action_name")
        encoded_args = encoded_action[1]
        device_name = as_str(encoded_action[2], "device_name")
        blockIdx = as_non_negative_int(encoded_action[3], "blockIdx")
        threadIdx = as_non_negative_int(encoded_action[4], "threadIdx")
        src_file = as_str(encoded_action[5], "src_file")
        src_line = as_non_negative_int(encoded_action[6], "src_line")

        if not isinstance(encoded_args, list):
            raise ValueError(f"{filename!r}: action #{i}, expect args list")

        args = []
        for arg_i, a in enumerate(encoded_args):
            try:
                args.append(decode_arg(a))
            except Exception as e:
                raise ValueError(f"{filename!r}: action #{i}, invalid arg {a!r}") from e

        actions.append(
            ExcutAction(
                action_name,
                args,
                device_name,
                blockIdx,
                threadIdx,
                src_file,
                src_line,
                filename,  # JSON file name
                i + 2,  # Guess the line number
            )
        )

    return actions


def require_concordance(ref_actions, trace_actions, varnames_set):
    ref_action_names = set()
    for r_act in ref_actions:
        assert isinstance(r_act, ExcutAction)
        ref_action_names.add(r_act.action_name)

    trace_actions = list(
        filter(lambda act: act.action_name in ref_action_names, trace_actions)
    )
    deductions: Dict[ExcutVariableID, ExcutDeduction] = {}

    # NOTE: we check len(ref_actions) == len(trace_actions) later, since a
    # "mismatched length" error is terrible for diagnosing a test case failure
    for ref_a, trace_a in zip(ref_actions, trace_actions):
        ref_a.match_trace(trace_a, deductions, varnames_set)

    if len(ref_actions) > len(trace_actions):
        raise ValueError(
            f"No trace action left to match {ref_actions[len(trace_actions)].srcinfo()}"
        )
    if len(ref_actions) < len(trace_actions):
        raise ValueError(
            f"No reference action left to match {trace_actions[len(ref_actions)].srcinfo()}"
        )

    # varname -> (value -> deduction)
    varname_value_deductions: Dict[str, Dict[int, ExcutDeduction]] = {}

    # Check "different values" for same-varname, different-idx variables
    for var_id, new_deduction in deductions.items():
        varname = var_id.varname
        if varname not in varname_value_deductions:
            value_deductions = {}
            varname_value_deductions[varname] = value_deductions
        else:
            value_deductions = varname_value_deductions[varname]

        value = new_deduction.value
        old_deduction = value_deductions.get(value)
        if old_deduction is None:
            value_deductions[value] = new_deduction
        elif old_deduction.id.idxs != new_deduction.id.idxs:
            old_id = old_deduction.id
            new_id = new_deduction.id
            raise ValueError(
                f"""Duplicate deduced value {hex(value)} for
{old_id.encode()} @ {old_deduction.srcinfo()}
{new_id.encode()} @ {new_deduction.srcinfo()}"""
            )
