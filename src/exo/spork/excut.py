"""excut: Exo eXecuted cuda test

Utility for logging actions on the CPU or CUDA device.
We provide InlinePtxGen for use by @instr implementations, to generate
inline PTX that's also logged to excut.

The executed cuda code generates a trace JSON file,
and the pytest case generates a reference JSON file.
The two files get checked for concordance.

Syntax: List of actions or sub-lists of actions,
with actions being the 7-tuple:

    [action name: str, args: List[str],
    device name: str, blockIdx.x: int, threadIdx.x: int,
    source file: str, line: int]

where device name is "cpu" or "cuda" (blockIdx.x = threadIdx.x = 0 for cpu)
and where each arg is a string holding:

    integer: "int:{variable-length integer, decimal or hex}"
    string: "str:{string}"
    variable: "var:{varname}[{idxs}] + {offset}"
    sink: "_"

Variables, sinks, and sub-lists must appear only in reference files.

* varname is an alphanumeric string (plus underscores)
  and must be in varname_set (to avoid accidentally passing
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

* a sub-list of N reference actions matches with N traced actions
  in any permutation (see deduction + permutation note")

Deduction Rules:

* Each (varname, idxs) pair identifies a variable

* When we match int:{N} with {varname}[{idxs}] + {offset},
  deduce the variable (varname, idxs) to be N - offset.

* All deductions for a given variable must deduce the same value.

* Variables with the same varname but different idxs must be deduced
  to be different values.

The main purpose of deduction is to match pointer values, without knowing
the concrete value of the pointer.

Out of Memory:

excut_begin_log_file's cuda_log_bytes value determines the size of the allocated
buffer used to transfer data from CUDA to the CPU. If this is too small, the
tracer will log an action named "excut::out_of_cuda_memory" with a single
int arg being the recommended cuda_log_bytes value.

Deduction + Permutation Note:

Deduction is potentially ambiguous when we use the sub-list
permutation matching feature.  This can cause the concordance to fail
in cases when a solution could have been found. To minimize the risk of
this, we try to deduce values based on non-permuted actions first,
then check permuted actions.

"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import string
from typing import Dict, List, Optional, Set, Tuple, Union
from ..core.prelude import get_srcinfo


########################################################################
# Excut Parsing and Concordance
# Functions for parsing JSON and comparing reference and trace actions
########################################################################


class ExcutConcordanceError(ValueError):
    pass


class ExcutOutOfCudaMemory(ValueError):
    __slots__ = ["bytes_needed"]

    def __init__(self, trace_filename, bytes_needed):
        super().__init__(
            f"{trace_filename}: not enough cuda memory; "
            f"set cuda_log_bytes={bytes_needed} in exo_excut_begin_log_file"
        )
        self.bytes_needed = bytes_needed


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
class ExcutVariableArg:
    id: ExcutVariableID
    offset: int

    def encode(self):
        return f"var:{self.id.encode()} + {self.offset}"

    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            idxs = self.id.idxs + idxs
        else:
            assert isinstance(idxs, int)
            idxs = self.id.idxs + (idxs,)
        return ExcutVariableArg(ExcutVariableID(self.id.varname, idxs), self.offset)

    def __add__(self, offset):
        return ExcutVariableArg(self.id, self.offset + offset)

    def __call__(self, deductions: Dict[ExcutVariableID, ExcutDeduction]):
        return deductions[self.id] + self.offset


@dataclass(slots=True)
class ExcutAction:
    # The nth reference action can match with any not-already-matched
    # trace action in trace_actions[base : base + permutation_slice_len].
    # where base = n - permutation_slice_offset
    permutation_slice_offset: int
    permutation_slice_len: int

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
        varname_set: Set[str],
        defer_deduction: bool,
    ):
        """Match self (from the reference actions) with trace's ExcutAction

        Store a deduced variable value, if appropriate, unless defer_deduction"""

        def fail(reason):
            raise ExcutConcordanceError(
                f"""excut concordance failed: {reason}
{self.as_json()} @ {self.srcinfo()}
{trace.as_json()} @ {trace.srcinfo()}"""
            )

        def require_eq(attr):
            a = getattr(self, attr)
            b = getattr(trace, attr)
            if a != b:
                fail(f"{attr}: {a!r} != {b!r}")

        require_eq("action_name")  # First; generally most relevant for error message
        require_eq("device_name")
        require_eq("blockIdx")
        require_eq("threadIdx")
        if len(self.args) != len(trace.args):
            fail(f"len(args): {len(self.args)} != {len(trace.args)}")

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
                        if var_id.varname not in varname_set:
                            fail(f"{var_id.varname!r} not in varname_set")
                        if not defer_deduction:
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
            offset = int(offset_text, 0)
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
            idxs = tuple(int(n, 0) for n in idx_strs.split(","))
        elif len(lbracket_split) != 1:
            raise ValueError("Too many [")

        # Remove whitespace and check valid varname
        varname = var_text.strip()
        for c in varname:
            if c != "_" and not c.isalnum():
                raise ValueError(f"Expected alphanumeric varname, not {varname!r}")

        return ExcutVariableArg(ExcutVariableID(varname, idxs), offset)

    raise ValueError("Expected _, int:, str:, or var:")


def parse_json_file(filename: str) -> List[ExcutAction]:
    filename = str(filename)
    f = open(filename)
    try:
        j = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse {filename!r}") from e

    if not isinstance(j, list):
        raise ValueError(f"{filename!r} needs to contain a list")

    actions = []
    max_oom_bytes = 0

    def add_action(encoded_action, permutation_slice_offset, permutation_slice_len):
        i = len(actions)
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

        if action_name == "excut::out_of_cuda_memory":
            nonlocal max_oom_bytes
            max_oom_bytes = max(max_oom_bytes, args[0])

        actions.append(
            ExcutAction(
                permutation_slice_offset,
                permutation_slice_len,
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

    for item in j:
        i = len(actions)
        if not isinstance(item, list):
            raise ValueError(f"{filename!r}: action #{i} needs to be a list")
        if len(item) == 0 or isinstance(item[0], list):
            for offset, action in enumerate(item):
                add_action(action, offset, len(item))
        else:
            add_action(item, 0, 1)

    if max_oom_bytes:
        raise ExcutOutOfCudaMemory(filename, max_oom_bytes)

    return actions


def require_concordance(
    ref_actions: List[ExcutAction],
    trace_actions: List[ExcutAction],
    varname_set: Set[str],
):
    ref_action_names = set()
    for r_act in ref_actions:
        assert isinstance(r_act, ExcutAction)
        ref_action_names.add(r_act.action_name)

    trace_actions = list(
        filter(lambda act: act.action_name in ref_action_names, trace_actions)
    )

    deductions: Dict[ExcutVariableID, ExcutDeduction] = {}
    trace_i_to_ref_i = [None] * len(trace_actions)

    def get_trace_action(trace_i, ref_a):
        if trace_i >= len(trace_actions):
            raise ExcutConcordanceError(
                f"No trace action left to match {ref_a.srcinfo()}"
            )
        return trace_actions[trace_i]

    # NOTE: we check for mismatched lengths as late as possible as a
    # "mismatched length" error is terrible for diagnosing a test case failure.
    #
    # As explained in "Deduction + Permutation Note", we try to avoid
    # deducing variable values from permuted reference actions.
    # Checking is done in two phases:
    #
    # Phase 0: match non-permuted reference actions, partially match permuted actions
    # (don't deduce variable values, and assume all variables match for now).
    #
    # Phase 1: match permuted reference actions
    # (checks "assume all variables match" is accurate).
    #
    # The "partial match" step in phase 0 is not strictly needed,
    # but this provides better feedback on test failures.
    for phase in range(2):
        for ref_i, ref_a in enumerate(ref_actions):
            offset = ref_a.permutation_slice_offset
            slice_len = ref_a.permutation_slice_len
            assert slice_len > 0
            fail_messages = []
            success = False
            if phase == 1 and slice_len == 1:
                continue
            for trace_i in range(ref_i - offset, ref_i - offset + slice_len):
                trace_a = get_trace_action(trace_i, ref_a)
                matched_ref_i = trace_i_to_ref_i[trace_i]
                defer_deduction = slice_len > 1 and phase == 0
                if matched_ref_i is not None:
                    fail_messages.append(
                        f"Already matched {trace_actions[trace_i].srcinfo()}"
                    )
                else:
                    try:
                        ref_a.match_trace(
                            trace_a, deductions, varname_set, defer_deduction
                        )
                        success = True
                        if not defer_deduction:
                            trace_i_to_ref_i[trace_i] = ref_i
                    except ExcutConcordanceError as e:
                        fail_messages.append(str(e))
            if not success:
                raise ExcutConcordanceError("\n".join(fail_messages))

    if len(trace_actions) > len(ref_actions):
        trace_a = trace_actions[len(ref_actions)]
        raise ExcutConcordanceError(
            f"No reference action left to match {trace_a.srcinfo()}"
        )

    # varname -> (value -> deduction)
    varname_value_deductions: Dict[str, Dict[int, ExcutDeduction]] = {}

    # Uniqueness property for deduced variables with same varname
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
            raise ExcutConcordanceError(
                f"""Duplicate deduced value {hex(value)} for
{old_id.encode()} @ {old_deduction.srcinfo()}
{new_id.encode()} @ {new_deduction.srcinfo()}"""
            )

    return deductions


@dataclass(slots=True)
class ExcutBuilderAction:
    json_action: str
    permute_group_id: Optional[int]


@dataclass(slots=True)
class ExcutReferenceGenerator:
    """Makes reference action list and writes them to a JSON file

    It's a bit funny how we generate this list in memory, write it to
    file, and immediately read it back again. However, this ensures
    the test directory contains this list in human-readable form,
    enhancing the debuggability of failing test cases.

    """

    _permute_group_id: Optional[int]
    _device_name: str
    _blockIdx: int
    _threadIdx: int
    _actions: List[ExcutBuilderAction]
    varname_set: Set[str]

    def __init__(self):
        self._permute_group_id = None
        self._device_name = "cpu"
        self._blockIdx = 0
        self._threadIdx = 0
        self._actions = []
        self.varname_set = set()

    def permuted(self):
        """Usage: with xrg.permuted(): ...

        Within the body of the with statement, the N-many logged reference
        actions will be matched with N-many trace actions in any permutation.

        """

        return ExcutPermuterContext(self, self._permute_group_id is not None)

    def stride_blockIdx(self, n, stride=1, offset=0):
        """Usage: for i in xrg.stride_blockIdx(n, stride): ...

        Iterates i in range(0, n).
        Within the loop body, actions are logged as being done on the cuda
        device with blockIdx = blockIdx" + offset + i * stride,
        blockIdx" being the blockIdx value outside the loop (initially 0).

        """
        old_device_name = self._device_name
        base_blockIdx = self._blockIdx

        def generator():
            self._device_name = "cuda"
            for i in range(n):
                self._blockIdx = base_blockIdx + i * stride
                yield i
            self._blockIdx = base_blockIdx
            self._device_name = old_device_name

        return generator()

    def stride_threadIdx(self, n, stride=1, offset=0):
        """Usage: for i in xrg.stride_threadIdx(n, stride): ...

        Same as stride_blockIdx, but modifies threadIdx instead.

        """
        old_device_name = self._device_name
        base_threadIdx = self._threadIdx

        def generator():
            self._device_name = "cuda"
            for i in range(n):
                self._threadIdx = base_threadIdx + i * stride
                yield i
            self._threadIdx = base_threadIdx
            self._device_name = old_device_name

        return generator()

    def __call__(self, action_name, *args, depth=0):
        """Log an action with args. Uses the current threadIdx, etc.

        An argument may be an int, str, ExcutVariableArg, or excut.sink

        """
        srcinfo = get_srcinfo(depth + 2)
        str_args = []
        for a in args:
            if isinstance(a, int):
                str_args.append(f"int:{hex(a)}")
            elif isinstance(a, str):
                str_args.append(f"str:{a}")
            elif a is sink:
                str_args.append("_")
            else:
                assert isinstance(a, ExcutVariableArg)
                str_args.append(a.encode())
        j = json.dumps(
            [
                action_name,
                str_args,
                self._device_name,
                self._blockIdx,
                self._threadIdx,
                srcinfo.filename,
                srcinfo.lineno,
            ]
        )
        self._actions.append(ExcutBuilderAction(j, self._permute_group_id))

    def new_varname(self, varname: str) -> ExcutVariableArg:
        """Make new variable usable as an action's argument.

        Use python [] and + operators to add indices and offsets, respectively.

        """
        assert varname not in self.varname_set
        self.varname_set.add(varname)
        return ExcutVariableArg(ExcutVariableID(varname, ()), 0)

    def write_json(self, f: file):
        """Serialize JSON to file"""
        actions = self._actions
        f.write("[\n")
        for i, action in enumerate(actions):
            prev_group = None if i == 0 else actions[i - 1].permute_group_id
            next_group = (
                None if i == len(actions) - 1 else actions[i + 1].permute_group_id
            )
            group = action.permute_group_id
            comma = " " if i == 0 else ","
            # Note [ and ] for permute groups don't cause an extra newline.
            # This is so the fake line numbers heuristic still works.
            lb = "[" if group is not None and group != prev_group else ""
            rb = "]" if group is not None and group != next_group else ""
            f.write(f"  {comma}{lb}{action.json_action}{rb}\n")
        f.write("]\n")


@dataclass(slots=True)
class ExcutPermuterContext:
    _generator: ExcutReferenceGenerator
    _was_permuted: bool

    def __enter__(self):
        if not self._was_permuted:
            self._generator._permute_group_id = len(self._generator._actions)

    def __exit__(self, a, b, c):
        if not self._was_permuted:
            self._generator._permute_group_id = None


########################################################################
# Excut Code Generation
# Global string ID table (str->int) and logged inline PTX generator.
# NOTE: the global-ness of the string table could compromise
# stability of test C++ outputs (goldens). For this reason, we refer to
# string IDs by macro (stable) and put the numeric values in a side file.
########################################################################


# Add chars if needed, but any special characters in JSON strings
# would require us to update exo_excut.cu to support escaping
_allowed_char_map = bytearray(128)
for c in string.ascii_letters:
    _allowed_char_map[ord(c)] = 1
for c in " 0123456789!@#$%^&*()',.<>;:[]{}+=-_/~`":
    _allowed_char_map[ord(c)] = 1


def valid_str(s):
    for c in s:
        code = ord(c)
        if code >= 128 or not _allowed_char_map[code]:
            return False
    return True


@dataclass(slots=True)
class ExcutStringTable:
    _str_to_id: Dict[str, int]
    _str_to_sanitized: Dict[str, str]
    _sanitized_names_to_str: Dict[str, str]

    def sanitized(self, name: str):
        """Translate string to C_name for the string.

        We store the string in the excut string table,
        at index EXO_EXCUT_STR_ID(C_name).

        """
        assert name
        s_name = self._str_to_sanitized.get(name)
        if s_name is not None:
            return s_name

        # Every character that's not alphanumeric becomes an underscore.
        # then we remove duplicate, leading, trailing underscores.
        cleaned = "".join(c if c.isalnum() else " " for c in name)
        s_name = "_".join(cleaned.split())
        assert s_name

        # Only up to 24 bits available for ID.
        id_table = self._str_to_id
        _id = len(id_table)
        assert _id < (1 << 24)
        id_table[name] = _id

        # Reserve sanitized name
        if conflict_name := self._sanitized_names_to_str.get(s_name):
            raise ValueError(
                f"excut str_id name conflict: both {name!r} and {conflict_name!r} mapped to {s_name!r}"
            )
        self._sanitized_names_to_str[s_name] = name
        self._str_to_sanitized[name] = s_name
        return s_name


_string_table = ExcutStringTable({}, {}, {})


def excut_c_str_id(name):
    """Register string in excut string table if not already there,
    and return a C macro expression for the string ID of that string."""
    c_name = _string_table.sanitized(name)
    return f"EXO_EXCUT_STR_ID({c_name})"


def generate_excut_str_table_header(namespace_name):
    """Generate header file for excut string table.

    This should be included by the .cuh file only if excut logging is
    enabled; it is separate from the .cuh to avoid polluting test goldens.

    """
    str_to_id = _string_table._str_to_id
    id_count = len(str_to_id)
    strings = [None] * id_count
    for s, i in str_to_id.items():
        assert i < id_count
        strings[i] = s
    lines = [
        "// excut debug utility: generated by excut.generate_excut_str_table_header"
    ]

    # EXO_EXCUT_STR_ID macro [outside header guard]
    lines.append("#ifdef EXO_EXCUT_STR_ID")
    lines.append("#undef EXO_EXCUT_STR_ID")
    lines.append("#endif")
    lines.append(
        f"#define EXO_EXCUT_STR_ID(c) ::{namespace_name}::exo_excut_str_id_##c"
    )

    lines.append("")

    header_guard = f"EXO_EXCUT_STR_TABLE_{namespace_name}"
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append(f"namespace {namespace_name} {{")

    # String table size
    lines.append("inline const unsigned exo_excut_str_id_count = %s;" % id_count)

    # String table contents
    lines.append("inline const char* const exo_excut_str_table[] = {")
    for i, s in enumerate(strings):
        c_str = json.dumps(s)
        lines.append(f"  {c_str},  // {i}")
    lines.append("};")

    # String-id constants
    for s, i in str_to_id.items():
        c_name = _string_table._str_to_sanitized[s]
        lines.append(f"constexpr unsigned exo_excut_str_id_{c_name} = {i};")

    lines.append("}  // end namespace")
    lines.append("#endif")

    return "\n".join(lines)


@dataclass(slots=True)
class InlinePtxParsedLine:
    # String split into fragments; placeholder #N# replaced with literal int N
    fragments: Tuple[Union[str, int]]
    log_action: Optional[str]  # If not None, passed to exo_excutLog.log_action(...)


@dataclass(slots=True)
class InlinePtxParsedArg:
    c_arg: str  # Passed to inline PTX as "+f"(c_arg) [or whatever constraint]
    logged_c_arg: str  # Passed to exo_excutLog.{log_fname}
    constraint: str
    arg_index: int  # Index in InlinePtxGen.{im}mutable_args, if not true_const
    mutable: bool
    true_const: bool
    log_fname: Optional[str]
    ptx_prefix: str = ""
    ptx_suffix: str = ""
    advise_newline: bool = True

    def c_log_stmt(self) -> Optional[str]:
        if fname := self.log_fname:
            return f"exo_excutLog.{fname}({self.logged_c_arg});"
        return None


def simple_ptx_c_lines(ptx_instr, *int_args, tab=""):
    """Wrapper around InlinePtxGen for no-arg inline PTX

    We still support true_const integer arguments"""
    assert "#" not in ptx_instr
    assert ";" not in ptx_instr
    spc = " " if int_args else ""
    ptx = InlinePtxGen(ptx_instr + spc + "#0#;", volatile=True)
    for a in int_args:
        assert isinstance(a, int)
        ptx.add_arg(a, constraint="n", log_as="bits")
    return ptx.as_c_lines(py_format=False, tab=tab)


@dataclass(init=False, slots=True)
class InlinePtxGen:
    """Utility for generating inline PTX, and optional excut logging

    The utility generates
    * the string for the asm statement, including % placeholders
    * the arguments to the inline PTX (e.g. "+f"(my_arg) )
    * excut logging statements, using exo_excutLog: exo_ExcutThreadLog

    The ptx_format can contain multiple lines, which we mostly preserve in the
    generated PTX (auto-inserting \\n\\t as recommended by the PTX guide).
    We expect no backslashes or other special characters in strings here.

    Each character sequence of the form #N#, N:int, is a "placeholder"
    which gets filled with arguments given by add_arg. Generating PTX
    placeholders (%5...) and arguments ("f"(foo)) is automatic.

    Logging: each line of ptx_format that contains a placeholder will
    generate excut logging. We use the first whitespace-separated
    word, before any placeholders, as the instr (action) name, and log
    all associated args. Instrs/args get logged in the order they
    appear in formatted PTX.

    NOTE: at least one line must contain a placeholder, even if the
    instr takes no arguments (e.g. "wgmma.fence.sync.aligned#0#;")
    so that at least one instruction is logged by excut.
    Consider simple_ptx_c_lines for such simple cases.

    """

    parsed_lines: List[InlinePtxParsedLine]
    placeholder_args: Dict[int, List[InlinePtxParsedArg]]
    mutable_args: List[InlinePtxParsedArg]  # mutable and not true_const
    immutable_args: List[InlinePtxParsedArg]  # not mutable and not true_const
    volatile: bool

    _placeholder_regex = re.compile("#[0-9]+#")

    def __init__(self, ptx_format: str, *, volatile: bool):
        self.parsed_lines = []
        self.placeholder_args = {}
        self.mutable_args = []
        self.immutable_args = []
        self.volatile = bool(volatile)

        # Parse ptx_format into List[InlinePtxParsedArg]
        # and take census of placeholder N values.
        for ln in ptx_format.split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            fragments = [ln]
            have_args = False
            while True:
                # Find the next #N# in the last string in fragments, if any.
                # If found, split said string into fragments
                s = fragments[-1]
                match = self._placeholder_regex.search(s)
                if match is None:
                    break
                lo, hi = match.span()
                fragments.pop()
                fragments.append(s[:lo])
                N = int(s[lo + 1 : hi - 1])
                fragments.append(N)
                fragments.append(s[hi:])
                have_args = True

                assert N not in self.placeholder_args, f"Duplicate #{N}#"
                self.placeholder_args[N] = []
            log_action = None
            if have_args:
                # This will fail if a placeholder starts the line or
                # the start of the line is otherwise weired
                log_action = fragments[0].split()[0]
            self.parsed_lines.append(InlinePtxParsedLine(fragments, log_action))

        assert (
            self.placeholder_args
        ), "ptx_lines must have at least on placeholder, e.g. #0#"

    def add_arg(
        self,
        c_expr: Union[int, str, List[str]],
        *,
        constraint: str,
        log_as: Optional[str],
        N: int = 0,
        brackets: Optional[bool] = None,
    ):
        """Add an argument for the inline PTX

        c_expr: int, single C-syntax str arg, or sequence of C-syntax args.
        int values get treated as "true constants", directly pasted into PTX.
        Sequences are formatted as a PTX vector: {%0, %1, ... }

        IMPORTANT: args must not have side effects (e.g. "++my_arg"),
        otherwise the logging will be incorrect!

        constraint: PTX constraint (e.g. "=f", "l"), or special values:
          "generic": 64-bit pointer, real constraint: "l"
          "smem": pointer converted to 32-bit integer, real constraint: "r"

        log_as: Controls how the argument gets excut logged:
          None: don't log the arg
          "str_id": log the C expression passed to inline PTX as a string
          "bits": log as integer bits (float/pointers bit-cast to int)
          "ptr_data": dereference and log pointer data

        N: arg added to list used to substitute placeholder #N#

        brackets: Enable (true) or disable (false)
        wrapping PTX argument with [ ] (pointer syntax)

        """
        constraint_arg = constraint
        del constraint

        # Unpack constraint
        is_smem = False
        if constraint_arg == "smem":
            if brackets is None:
                brackets = True
            constraint_letter = "r"
            real_constraint = "r"
            mutable = False
            is_ptr = True
            is_smem = True
        elif constraint_arg == "generic":
            if brackets is None:
                brackets = True
            constraint_letter = "l"
            real_constraint = "l"
            mutable = False
            is_ptr = True
        elif len(constraint_arg) == 1:
            constraint_letter = constraint_arg
            real_constraint = constraint_arg
            mutable = False
            is_ptr = False
        elif len(constraint_arg) == 2:
            assert constraint_arg[0] == "=" or constraint_arg[0] == "+"
            constraint_letter = constraint_arg[1]
            real_constraint = constraint_arg
            mutable = True
            is_ptr = False
        else:
            assert 0, f"Unknown PTX constraint {constraint_arg!r}"

        # Convert all possible c_expr args to List[str], but record what
        # type it was originally as is_vector, true_const
        if isinstance(c_expr, int):
            is_vector = False
            c_expr = (str(c_expr),)
            true_const = True
        elif isinstance(c_expr, str):
            is_vector = False
            c_expr = (c_expr,)
            true_const = False
        else:
            assert all(
                isinstance(c, str) for c in c_expr
            ), "Expect int, str, or List[str]"
            is_vector = True
            true_const = False
            assert (
                not brackets
            ), "vector expression (c_expr: List[str]) cannot require brackets (PTX pointer syntax)"

        # Select exo_ExcutThreadLog member function, and formatter for
        # converting C expression to C logging argument.
        if log_as is None:
            log_fname = None
            as_logged = lambda s: ""
        elif log_as == "str_id":
            log_fname = "log_str_id_arg"
            as_logged = lambda s: f"{excut_c_str_id(s)}"
        elif log_as == "bits":
            # I try to pre-emptively support many cases even if I do not
            # actively use them myself so others don't have to hack this later.
            # But, rare paths are not tested, sorry if you have issues.
            if is_smem:
                log_fname = "log_u32_arg"
                as_logged = lambda s: f"exo_smemU32({s})"
            elif is_ptr:
                log_fname = "log_ptr_arg"
                as_logged = lambda s: s
            elif constraint_letter == "f":
                log_fname = "log_u32_arg"
                as_logged = lambda s: f"__float_as_uint({s})"
            elif constraint_letter == "d":
                log_fname = "log_u64_arg"
                as_logged = (
                    lambda s: f"static_cast<uint64_t>(__double_as_longlong({s}))"
                )
            elif constraint_letter == "l":
                log_fname = "log_u64_arg"
                as_logged = lambda s: f"static_cast<uint64_t>({s})"
            else:
                log_fname = "log_u32_arg"
                as_logged = lambda s: f"static_cast<uint32_t>({s})"
        elif log_as == "ptr_data":
            assert (
                is_ptr
            ), 'Cannot log non-pointer as ptr_data (set constraint "smem" or "generic")'
            log_fname = "log_ptr_data_arg"
            as_logged = lambda s: s

        # Translate to list of InlinePtxParsedArg and record
        # in {im}mutable_args, as appropriate, and to the list of
        # args associated with placeholder #N#
        placeholders = self.placeholder_args.get(N)
        assert placeholders is not None, f"No match for #{N}#"
        if true_const:
            args_list = None
            base_arg_index = 0
        elif mutable:
            args_list = self.mutable_args
            base_arg_index = len(args_list)
        else:
            args_list = self.immutable_args
            base_arg_index = len(args_list)

        for i, c in enumerate(c_expr):
            if is_smem:
                arg_expr = f"exo_smemU32({c})"
            else:
                arg_expr = c
            logged_expr = as_logged(c)

            arg = InlinePtxParsedArg(
                arg_expr,
                logged_expr,
                real_constraint,
                i + base_arg_index,
                mutable,
                true_const,
                log_fname,
            )

            if brackets:
                arg.ptx_prefix = "["
                arg.ptx_suffix = "]"
            elif is_vector:
                if i == 0:
                    arg.ptx_prefix = "{"
                if i + 1 == len(c_expr):
                    arg.ptx_suffix = "}"
                elif i % 8 != 7:
                    arg.advise_newline = False

            if args_list is not None:
                args_list.append(arg)
            placeholders.append(arg)

    def arg_ptx_fragment(self, parsed: InlinePtxParsedArg, py_format: bool):
        if parsed.true_const:
            s = parsed.c_arg
        elif parsed.mutable:
            s = f"%{parsed.arg_index}"
        else:
            s = f"%{parsed.arg_index + len(self.mutable_args)}"
        s = parsed.ptx_prefix + s + parsed.ptx_suffix
        if py_format:
            s = s.replace("{", "{{").replace("}", "}}")
        return s

    def as_c_lines(self, *, py_format: bool, tab="") -> List[str]:
        """Compile to list of C source code lines.

        If py_format is True, then we double all { and } in ptx_format
        so the generated code is usable in str.format. However; we do
        not do this transformation for anything passed to add_arg.

        LIMITATION: log_as="str_id" will not work well if PTX args are
        formatted! We'll log the Python format string, NOT the real
        (substituted) C expression. Hence why @instr now uses a
        codegen() function instead of the old str.format system.

        """
        c_ptx_lines = []
        c_log_lines = []

        # Populate c_ptx_lines and c_log_lines
        for ptx_lineno, parsed_line in enumerate(self.parsed_lines):
            logging = parsed_line.log_action is not None
            if logging:
                # HACK: currently hardwired 0 as ID of file name
                action_id = excut_c_str_id(parsed_line.log_action)
                c_log_lines.append(
                    f"exo_excutLog.log_action({action_id}, 0, __LINE__);"
                )

            def process_fragment(fragment):
                # Will be run over each str/int item in
                # InlinePtxParsedLine.fragments; replace int (#N#) with PTX
                # %[0-9] placeholders, and emit arg logging code along the way.
                if isinstance(fragment, str):
                    if py_format:
                        return fragment.replace("{", "{{").replace("}", "}}")
                    return fragment

                assert isinstance(fragment, int)
                args = self.placeholder_args[fragment]

                # Add to c_log_lines
                if logging:
                    for a in args:
                        stmt = a.c_log_stmt()
                        if stmt:
                            c_log_lines.append(stmt)

                # Substitute placeholder #N#
                return ", ".join(self.arg_ptx_fragment(a, py_format) for a in args)

            c_line = "".join(
                process_fragment(fragment) for fragment in parsed_line.fragments
            )
            nt = "" if ptx_lineno + 1 == len(self.parsed_lines) else "\\n\\t"
            c_ptx_lines.append(f'  "{c_line}{nt}"')

        def format_arg_lines(mutable):
            if not self.mutable_args and not self.immutable_args:
                return []

            args = self.mutable_args if mutable else self.immutable_args
            lines = [":"]
            for i, a in enumerate(args):
                assert not a.true_const
                assert a.mutable == mutable
                # Modify lines[-1], XXX this is really inefficient
                # if we have lots of args.
                s = lines.pop()
                comma = "" if i + 1 == len(args) else ", "
                s += f'"{a.constraint}"({a.c_arg}){comma}'
                lines.append(s)
                if a.advise_newline:
                    lines.append("")

            # Indent and remove incidental empty lines
            real_lines = []
            for ln in lines:
                if stripped := ln.strip():
                    real_lines.append("    " + stripped)
            return real_lines

        # Glue this mess together
        _volatile = " volatile" if self.volatile else ""
        lines = (
            [f"asm{_volatile}("]
            + c_ptx_lines
            + format_arg_lines(True)
            + format_arg_lines(False)
            + [");"]
            + c_log_lines
        )
        if tab:
            lines = [tab + ln for ln in lines]
        return lines

        # This code is really confusing, and that's why it exists.
        # Before, I was dealing with this sort of logic over-and-over, anywhere
        # I needed inline PTX, and it was a litany of %off-by-1 errors,
        # incorrectly escaped stuff, and so on...
