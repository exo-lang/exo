from ..core.LoopIR import LoopIR, LoopIR_Rewrite
from ..spork.loop_mode import LoopMode, Seq, Par

from ..rewrite.new_eff import Check_ParallelizeLoop, SchedulingError

import warnings


class ParallelAnalysis(LoopIR_Rewrite):
    def __init__(self):
        self._errors = []

    def run(self, proc):
        assert isinstance(proc, LoopIR.proc)
        self.proc = proc
        proc = super().apply_proc(proc)
        if self._errors:
            errs = "\n".join(self._errors)
            raise TypeError(f"Errors occurred during precision checking:\n{errs}")
        return proc

    def err(self, node, msg):
        self._errors.append(f"{node.srcinfo}: {msg}")

    def map_s(self, s):
        if isinstance(s, LoopIR.For) and not isinstance(s.loop_mode, Seq):
            try:
                if isinstance(s.loop_mode, Par):
                    Check_ParallelizeLoop(self.proc, s)
                else:
                    warnings.warn(
                        f"Not implemented: data race check for {type(s.loop_mode)}"
                    )
            except SchedulingError:
                self.err(
                    s,
                    "parallel loop's body is not parallelizable because of potential data races",
                )
