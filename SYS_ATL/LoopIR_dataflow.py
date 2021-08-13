from .prelude import *
from .LoopIR import LoopIR, LoopIR_Do

from collections import defaultdict, ChainMap



# data-flow dependencies between variable names
class LoopIR_Dependencies(LoopIR_Do):
    def __init__(self, buf_sym, stmts):
        self._buf_sym = buf_sym
        self._lhs     = None
        self._depends = defaultdict(lambda: set())
        self._alias   = defaultdict(lambda: None)

        self.do_stmts(stmts)

    def result(self):
        depends = self._depends[self._buf_sym]
        new     = list(depends)
        done    = []
        while True:
            if len(new) == 0:
                break
            sym = new.pop()
            done.append(sym)
            d = self._depends[sym]
            depends.update(d)
            [ new.append(s) for s in d if s not in done ]

        return depends

    def do_s(self, s):
        if type(s) is LoopIR.Assign or type(s) is LoopIR.Reduce:
            lhs         = self._alias[s.name] or s.name
            self._lhs   = lhs
            self._depends[lhs].add(lhs)
        elif type(s) is LoopIR.WindowStmt:
            rhs_buf     = self._alias[s.rhs.name] or s.rhs.name
            self._alias[s.lhs] = rhs_buf
            self._lhs   = rhs_buf
            self._depends[rhs_buf].add(rhs_buf)
        elif type(s) is LoopIR.Call:
            # internal dependencies of each argument
            for a in s.args:
                self.do_e(a)
            # giant cross-bar of dependencies on the arguments
            for fa, a in zip(s.f.args, s.args):
                if fa.type.is_numeric():
                    name = self._alias[a.name] or a.name
                    # handle any potential indexing of this variable
                    for aa in s.args:
                        if aa.type.is_indexable():
                            self.do_e(aa)
                    # if this buffer is being written,
                    # then handle dependencies on other buffers
                    maybe_write = self.analyze_eff(s.f.eff, fa.name,
                                                   write=True)
                    if maybe_write:
                        self._lhs = name
                        for faa, aa in zip(s.f.args, s.args):
                            if faa.type.is_numeric():
                                maybe_read  = self.analyze_eff(s.f.eff, faa.name,
                                                               read=True)
                                if maybe_read:
                                    self.do_e(aa)
                        self._lhs = None

            # already handled all sub-terms above
            # don't do the usual statement processing
            return

        super().do_s(s)
        self._lhs = None

    def analyze_eff(self, eff, buf, write=False, read=False):
        if read:
            if any(es.buffer == buf for es in eff.reads):
                return True
        if write:
            if any(es.buffer == buf for es in eff.writes):
                return True
        if read or write:
            if any(es.buffer == buf for es in eff.reduces):
                return True

        return False

    def do_e(self, e):
        if type(e) is LoopIR.Read or type(e) is LoopIR.WindowExpr:
            def visit_idx(e):
                if type(e) is LoopIR.Read:
                    for i in e.idx:
                        self.do_e(i)
                else:
                    for w in e.idx:
                        if type(w) is LoopIR.Interval:
                            self.do_e(w.lo)
                            self.do_e(w.hi)
                        else:
                            self.do_e(w.pt)

            lhs     = self._lhs
            name    = self._alias[e.name] or e.name
            self._lhs = name
            if lhs:
                self._depends[lhs].add(name)
            visit_idx(e)
            self._lhs = lhs
            visit_idx(e)

        else:
            super().do_e(e)

    def do_t(self, t):
        pass
    def do_eff(self, eff):
        pass

