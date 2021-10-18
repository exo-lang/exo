from collections import defaultdict
from weakref import WeakKeyDictionary

from .LoopIR import LoopIR, LoopIR_Do

_WC_Leaf = {}  # unique object to use as key....
class WeakCache(WeakKeyDictionary):
    def __init__(self):
        self._tuple_dict = WeakKeyDictionary()
        self._dict       = WeakKeyDictionary()

    def __contains__(self, key):
        if isinstance(key, (tuple, list)):
            lookup = self._tuple_dict
            for k in key:
                if k not in lookup:
                    return False
                else:
                    lookup = lookup[k]
            return _WC_Leaf in lookup
        else:
            return key in self._dict

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            lookup = self._tuple_dict
            for k in key:
                lookup = lookup[k]
            return lookup[_WC_Leaf]
        else:
            return self._dict[key]

    def __setitem__(self, key, value):
        if isinstance(key, (tuple, list)):
            lookup = self._tuple_dict
            for k in key:
                if k not in lookup:
                    lookup[k] = WeakKeyDictionary()
                lookup = lookup[k]
            lookup[_WC_Leaf] = value
        else:
            self._dict[key] = value


#
# So, what is dependency analysis?
# Or to put it another way, what extensional property(s)
# does dependency analysis guarantee?
#
# Let B be a block of statements,
#     s be a store, and
#     x, y, … be names/symbols.
# Let FV(B) be the set of names that are free in B
#
# Then, first observe that the "meaning" of B is
#
#   Exec[[B]] : (FV(B) -> Value) -> Store -> Store
# 
# (note that (FV(B) -> Value) is a valuation/mapping specifying the values
#       of all free variables)
# (further note that Store = (Name -> Maybe Value) is a valuation/mapping
#       of variables that models the heap/store)
# 
# Then, (not x DependsOn y in B) for some y in FV(B) implies that
#
#   (Exec[[B]] (env[ y := v1 ]) s)[x] =
#   (Exec[[B]] (env[ y := v2 ]) s)[x]
#
# for all v1, v2
#
# Or in other words, the meaning of B
# w.r.t. its effect on x
# is invariant to the value of y
# when x does not depend on y in B
#


# data-flow dependencies between variable names
class LoopIR_Dependencies(LoopIR_Do):
    def __init__(self, buf_sym, stmts):
        self._buf_sym   = buf_sym
        self._lhs       = None
        self._depends   = defaultdict(set)
        self._alias     = dict()

        # If `lhs` is not None, then `lhs` will become dependent
        # on anything read.
        self._lhs       = None

        # variables that affect whether or not the
        # currently examined code is even running
        self._context   = set()

        # If `control` is True, then anything read will be added
        # to `context`.
        self._control   = False

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
            new.extend(s for s in d if s not in done)

        return depends

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            lhs       = self._alias.get(s.name, s.name)
            self._lhs = lhs
            self._depends[lhs].add(lhs)
            self._depends[lhs].update(self._context)
            for i in s.idx:
                self.do_e(i)
            self.do_e(s.rhs)
            self._lhs = None
        elif isinstance(s, LoopIR.WriteConfig):
            lhs       = (s.config, s.field)
            self._lhs = lhs
            self._depends[lhs].add(lhs)
            self._depends[lhs].update(self._context)
            self.do_e(s.rhs)
            self._lhs = None
        elif isinstance(s, LoopIR.WindowStmt):
            rhs_buf = self._alias.get(s.rhs.name, s.rhs.name)
            self._alias[s.lhs] = rhs_buf
            self._lhs   = rhs_buf
            self._depends[rhs_buf].add(rhs_buf)
            self.do_e(s.rhs)
            self._lhs   = None

        elif isinstance(s, LoopIR.If):
            old_context   = self._context
            self._context = old_context.copy()

            self._control = True
            self.do_e(s.cond)
            self._control = False

            self.do_stmts(s.body)
            self.do_stmts(s.orelse)

            self._context = old_context

        elif isinstance(s, (LoopIR.ForAll, LoopIR.Seq)):
            old_context   = self._context
            self._context = old_context.copy()

            self._control = True
            self._lhs     = s.iter
            self._depends[s.iter].add(s.iter)
            self.do_e(s.hi)
            self._lhs     = None
            self._control = False

            self.do_stmts(s.body)

            self._context = old_context

        elif isinstance(s, LoopIR.Call):

            def process_reads():
                # now handle dependencies on buffers that might
                # be read from in the sub-procedure
                # and dependencies on other arguments
                for faa, aa in zip(s.f.args, s.args):
                    if faa.type.is_numeric():
                        maybe_read = self.analyze_eff(s.f.eff, faa.name,
                                                      read=True)
                    else:
                        maybe_read = True

                    if maybe_read:
                        self.do_e(aa)

                # additionally, we need to handle dependencies
                # on configuration fields
                for ce in s.f.eff.config_reads:
                    name = (ce.config, ce.field)
                    if self._lhs:
                        self._depends[self._lhs].add(name)

            # for every argument that represents a buffer being
            # written to
            for fa, a in zip(s.f.args, s.args):
                maybe_write = ( fa.type.is_numeric() and
                                self.analyze_eff(s.f.eff, fa.name,
                                                 write=True) )
                if maybe_write:
                    name = self._alias.get(a.name, a.name)
                    self._lhs = name
                    self._depends[name].add(name)
                    self._depends[name].update(self._context)
                    process_reads()
                    self._lhs = None

            # secondly, for every configuration field being written to
            # by this sub-procedure, we need to determine dependencies
            for ce in s.f.eff.config_writes:
                name = (ce.config, ce.field)
                self._lhs = name
                self._depends[name].add(name)
                self._depends[name].update(self._context)
                process_reads()
                self._lhs = None

        elif isinstance(s, (LoopIR.Pass, LoopIR.Alloc)):
            pass
        else:
            assert False, "bad case"

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
        if isinstance(e, (LoopIR.Read, LoopIR.WindowExpr)):
            def visit_idx(e):
                if isinstance(e, LoopIR.Read):
                    for i in e.idx:
                        self.do_e(i)
                else:
                    for w in e.idx:
                        if isinstance(w, LoopIR.Interval):
                            self.do_e(w.lo)
                            self.do_e(w.hi)
                        else:
                            self.do_e(w.pt)

            name = self._alias.get(e.name, e.name)
            if self._lhs:
                self._depends[self._lhs].add(name)
            if self._control:
                self._context.add(name)

            visit_idx(e)

        elif isinstance(e, LoopIR.ReadConfig):
            name = (e.config, e.field)
            if self._lhs:
                self._depends[self._lhs].add(name)
            if self._control:
                self._context.add(name)

        else:
            super().do_e(e)

    def do_t(self, t):
        pass

    def do_eff(self, eff):
        pass
