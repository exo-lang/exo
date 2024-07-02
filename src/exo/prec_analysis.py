from .LoopIR import LoopIR, LoopIR_Rewrite, T

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Default Precision Management

_default_prec = T.f32


def set_default_prec(name):
    global _default_prec
    vals = {
        "f16": T.f16,
        "f32": T.f32,
        "f64": T.f64,
        "i8": T.i8,
        "i32": T.i32,
    }
    if name not in vals:
        raise TypeError(
            f"Got {name}, but "
            "expected one of the following precision types: "
            + ",".join([k for k in vals])
        )
    _default_prec = vals[name]


def get_default_prec():
    return _default_prec


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Precision Analysis Pass


class PrecisionAnalysis(LoopIR_Rewrite):
    def __init__(self):
        self._errors = []
        self._types = {}
        self.default = get_default_prec()

    def run(self, proc):
        assert isinstance(proc, LoopIR.proc)
        proc = super().apply_proc(proc)
        if self._errors:
            errs = "\n".join(self._errors)
            raise TypeError(f"Errors occurred during precision checking:\n{errs}")
        return proc

    def err(self, node, msg):
        self._errors.append(f"{node.srcinfo}: {msg}")

    def set_type(self, name, typ):
        assert name not in self._types
        self._types[name] = typ

    def get_type(self, name, default=None):
        if name not in self._types and default is not None:
            return default
        else:
            return self._types[name]

    def splice_type(self, t, bt):
        if t.is_real_scalar():
            return bt
        elif isinstance(t, T.Tensor):
            return T.Tensor(t.hi, t.is_window, self.splice_type(t.type, bt))
        elif isinstance(t, T.Window):
            return T.Window(
                self.splice_type(t.src_type, bt),
                self.splice_type(t.as_tensor, bt),
                t.src_buf,
                t.idx,
            )
        else:
            return t

    def map_fnarg(self, a):
        typ = a.type
        if typ.is_numeric() and typ.basetype() == T.R:
            typ = self.splice_type(typ, self.default)

        self.set_type(a.name, typ)

        return LoopIR.fnarg(a.name, typ, a.mem, a.srcinfo)

    def map_s(self, s):
        # always do standard sub-processing
        # and then possibly patch up the results
        # in a post-traversal sort of way below
        # before returning
        result = super().map_s(s)
        if result is None:
            result = [s]
        if isinstance(s, LoopIR.Call):
            assert len(result) == 1

            # check call arguments for precision consistency...
            args = result[0].args
            for call_a, sig_a in zip(args, s.f.args):
                ct = call_a.type.basetype()
                st = sig_a.type.basetype()
                st = self.default if st == T.R else st
                if st.is_numeric() and st != ct:
                    self.err(call_a, f"expected precision {st}, but got {ct}")

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            rtyp = result[0].rhs.type
            ltyp = self.get_type(s.name).basetype()
            assert ltyp != T.err and ltyp != T.R

            # update the type annotation here if needed
            result[0] = result[0].update(type=ltyp)

            # potentially coerce the entire right-hand-side
            if rtyp != T.err:
                # potentially coerce the entire right-hand-side
                if rtyp == T.R:
                    result[0] = result[0].update(rhs=self.coerce_e(result[0].rhs, ltyp))
                    rtyp = ltyp

        elif isinstance(s, LoopIR.WriteConfig):
            rtyp = result[0].rhs.type
            ltyp = s.config.lookup_type(s.field)
            assert ltyp != T.err and ltyp != T.R

            # potentially coerce the entire right-hand-side
            if rtyp != T.err:
                if rtyp == T.R:
                    result[0] = result[0].update(rhs=self.coerce_e(result[0].rhs, ltyp))

        elif isinstance(s, LoopIR.WindowStmt):
            # update the type binding for this symbol...
            self.set_type(result[0].name, result[0].rhs.type)

        elif isinstance(s, LoopIR.Alloc):
            typ = result[0].type
            if s.type.basetype() == T.R:
                typ = self.splice_type(s.type, self.default)
            result[0] = result[0].update(type=typ)
            self.set_type(s.name, typ)

        return result

    def map_e(self, e):
        if isinstance(e, LoopIR.Read):
            typ = e.type
            if typ.is_numeric():
                btyp = self.get_type(e.name).basetype()
                typ = self.splice_type(e.type, btyp)
            return LoopIR.Read(e.name, e.idx, typ, e.srcinfo)

        elif isinstance(e, LoopIR.WindowExpr):
            btyp = self.get_type(e.name).basetype()
            wtyp = self.splice_type(e.type, btyp)
            return LoopIR.WindowExpr(e.name, e.idx, wtyp, e.srcinfo)

        elif isinstance(e, LoopIR.USub):
            arg = self.apply_e(e.arg)

            if not e.type.is_numeric():
                return LoopIR.USub(arg, e.type, e.srcinfo)

            assert arg.type.is_real_scalar() or arg.type == T.err
            return LoopIR.USub(arg, arg.type, e.srcinfo)

        elif isinstance(e, LoopIR.BinOp):
            lhs = self.apply_e(e.lhs)
            rhs = self.apply_e(e.rhs)

            # first let's get the index expressions
            # and booleans out of the way
            if not e.type.is_numeric():
                return LoopIR.BinOp(e.op, lhs, rhs, e.type, e.srcinfo)

            assert (lhs.type == T.err or lhs.type.is_real_scalar()) and (
                rhs.type == T.err or rhs.type.is_real_scalar()
            )
            if lhs.type == T.err or rhs.type == T.err:
                typ = T.err
            elif lhs.type == T.R and rhs.type == T.R:
                typ = T.R
            elif lhs.type == T.R:
                typ = rhs.type
                lhs = self.coerce_e(lhs, typ)
            elif rhs.type == T.R:
                typ = lhs.type
                rhs = self.coerce_e(rhs, typ)
            elif lhs.type != rhs.type:  # no T.R or T.err left, so...
                self.err(
                    e,
                    f"cannot compute operation '{e.op}' between "
                    f"inconsistent precision types: "
                    f"{lhs.type} and {rhs.type}",
                )
                typ = T.err
            else:
                typ = lhs.type
            return LoopIR.BinOp(e.op, lhs, rhs, typ, e.srcinfo)

        return super().map_e(e)

    # this routine allows for us to retro-actively
    # induce appropriate casts onto T.R typed constants at
    # the leaves of numeric expressions
    def coerce_e(self, e, btyp):
        if isinstance(e, LoopIR.Const):
            assert e.type == btyp or e.type == T.R
            return LoopIR.Const(e.val, btyp, e.srcinfo)
        elif isinstance(e, LoopIR.USub):
            arg = e.arg
            if arg.type == T.R:
                arg = self.coerce_e(arg, btyp)
            assert arg.type == btyp
            return LoopIR.USub(arg, btyp, e.srcinfo)
        elif isinstance(e, LoopIR.BinOp):
            lhs, rhs = e.lhs, e.rhs
            if lhs.type == T.R:
                lhs = self.coerce_e(lhs, btyp)
            if rhs.type == T.R:
                rhs = self.coerce_e(rhs, btyp)
            assert lhs.type == btyp
            assert rhs.type == btyp

            return LoopIR.BinOp(e.op, lhs, rhs, btyp, e.srcinfo)
        else:
            assert False, f"Should not be coercing a {type(e)} Node"

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_t(self, t):
        return None
