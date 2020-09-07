from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler

# top level compiler function called by tests!
def run_compile(proc_list,c_file,h_file):
    # take proc_list
    # for each p in proc_list:
    #   run Compiler() pass to get (decl, def)
    #
    # check for name conflicts between procs
    #
    # write out c_file and h_file

    decl = ""
    body = ""
    for p in proc_list:
        d, b = p.comp_top()
        fwd_decls += d
        body += b
    
    f_header = open(h_file, "w")
    f_header.write(fwd_delcs)
    f_header.close()

    f_cpp = open(c_file, "w")
    f_cpp.write(body)
    f_cpp.close()

def _eshape(typ,env):
    return tuple( r if is_pos_int(r) else env[r]
                  for r in typ.shape() )

def _simple_typecheck_buffer(typ, buf, env):
    if type(buf) is not np.ndarray:
        return False
    elif buf.dtype != float and buf.dtype != np.float64:
        return False

    if typ is T.R:
        if tuple(buf.shape) != (1,):
            return False
    else:
        shape = _eshape(typ,env)
        if shape != tuple(buf.shape):
            return False

    return True

class Compiler:
    def __init__(self, proc, use_randomization=False, **kwargs):
        assert type(proc) is LoopIR.proc

        self.proc   = proc
        self.env    = Environment()
        self.use_randomization = use_randomization

        # setup, size argument binding
        for sz in proc.sizes:
            if not str(sz) in kwargs:
                raise TypeError(f"expected size '{sz}' "
                                f"to be supplied")
            if not is_pos_int(kwargs[str(sz)]):
                raise TypeError(f"expected size '{sz}' to "
                                f"have positive integer value")
            self.env[sz] = kwargs[str(sz)]

        # setup, buffer argument binding
        for a in proc.args:
            if not str(a.name) in kwargs:
                raise TypeError(f"expected argument '{a.name}' "
                                f"to be supplied")
            if not _simple_typecheck_buffer(a.type, kwargs[str(a.name)],
                                            self.env):
                raise TypeError(f"type of argument '{a.name}' "
                                f"value mismatches")
            self.env[a.name] = kwargs[str(a.name)]

    def comp_top(self):
        self.env.push()
        stmt_str = self.comp_s(self.proc.body)
        self.env.pop()

        # Generate headers here?
        proc_def = (f"void {proc.name}(buffer_pointers)\n"+
                    f"\n"+
                    f"\n")
        #proc_decl = ()

        return proc_f, stmt_str

    def comp_s(self, s):
        styp    = type(s)

        if styp is LoopIR.Seq:
            first = self.comp_s(s.s0)
            second = self.comp_s(s.s1)
            
            return (f"{first}\n\n{second}")
        elif styp is LoopIR.Pass:
            return (f"; // NOP :")
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            # lbuf[a0,a1,...] = rhs
            lbuf = self.env[s.name]
            if len(s.idx) == 0:
                idx = (0,)
            else:
                idx  = tuple( self.comp_a(a) for a in s.idx )
            rhs  = self.comp_e(s.rhs)
            if styp is LoopIR.Assign:
                return (f"{lbuf}[{idx}] = {rhs}")
            else:
                return (f"{lbuf}[{idx}] += {rhs}")
        elif styp is LoopIR.If:
            cond = self.comp_p(s.cond)
            body = self.comp_s(s.body)
            return (f"if ({cond}) {{\n"+
                    f"{body}\n"+
                    f"}}\n")

            # TODO: Do we have to push env here??
            #self.env.push()
            #self.env.pop()
        elif styp is LoopIR.ForAll:
            hi      = self.env[s.hi] # this should be a string
            #itr    = self.new_varname(s.iter) # allocate a new string
            itr    = s.iter
            body    = self.comp_s(s.body)
            return (f"for (int {itr}=0; {itr} < {hi}; {itr}++) {{\n"+
                    f"{body}\n"+
                    f"}}")
        elif styp is LoopIR.Alloc:
            if s.type is T.R:
                self.env[s.name] = np.empty([1])
            else:
                size = _eshape(s.type, self.env)
                #TODO: Maybe randomize?
                self.env[s.name] = np.empty(size)
        else: assert False, "bad case"

    def comp_e(self, e):
        etyp    = type(e)

        if etyp is LoopIR.Read:
            buf = self.env[e.name]
            idx = ( (0,) if len(e.idx) == 0
                         else tuple( self.comp_a(a) for a in e.idx ))
            return buf[idx]
        elif etyp is LoopIR.Const:
            return e.val
        elif etyp is LoopIR.BinOp:
            lhs, rhs = self.comp_e(e.lhs), self.comp_e(e.rhs)
            if e.op == "+":
                return lhs + rhs
            elif e.op == "-":
                return lhs - rhs
            elif e.op == "*":
                return lhs * rhs
            elif e.op == "/":
                return lhs / rhs
        elif etyp is LoopIR.Select:
            cond    = self.comp_p(e.cond)
            return self.comp_e(e.body) if cond else 0.0
        else: assert False, "bad case"

    def comp_a(self, a):
        atyp    = type(a)

        if atyp is LoopIR.AVar or atyp is LoopIR.ASize:
            return self.env[a.name]
        elif atyp is LoopIR.AConst:
            return a.val
        elif atyp is LoopIR.AScale:
            return a.coeff * self.comp_a(a.rhs)
        elif atyp is LoopIR.AAdd:
            return self.comp_a(a.lhs) + self.comp_a(a.rhs)
        elif atyp is LoopIR.ASub:
            return self.comp_a(a.lhs) - self.comp_a(a.rhs)
        else: assert False, "bad case"

    def comp_p(self, p):
        ptyp = type(p)

        if ptyp is LoopIR.BConst:
            return p.val
        elif ptyp is LoopIR.Cmp:
            lhs, rhs = self.comp_a(p.lhs), self.comp_a(p.rhs)
            if p.op == "==":
                return (lhs == rhs)
            elif p.op == "<":
                return (lhs < rhs)
            elif p.op == ">":
                return (lhs > rhs)
            elif p.op == "<=":
                return (lhs <= rhs)
            elif p.op == ">=":
                return (lhs >= rhs)
            else: assert False, "bad case"
        elif ptyp is LoopIR.And or ptyp is LoopIR.Or:
            lhs, rhs = self.comp_p(p.lhs), self.comp_p(p.rhs)
            if ptyp is LoopIR.And:
                return (lhs and rhs)
            elif ptyp is LoopIR.Or:
                return (lhs or rhs)
            else: assert False, "bad case"
        else: assert False, "bad case"
