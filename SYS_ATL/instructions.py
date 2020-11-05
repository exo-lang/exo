
from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

from .instruction_type import Instruction

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Specific Instructions

Instr_Lookup  = dict()
def newinstr(cls):
    Instr_Lookup[cls.__name__] = cls
    return cls

@newinstr
class GEMM_Load(Instruction):
    def __init__(self):
        self.sp_start_addr = 0

    def typecheck(self, subtree, tc):
        # General form is zero or more loops around an assignment
        # whose right-hand-side is simply a read.
        # Pattern match the above form to extract the following
        pattern_err = """
        expected a Load instruction to be a simple copy assignment
        wrapped by a if and zero or more for-loops
        """
        itrs = []
        his  = []
        lidx = []
        ridx = []
        lbuf = None
        rbuf = None
        cond = None
        while type(subtree) is not LoopIR.If:
            if type(subtree) is LoopIR.ForAll:
                itrs.append(subtree.iter)
                his.append(subtree.hi)
                subtree = subtree.body
            else:
                tc.err(subtree, pattern_err)
                return

        assert type(subtree) is LoopIR.If
        cond = subtree.cond
        subtree = subtree.body

        assert type(subtree) is LoopIR.Assign
        if type(subtree.rhs) is not LoopIR.Read:
            tc.err(subtree.rhs, pattern_err)
            return
        lbuf, lidx = subtree.name, subtree.idx
        rbuf, ridx = subtree.rhs.name, subtree.rhs.idx

        #TODO: More typecheck here?
        # lbuf != rbuf?
        # len(lidx) == len(ridx)??


    #def memcheck(self, subtree):
    #    pass

    def compile(self, subtree, comp):
        itrs = []
        his  = []
        lidx = []
        ridx = []
        lbuf = None
        rbuf = None
        cond = None
        while type(subtree) is not LoopIR.If:
            if type(subtree) is LoopIR.ForAll:
                itrs.append(comp.new_varname(subtree.iter))
                his.append(comp.comp_a(subtree.hi))
                subtree = subtree.body
            else:
                tc.err(subtree, pattern_err)
                return

        assert type(subtree) is LoopIR.If
        cond    = subtree.cond
        subtree = subtree.body

        assert type(subtree) is LoopIR.Assign
        if type(subtree.rhs) is not LoopIR.Read:
            tc.err(subtree.rhs, pattern_err)
            return
        lbuf = comp.env[subtree.name]
        rbuf = comp.env[subtree.rhs.name]
        lidx = [comp.comp_a(i) for i in subtree.idx]
        ridx = [comp.comp_a(i) for i in subtree.rhs.idx]

        if len(itrs) is not len(lidx) or len(lidx) is not len(ridx):
            comp.err("indices has to be the same size", subtree)

        if len(itrs) != 1:
            raise NotImplementedError()

        # No idea how we can handle bounds checking here.
        # Ignore the if statement for now..
        res = ""
        res += f"// Move-in {rbuf} to {lbuf}\n"
        res += "gemmini_extended_config_ld(0, 1);\n"
        oidx = lidx[0]
        spad = oidx + "+" + str(self.sp_start_addr)
        self.sp_start_addr += 1
        # TODO: How to remember sp_start_addr??
        res += f"gemmini_extended_mvin(*({rbuf}) + {oidx}*DIM, {spad}, 1, DIM);\n"
        res += f"{lbuf}[{oidx}] = {spad};\n"

        return res
"""
    def access_str(self, nm, idx_list):
        buf = self.env[nm]
        type = self.envtyp[nm]
        idxs = [self.comp_a(i) for i in idx_list]
        idx = _type_idx(type, idxs, self.env)
        return f"{buf}[{idx}]"


        ptr : R[n] @ HEAP
        for i in par(0,n/16):
            instr(GEMM_Load)
            for i2 in par(0,16):
                if i*16+i2 < n:
                    ptr[i] = x[i*16+i2]
        =>
        for i in par(0,n/16):
            gemmni_extended_mvin(*x + i*DIM, i+sp_start_addr, 1, rows)
            ptr[i] = i+sp_start_addr
"""


@newinstr
class GEMM_Store(Instruction):
    """
    def __init__(self):
        pass

    def typecheck(self, subtree):
        # check that subtree is a copy loop?
        pass

    def memcheck(self, subtree):
        pass

    def compile(self):
        return "GEMM_Load(...);"
    """

@newinstr
class GEMM_Mul(Instruction):
    """
    def __init__(self):
        pass

    def typecheck(self, subtree):
        # check that subtree is a copy loop?
        pass

    def memcheck(self, subtree):
        pass

    def compile(self):
        return "GEMM_Load(...);"
    """
