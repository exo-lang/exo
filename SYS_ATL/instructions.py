
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
        lbuf      = subtree.name
        rbuf      = subtree.rhs.name
        lbuf_name = comp.env[lbuf]
        rbuf_name = comp.env[rbuf]
        lidx      = subtree.idx
        ridx      = subtree.rhs.idx
        lidx_name = [comp.comp_a(i) for i in lidx]
        ridx_name = [comp.comp_a(i) for i in ridx]

        if len(itrs) is not len(lidx) or len(lidx) is not len(ridx):
            comp.err("indices has to be the same size", subtree)

        # No idea how we can handle bounds checking here.
        # Ignore the if statement for now..
        res = ""
        res += f"//Move-in {rbuf_name} to {lbuf_name}\n"
        res += "gemmini_extended_config_ld(0, 1);\n"
        itr = lidx_name[0]
        for i, n in zip(lidx_name[1:], his[1:]):
            itr = f"({itr})*{n}+({i})"

        spad = itr + "+" + str(self.sp_start_addr)
        self.sp_start_addr += 1
        # TODO: How to remember sp_start_addr??
        res += f"gemmini_extended_mvin(*{rbuf_name} + ({itr})*DIM, {spad}, DIM, DIM);\n"
        res += f"{lbuf}[{itr}] = {spad};\n"

        return res


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
