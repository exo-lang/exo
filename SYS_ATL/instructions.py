
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
    def typecheck(self, subtree, tc):
        # General form is zero or more loops around an assignment
        # whose right-hand-side is simply a read.
        # Pattern match the above form to extract the following
        pattern_err = """
        expected a Load instruction to be a simple copy assignment
        wrapped by zero or more for-loops
        """
        itrs = []
        his  = []
        lidx = []
        ridx = []
        lbuf = None
        rbuf = None
        while type(subtree) is not LoopIR.Assign:
            if type(subtree) is LoopIR.ForAll:
                itrs.append(subtree.iter)
                his.append(subtree.hi)
                subtree = subtree.body
            else:
                tc.err(subtree, pattern_err)
                return
        assert type(subtree) is LoopIR.Assign
        if type(subtree.rhs) is not LoopIR.Read:
            tc.err(subtree.rhs, pattern_err)
            return
        lbuf, lidx = subtree.name, subtree.idx
        rbuf, ridx = subtree.rhs.name, subtree.rhs.idx



        # loop around a stmt?
        # check that subtree is a copy loop?
        pass

    #def memcheck(self, subtree):
    #    pass

    def compile(self, subtree, comp):
        return "GEMM_Load(...);"

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
