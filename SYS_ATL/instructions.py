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
        wrapped by zero or more for-loops
        """
        self.itrs = []
        self.his  = []
        self.lidx = []
        self.ridx = []
        self.lbuf = None
        self.rbuf = None

        while type(subtree) is not LoopIR.Assign:
            if type(subtree) is LoopIR.ForAll:
                self.itrs.append(subtree.iter)
                self.his.append(subtree.hi)
                if len(subtree.body) != 1:
                    tc.err(subtree, pattern_err)
                subtree = subtree.body[0]
            else:
                tc.err(subtree, pattern_err)
                return
        assert type(subtree) is LoopIR.Assign
        if type(subtree.rhs) is not LoopIR.Read:
            tc.err(subtree.rhs, pattern_err)
            return
        self.lbuf, self.lidx = subtree.name, subtree.idx
        self.rbuf, self.ridx = subtree.rhs.name, subtree.rhs.idx

        if len(self.itrs) is not len(self.lidx)\
                    or len(self.lidx) is not len(self.ridx):
            tc.err("indices has to be the same size", subtree)

        #TODO: More typecheck here?
        # lbuf != rbuf?
        # len(lidx) == len(ridx)??


    #def memcheck(self, subtree):
    #    pass

    def compile(self, subtree, comp):
        his_comp  = [comp.comp_e(i) for i in self.his]
        itrs_comp = [comp.new_varname(i, typ=T.index) for i in self.itrs]
        rbuf_name = comp.env[self.rbuf]
        lbuf_name = comp.env[self.lbuf]
        lidx_comp = [comp.comp_e(i) for i in self.lidx]
        ridx_comp = [comp.comp_e(i) for i in self.ridx]

        # No idea how we can handle bounds checking here.
        # Ignore the if statement for now..
        res = ""
        res += f"//Move-in {rbuf_name} to {lbuf_name}\n"
        res += "gemmini_extended_config_ld(0, 1);\n"
        itr = lidx_comp[0]
        for i, n in zip(lidx_comp[1:], his_comp[1:]):
            itr = f"({itr})*{n}+({i})"

        spad = itr + "+" + str(self.sp_start_addr)
        self.sp_start_addr += 1
        # TODO: How to remember sp_start_addr??
        res += (f"gemmini_extended_mvin({rbuf_name} + "+
                f"({itr})*DIM, {spad}, DIM, DIM);\n")
        res += f"{self.lbuf}[{itr}] = {spad};\n"

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
