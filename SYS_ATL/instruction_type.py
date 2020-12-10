from .prelude import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# List of valid memories

MEM_NAMES = {
    "GEMM",
    "HEAP",
}

def is_valid_mem(x):
    return x in MEM_NAMES


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The Instruction Object

class Instruction:
    def opname(self):
        return type(self).__name__

    def typecheck(self, subtree, typechecker):
        """
        Should check whether or not the subtree (LoopIR stmt)
        matches the template specified by Instruction
        """
        raise NotImplementedError()

    def memcheck(self, subtree, memchecker):
        """
        TO BE DESIGNED WITH MEM CHECKING PASS
        """
        raise NotImplementedError()

    def compile(self, subtree, compiler):
        """
        Should return a string to be spliced into compiled C-code
        """
        raise NotImplementedError()
