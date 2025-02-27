class SporkLoweringCtx(object):
    """Communication object between main LoopIR compiler and Spork backend.

    The task of the spork backend is to transform a subtree of LoopIR
    to a new subtree of LoopIR that the main compiler is able to
    understand.  Usually, the backend will return a tree rooted with
    an ExtWithContext to redirect the generated subtree C-like code to
    separate files for accelerator code (e.g. .cuh or .cu code for
    cuda).

    """

    __slots__ = [
        "_proc_name",
        "_kernel_index",
    ]

    def __init__(self, proc_name, kernel_index):
        self._proc_name = proc_name
        self._kernel_index = kernel_index

    def proc_name(self):
        return self._proc_name

    def kernel_index(self):
        return self._kernel_index
