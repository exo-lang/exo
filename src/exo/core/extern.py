import math

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Extern superclass


class Extern_Typecheck_Error(Exception):
    def __init__(self, msg):
        self._builtin_err_msg = str(msg)

    def __str__(self):
        return self._builtin_err_msg


_EErr = Extern_Typecheck_Error


class Extern:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def globl(self, prim_type):
        raise NotImplementedError()

    def typecheck(self, args):
        raise NotImplementedError()

    def interpret(self, args):
        raise NotImplementedError()

    def compile(self, args, prim_type):
        raise NotImplementedError()
