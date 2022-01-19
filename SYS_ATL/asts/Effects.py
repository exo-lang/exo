from typing import List, Optional

import attrs
from attrs import validators

from . import OP_STRINGS
from .LoopIR import type as LoopIR_type
from .configs import Config
from ..prelude import SrcInfo, Sym

_op_prec = {
    "ternary": 5,
    #
    "or":      10,
    #
    "and":     20,
    #
    "<":       30,
    ">":       30,
    "<=":      30,
    ">=":      30,
    "==":      30,
    #
    "+":       40,
    "-":       40,
    #
    "*":       50,
    "/":       50,
    "%":       50,
    #
    # unary - 60
}


@attrs.frozen
class expr:
    def __str__(self):
        return self._exprstr()

    def _exprstr(self, prec=0):
        raise NotImplementedError('missing case')


# JRK: the notation of this comprehension is confusing - maybe just use math:
# this corresponds to `{ buffer : loc for *names in int if pred }`
@attrs.frozen
class effset:
    buffer: Sym
    loc: List[expr]  # e.g. reading at (i+1,j+1)
    names: List[Sym]
    pred: Optional[expr]
    srcinfo: SrcInfo


@attrs.frozen
class config_eff:
    config: Config
    field: str
    value: Optional[expr]  # need not be supplied for reads
    pred: Optional[expr]
    srcinfo: SrcInfo


@attrs.frozen
class effect:
    reads: List[effset]
    writes: List[effset]
    reduces: List[effset]
    config_reads: List[config_eff]
    config_writes: List[config_eff]
    srcinfo: SrcInfo

    def __str__(self):
        def esstr(es, tab="  "):
            lines = []
            buf = str(es.buffer)
            loc = "(" + ','.join([str(l) for l in es.loc]) + ")"
            if len(es.names) == 0:
                names = ""
            else:
                names = f"for ({','.join([str(n) for n in es.names])}) in Z"

            if es.pred is None:
                lines.append(f"{tab}{{ {buf} : {loc} {names} }}")
            else:
                lines.append(f"{tab}{{ {buf} : {loc} {names} if")
                tab += "  "
                pred = str(es.pred)
                lines.append(f"{tab}{pred} }}")

            return '\n'.join(lines)

        def cestr(ce):
            val, pred = "", ""
            if ce.value:
                val = f" = {ce.value}"
            if ce.pred:
                pred = f" if {ce.pred}"
            return f"{ce.config.name()}.{ce.field}{val}{pred}"

        eff_str = ""
        if self.reads:
            eff_str += "Reads:\n"
            eff_str += '\n'.join([esstr(es) for es in self.reads])
            eff_str += "\n"
        if self.writes:
            eff_str += f"Writes:\n  "
            eff_str += '\n'.join([esstr(es) for es in self.writes])
            eff_str += "\n"
        if self.reduces:
            eff_str += f"Reduces:\n  "
            eff_str += '\n'.join([esstr(es) for es in self.reduces])
            eff_str += "\n"
        if self.config_reads:
            eff_str += f"Config Reads:\n"
            eff_str += '\n'.join([cestr(ce) for ce in self.config_reads])
            eff_str += "\n"
        if self.config_writes:
            eff_str += f"Config Writes:\n"
            eff_str += '\n'.join([cestr(ce) for ce in self.config_writes])
            eff_str += "\n"

        return eff_str


@attrs.frozen
class Var(expr):
    name: Sym
    type: LoopIR_type
    srcinfo: SrcInfo

    def _exprstr(self, prec=0):
        return str(self.name)


@attrs.frozen
class Not(expr):
    arg: expr
    type: LoopIR_type
    srcinfo: SrcInfo

    def _exprstr(self, prec=0):
        return f'(not {self.arg})'


@attrs.frozen
class Const(expr):
    val: object
    type: LoopIR_type
    srcinfo: SrcInfo

    def _exprstr(self, prec=0):
        return str(self.val)


@attrs.frozen
class BinOp(expr):
    op: str = attrs.field(validator=validators.in_(OP_STRINGS))
    lhs: expr
    rhs: expr
    type: LoopIR_type
    srcinfo: SrcInfo

    def _exprstr(self, prec=0):
        local_prec = _op_prec[self.op]
        lhs = self.lhs._exprstr(local_prec)
        rhs = self.rhs._exprstr(local_prec + 1)

        full_expr = f'{lhs} {self.op} {rhs}'
        if local_prec < prec:
            return f'({full_expr})'
        return full_expr


@attrs.frozen
class Stride(expr):
    name: Sym
    dim: int
    type: LoopIR_type
    srcinfo: SrcInfo

    def _exprstr(self, prec=0):
        return f'stride({self.name}, {self.dim})'


@attrs.frozen
class Select(expr):
    cond: expr
    tcase: expr
    fcase: expr
    type: LoopIR_type
    srcinfo: SrcInfo

    def _exprstr(self, prec=0):
        local_prec = _op_prec['ternary']
        cond = self.cond._exprstr()
        tcase = self.tcase._exprstr(local_prec + 1)
        fcase = self.fcase._exprstr(local_prec + 1)

        full_expr = f'({cond}) ? {tcase} : {fcase}'
        if local_prec < prec:
            return f'({full_expr})'
        return full_expr


@attrs.frozen
class ConfigField(expr):
    config: Config
    field: str
    type: LoopIR_type
    srcinfo: SrcInfo

    def _exprstr(self, prec=0):
        return f'{self.config.name()}.{self.field}'
