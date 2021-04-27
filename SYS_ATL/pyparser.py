from __future__ import annotations

import re
import types
import inspect
import ast as pyast
import astor
import textwrap
import sys

from .prelude import *
from .LoopIR import UAST, front_ops, PAST
#from .LoopIR import T

from collections import ChainMap

from .API import Procedure

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Helpers


class ParseError(Exception):
    pass


class SizeStub:
    def __init__(self, nm):
        assert type(nm) is Sym
        self.nm = nm

def str_to_mem(name):
    return getattr(sys.modules[__name__], name)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Top-level decorator

def instr(instruction, _testing=None):
    if type(instruction) is not str:
        raise TypeError("@instr decorator must be @instr(<your instuction>)")
    def inner(f):
        if type(f) is not types.FunctionType:
            raise TypeError("@instr decorator must be applied to a function")

        return proc(f, _instr=instruction, _testing=_testing)

    return inner

def proc(f, _instr=None,_testing=None):
    if type(f) is not types.FunctionType:
        raise TypeError("@proc decorator must be applied to a function")

    # note that we must dedent in case the function is defined
    # inside of a local scope
    rawsrc = inspect.getsource(f)
    src = textwrap.dedent(rawsrc)
    n_dedent = (len(re.match('^(.*)', rawsrc).group()) -
                len(re.match('^(.*)', src).group()))
    srcfilename = inspect.getsourcefile(f)
    _, srclineno = inspect.getsourcelines(f)
    srclineno -= 1  # adjust for decorator line

    # convert into AST nodes; which should be a module with a single
    # FunctionDef node
    module = pyast.parse(src)
    assert len(module.body) == 1
    assert type(module.body[0]) == pyast.FunctionDef

    # get global and local environments for context capture purposes
    func_globals = f.__globals__
    stack_frames = inspect.stack()
    assert(len(stack_frames) >= 1)
    assert(type(stack_frames[1]) == inspect.FrameInfo)
    func_locals = stack_frames[1].frame.f_locals
    assert(type(func_locals) == dict)
    srclocals = ChainMap(func_locals)

    # patch in Built-In functions to scope
    # srclocals['name'] = object

    # create way to query for src-code information
    def getsrcinfo(node):
        return SrcInfo(filename=srcfilename,
                       lineno=node.lineno+srclineno,
                       col_offset=node.col_offset+n_dedent,
                       end_lineno=(None if node.end_lineno is None
                                   else node.end_lineno+srclineno),
                       end_col_offset=(None if node.end_col_offset is None
                                       else node.end_col_offset+n_dedent))

    # try to attribute parse error messages with minimal internal
    # stack traces...
    # try:
    parser = Parser(module.body[0], func_globals,
                    srclocals, getsrcinfo,
                    instr   =_instr,
                    as_func =True)
    # except ParseError as pe:
    #  pemsg = "Encountered error while parsing decorated function:\n"+str(pe)
    #  raise ParseError(pemsg) from pe

    return Procedure(parser.result(), _testing=_testing)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pattern-Parser top-level, invoked on strings rather than as a decorator

def pattern(s, filename=None, lineno=None):
    assert type(s) is str

    src = s
    n_dedent = 0
    if filename is not None:
        srcfilename = filename
        srclineno   = lineno
    else:
        srcfilename = "string"
        srclineno   = 0

    module = pyast.parse(src)
    assert type(module) == pyast.Module

    # create way to query for src-code information
    def getsrcinfo(node):
        return SrcInfo(filename=srcfilename,
                       lineno=node.lineno+srclineno,
                       col_offset=node.col_offset+n_dedent,
                       end_lineno=(None if node.end_lineno is None
                                   else node.end_lineno+srclineno),
                       end_col_offset=(None if node.end_col_offset is None
                                       else node.end_col_offset+n_dedent))

    parser = PatternParser(module.body, getsrcinfo)
    return parser.result()


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Parser Pass object


class Parser:
    def __init__(self, module_ast, func_globals, srclocals, getsrcinfo,
                 as_func=False, as_macro=False,
                 as_quote=False, as_index=False, instr=None):

        self.module_ast = module_ast
        self.globals = func_globals
        self.locals = srclocals
        self.getsrcinfo = getsrcinfo

        self.as_index = as_index

        self.push()
        if as_func:
            self._cached_result = self.parse_fdef(module_ast, instr=instr)
        # elif as_macro:
        #  self._cached_result = self.parse_fdef(module_ast)
        # elif as_quote:
        #  self._cached_result = self.parse_fdef(module_ast)
        else:
            assert False, "parser mode configuration unsupported"
        self.pop()

    def result(self):
        return self._cached_result

    def push(self):
        self.locals = self.locals.new_child()
    def pop(self):
        self.locals = self.locals.parents

    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    # parser helper routines

    def err(self, node, errstr):
        raise ParseError(f"{self.getsrcinfo(node)}: {errstr}")

    def eval_expr(self, expr):
        assert isinstance(expr, pyast.expr)
        code = compile(pyast.Expression(expr), '', 'eval')
        e_obj = eval(code, self.globals, self.locals)
        return e_obj

    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    # structural parsing rules...

    def parse_fdef(self, fdef, instr=None):
        assert type(fdef) == pyast.FunctionDef

        fargs = fdef.args
        bad_arg_syntax_errmsg = """
    SYS_ATL expects function arguments to not use these python features:
      - position-only arguments
      - unnamed (position or keyword) arguments (i.e. *varargs, **kwargs)
      - keyword-only arguments
      - default argument values
    """
        if (len(fargs.posonlyargs) > 0 or fargs.vararg is not None or
            len(fargs.kwonlyargs) > 0 or len(fargs.kw_defaults) > 0 or
                fargs.kwarg is not None or len(fargs.defaults) > 0):
            self.err(fargs, bad_arg_syntax_errmsg)

        # process each argument in order
        # we will assume for now that all sizes come first
        #sizes = []
        args = []
        names = set()
        for a in fargs.args:
            if a.annotation is None:
                self.err(a, "expected argument to be typed, i.e. 'x : T'")
            tnode = a.annotation

            typ, mem   = self.parse_arg_type(tnode)
            if a.arg in names:
                self.err(a, f"repeated argument name: '{a.arg}'")
            names.add(a.arg)
            nm = Sym(a.arg)
            if type(typ) == UAST.Size:
                self.locals[a.arg] = SizeStub(nm)
            else:
                # note we don't need to stub the index variables
                self.locals[a.arg] = nm
            args.append(UAST.fnarg(nm, typ, mem, self.getsrcinfo(a)))

        # return types are non-sensical for SYS_ATL, b/c it models procedures
        if fdef.returns is not None:
            self.err(fdef, "SYS_ATL does not support function return types")

        # parse out any assertions at the front of the statement block
        # first split the assertions out
        first_non_assert = 0
        while (first_non_assert < len(fdef.body) and
               type(fdef.body[first_non_assert]) == pyast.Assert):
            first_non_assert += 1
        assertions = fdef.body[0:first_non_assert]
        pyast_body = fdef.body[first_non_assert:]

        # then parse out predicates from the assertions
        preds = []
        for a in assertions:
            if a.msg is not None:
                self.err(a, "SYS_ATL procedure assertions should "+
                            "not have messages")
            preds.append(self.parse_expr(a.test))

        # parse the procedure body
        body = self.parse_stmt_block(pyast_body)
        return UAST.proc(name=fdef.name,
                         args=args,
                         preds=preds,
                         body=body,
                         instr=instr,
                         srcinfo=self.getsrcinfo(fdef))

    def parse_arg_type(self, node):
        # Arg was of the form ` name : annotation `
        # The annotation will be of one of the following forms
        #   ` type `
        #   ` type @ memory `
        def is_at(x):
            return type(x) is pyast.BinOp and type(x.op) is pyast.MatMult

        # decompose top-level of annotation syntax
        if is_at(node):
            typ_node = node.left
            mem_node = node.right
        else:
            typ_node = node
            mem_node = None

        # detect which sort of type we have here
        is_size     = lambda x: type(x) is pyast.Name and x.id == "size"
        is_index    = lambda x: type(x) is pyast.Name and x.id == "index"

        # parse each kind of type here
        if is_size(typ_node):
            if mem_node is not None:
                self.err(node, "size types should not be annotated with "+
                               "memory locations")
            return UAST.Size(), None

        elif is_index(typ_node):
            if mem_node is not None:
                self.err(node, "size types should not be annotated with "+
                               "memory locations")
            return UAST.Index(), None

        else:
            typ = self.parse_num_type(typ_node)

            mem = self.eval_expr(mem_node) if mem_node else None

            return typ, mem


    def parse_alloc_typmem(self, node):
        if type(node) is pyast.BinOp and type(node.op) is pyast.MatMult:
            # node.right == Name
            # x[n] @ DRAM
            # x[n] @ lib.scratch
            mem     = self.eval_expr(node.right)
            node    = node.left
        else:
            mem     = None
        typ = self.parse_num_type(node)
        return typ, mem

    _prim_types = {
        'R'     : UAST.Num(),
        'f32'   : UAST.F32(),
        'f64'   : UAST.F64(),
        'i8'    : UAST.INT8(),
    }

    def parse_num_type(self, node):
        if type(node) is pyast.Subscript:
            if (type(node.value) is not pyast.Name
                    or node.value.id not in Parser._prim_types):
                self.err(node, "expected tensor type to be "+
                               "of the form 'R[...]', 'f32[...]', etc.")
            typ = Parser._prim_types[node.value.id]

            if sys.version_info[:3] >= (3, 9):
                # unpack single or multi-arg indexing to list of slices/indices
                if type(node.slice) is pyast.Slice:
                    self.err(node, "index-slicing not allowed")
                else:
                    if type(node.slice) is pyast.Tuple:
                        dims = node.slice.elts
                    else:
                        assert (type(node.slice) is pyast.Name or
                                type(node.slice) is pyast.Constant)
                        dims = [node.slice]
            else:
                # unpack single or multi-arg indexing to list of slices/indices
                if (type(node.slice) is pyast.Slice or
                    type(node.slice) is pyast.ExtSlice):
                    self.err(node, "index-slicing not allowed")
                else:
                    assert type(node.slice) is pyast.Index
                    if type(node.slice.value) is pyast.Tuple:
                        dims = node.slice.value.elts
                    else:
                        dims = [node.slice.value]

            # convert the dimension list into a full tensor type
            exprs = [self.parse_expr(idx) for idx in dims]
            #TODO: Should we set default stride here??
            typ = UAST.Tensor(exprs, [], typ)

            return typ

        elif type(node) is pyast.Name and node.id in Parser._prim_types:
            return Parser._prim_types[node.id]
        else:
            self.err(node, "unrecognized type: "+astor.dump_tree(node))

    def parse_stmt_block(self, stmts):
        assert type(stmts) is list

        rstmts = []

        for s in stmts:
            # ----- Assginment, Reduction, Var Declaration/Allocation parsing
            if (type(s) is pyast.Assign or type(s) is pyast.AnnAssign or
                    type(s) is pyast.AugAssign):
                # parse the rhs first, if it's present
                rhs = None
                if type(s) is pyast.AnnAssign:
                    if s.value is not None:
                        self.err(s, "Variable declaration should not "+
                                    "have value assigned")
                else:
                    rhs = self.parse_expr(s.value)

                # parse the lvalue expression
                if type(s) is pyast.Assign:
                    if len(s.targets) > 1:
                        self.err(s, "expected only one expression " +
                                    "on the left of an assignment")
                    name_node, idxs = self.parse_lvalue(s.targets[0])
                else:
                    name_node, idxs = self.parse_lvalue(s.target)
                if type(s) is pyast.AnnAssign and len(idxs) > 0:
                    self.err(tgt, "expected simple name in declaration")

                # insert any needed Allocs
                if type(s) is pyast.AnnAssign:
                    nm = Sym(name_node.id)
                    self.locals[name_node.id] = nm
                    typ, mem = self.parse_alloc_typmem(s.annotation)
                    rstmts.append(UAST.Alloc(nm, typ, mem, self.getsrcinfo(s)))
                elif type(s) is pyast.Assign and len(idxs) == 0:
                    if name_node.id not in self.locals:
                        # TODO: Fix here
                        nm = Sym(name_node.id)
                        self.locals[name_node.id] = nm
                        rstmts.append(UAST.Alloc(nm, UAST.Num(), None,
                                                 self.getsrcinfo(s)))

                # get the symbol corresponding to the name on the
                # left-hand-side
                if type(s) is pyast.Assign or type(s) is pyast.AugAssign:
                    if name_node.id not in self.locals:
                        self.err(
                            name_node, f"variable '{name_node.id}' undefined")
                    nm = self.locals[name_node.id]
                    if type(nm) is SizeStub:
                        self.err(name_node, f"cannot write to " +
                                            f"size variable '{name_node.id}'")
                    elif type(nm) is not Sym:
                        self.err(name_node, f"expected '{name_node.id}' to "+
                                            f"refer to a local variable")

                # generate the assignemnt or reduction statement
                if type(s) is pyast.Assign:
                    rstmts.append(UAST.Assign(
                        nm, idxs, rhs, self.getsrcinfo(s)))
                elif type(s) is pyast.AugAssign:
                    if type(s.op) is not pyast.Add:
                        self.err(s, "only += reductions currently supported")
                    rstmts.append(UAST.Reduce(
                        nm, idxs, rhs, self.getsrcinfo(s)))

            # ----- For Loop parsing
            elif type(s) is pyast.For:
                if len(s.orelse) > 0:
                    self.err(s, "else clause on for-loops unsupported")

                self.push()
                if type(s.target) is not pyast.Name:
                    self.err(
                        s.target, "expected simple name for iterator variable")
                itr = Sym(s.target.id)
                self.locals[s.target.id] = itr
                hi   = self.parse_loop_cond(s.iter)
                body = self.parse_stmt_block(s.body)
                self.pop()

                rstmts.append(UAST.ForAll(itr, hi, body, self.getsrcinfo(s)))

            # ----- If statement parsing
            elif type(s) is pyast.If:
                cond = self.parse_expr(s.test)

                self.push()
                body = self.parse_stmt_block(s.body)
                self.pop()
                self.push()
                orelse = self.parse_stmt_block(s.orelse)
                self.pop()

                rstmts.append(UAST.If(cond, body, orelse, self.getsrcinfo(s)))

            # ----- Sub-routine call parsing
            elif (type(s) is pyast.Expr and
                  type(s.value) is pyast.Call and
                  type(s.value.func) is pyast.Name):
                fname = s.value.func.id
                if fname in self.locals:
                    f = self.locals[fname]
                elif fname in self.globals:
                    f = self.globals[fname]
                else:
                    self.err(s.value.func, f"procedure '{fname}' "+
                                            "was undefined")
                if not isinstance(f, Procedure):
                    self.err(s.value.func, f"expected '{fname}' "+
                                            "to be a procedure")

                if len(s.value.keywords) > 0:
                    self.err(s.value, "cannot call procedure() "+
                                      "with keyword arguments")

                args = [ self.parse_expr(a) for a in s.value.args ]

                rstmts.append(UAST.Call(f.INTERNAL_proc(),
                              args, self.getsrcinfo(s.value)))

            # ----- Pass no-op parsing
            elif type(s) is pyast.Pass:
                rstmts.append(UAST.Pass(self.getsrcinfo(s)))

            elif type(s) is pyast.Assert:
                self.err(s, "predicate assert should happen at the beginning "+
                            "of a function")
            else:
                self.err(s, "unsupported type of statement")

        return rstmts

    def parse_loop_cond(self, cond):
        if type(cond) is pyast.Call:
            if type(cond.func) is pyast.Name and cond.func.id == "par":
                if len(cond.keywords) > 0:
                    self.err(cond, "par() does not support named arguments")
                elif len(cond.args) != 2:
                    self.err(cond, "par() expects exactly 2 arguments")
                lo = self.parse_expr(cond.args[0])
                hi = self.parse_expr(cond.args[1])
                return UAST.ParRange(lo, hi, self.getsrcinfo(cond))
            else:
                self.err(cond, "expected 'par(..., ...)' or a predicate")
        else:
            return self.parse_expr(cond)

    # parse the left-hand-side of an assignment
    def parse_lvalue(self, node):
        if type(node) is not pyast.Name and type(node) is not pyast.Subscript:
            self.err(tgt, "expected lhs of form 'x' or 'x[...]'")
        else:
            return self.parse_array_indexing(node)

    def parse_array_indexing(self, node):
        if type(node) is pyast.Name:
            return node, []
        elif type(node) is pyast.Subscript:
            if sys.version_info[:3] >= (3, 9):
                # unpack single or multi-arg indexing to list of slices/indices
                if type(node.slice) is pyast.Slice:
                    self.err(node, "index-slicing not allowed")
                else:
                    if type(node.slice) is pyast.Tuple:
                        dims = node.slice.elts
                    else:
                        assert (type(node.slice) is pyast.Name or
                                type(node.slice) is pyast.Constant or
                                type(node.slice) is pyast.BinOp)
                        dims = [node.slice]
            else:
                # unpack single or multi-arg indexing to list of slices/indices
                if (type(node.slice) is pyast.Slice or
                    type(node.slice) is pyast.ExtSlice):
                    self.err(node, "index-slicing not allowed")
                else:
                    assert type(node.slice) is pyast.Index
                    if type(node.slice.value) is pyast.Tuple:
                        dims = node.slice.value.elts
                    else:
                        dims = [node.slice.value]

            if type(node.value) is not pyast.Name:
                self.err(node, "expected access to have form 'x' or 'x[...]'")

            idxs    = [ self.parse_expr(e) for e in dims ]

            return node.value, idxs

    # parse expressions, including values, indices, and booleans
    def parse_expr(self, e):
        if type(e) is pyast.Name or type(e) is pyast.Subscript:
            nm_node, idxs = self.parse_array_indexing(e)

            # get the buffer name
            if nm_node.id not in self.locals:
                self.err(nm_node, f"variable '{nm_node.id}' undefined")
            nm = self.locals[nm_node.id]
            if type(nm) is SizeStub:
                nm = nm.nm
            elif type(nm) is not Sym:
                self.err(nm_node, f"expected '{nm_node.id}' to refer to " +
                                  f"a local variable")

            return UAST.Read(nm, idxs, self.getsrcinfo(e))

        elif type(e) is pyast.Constant:
            return UAST.Const(e.value, self.getsrcinfo(e))

        elif type(e) is pyast.UnaryOp:
            if type(e.op) is pyast.USub:
                arg = self.parse_expr(e.operand)
                return UAST.USub(arg, self.getsrcinfo(e))
            else:
                opnm = ("+" if type(e.op) is pyast.UAdd else
                        "not" if type(e.op) is pyast.Not else
                        "~" if type(e.op) is pyast.Invert else
                        "ERROR-BAD-OP-CASE")
                self.err(e, f"unsupported unary operator: {opnm}")

        elif type(e) is pyast.BinOp:
            lhs = self.parse_expr(e.left)
            rhs = self.parse_expr(e.right)
            if type(e.op) is pyast.Add:
                op = "+"
            elif type(e.op) is pyast.Sub:
                op = "-"
            elif type(e.op) is pyast.Mult:
                op = "*"
            elif type(e.op) is pyast.Div:
                op = "/"
            elif type(e.op) is pyast.FloorDiv:
                op = "//"
            elif type(e.op) is pyast.Mod:
                op = "%"
            elif type(e.op) is pyast.Pow:
                op = "**"
            elif type(e.op) is pyast.LShift:
                op = "<<"
            elif type(e.op) is pyast.RShift:
                op = ">>"
            elif type(e.op) is pyast.BitOr:
                op = "|"
            elif type(e.op) is pyast.BitXor:
                op = "^"
            elif type(e.op) is pyast.BitAnd:
                op = "&"
            elif type(e.op) is pyast.MatMult:
                op = "@"
            else:
                assert False, "unrecognized op"
            if op not in front_ops:
                self.err(e, f"unsupported binary operator: {op}")

            return UAST.BinOp(op, lhs, rhs, self.getsrcinfo(e))

        elif type(e) is pyast.BoolOp:
            assert len(e.values) > 1
            lhs = self.parse_expr(e.values[0])

            if type(e.op) is pyast.And:
                op = "and"
            elif type(e.op) is pyast.Or:
                op = "or"
            else:
                assert False, "unrecognized op"

            for rhs in e.values[1:]:
                lhs = UAST.BinOp(op, lhs, self.parse_expr(
                    rhs), self.getsrcinfo(e))

            return lhs

        elif type(e) is pyast.Compare:
            assert len(e.ops) == len(e.comparators)
            vals = ([self.parse_expr(e.left)] +
                    [self.parse_expr(v) for v in e.comparators])
            srcinfo = self.getsrcinfo(e)

            res = None
            for opnode, lhs, rhs in zip(e.ops, vals[:-1], vals[1:]):
                if type(opnode) is pyast.Eq:
                    op = "=="
                elif type(opnode) is pyast.NotEq:
                    op = "!="
                elif type(opnode) is pyast.Lt:
                    op = "<"
                elif type(opnode) is pyast.LtE:
                    op = "<="
                elif type(opnode) is pyast.Gt:
                    op = ">"
                elif type(opnode) is pyast.GtE:
                    op = ">="
                elif type(opnode) is pyast.Is:
                    op = "is"
                elif type(opnode) is pyast.IsNot:
                    op = "is not"
                elif type(opnode) is pyast.In:
                    op = "in"
                elif type(opnode) is pyast.NotIn:
                    op = "not in"
                else:
                    assert False, "unrecognized op"
                if op not in front_ops:
                    self.err(e, f"unsupported binary operator: {op}")
                c = UAST.BinOp(op, lhs, rhs, self.getsrcinfo(e))
                res = c if res is None else UAST.BinOp("and", res, c, srcinfo)

            return res

        else:
            self.err(e, "unsupported form of expression")



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Parser Pass for Patterns object;

# TODO: should we try to eliminate redundancy between this and the other
#       parser, in order to ensure more consistency?


class PatternParser:
    def __init__(self, module_stmts, getsrcinfo):

        self.module_stmts   = module_stmts
        self.getsrcinfo     = getsrcinfo

        is_expr = False
        if len(module_stmts) == 1:
            s = module_stmts[0]
            if type(s) is pyast.Expr and type(s.value) is not pyast.Call:
                is_expr = True

        if is_expr:
            self._past_result   = self.parse_expr(s.value)
        else:
            self._past_result   = self.parse_stmts(module_stmts)

    def result(self):
        return self._past_result

    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    # parser helper routines

    def err(self, node, errstr):
        raise ParseError(f"{self.getsrcinfo(node)}: {errstr}")

    def push(self):
        pass

    def pop(self):
        pass

    #def eval_expr(self, expr):
    #    assert isinstance(expr, pyast.expr)
    #    code = compile(pyast.Expression(expr), '', 'eval')
    #    e_obj = eval(code)
    #    return e_obj

    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    # structural parsing rules...

    def parse_stmts(self, stmts):
        assert type(stmts) is list

        rstmts = []

        for s in stmts:
            # ----- Assginment, Reduction, Var Declaration/Allocation parsing
            if (type(s) is pyast.Assign or type(s) is pyast.AnnAssign or
                type(s) is pyast.AugAssign):
                # parse the rhs first, if it's present
                rhs = None
                if type(s) is pyast.AnnAssign:
                    if s.value is not None:
                        self.err(s, "Variable declaration should not "+
                                    "have value assigned")
                    name_node, idxs = self.parse_lvalue(s.target)
                    if len(idxs) > 0:
                        self.err(tgt, "expected simple name in declaration")
                    nm = name_node.id
                    if nm != '_' and not is_valid_name(nm):
                        self.err(name_node, "expected valid name or _")
                    rstmts.append(PAST.Alloc(nm, self.getsrcinfo(s)))
                    continue # escape rest of case
                else:
                    rhs = self.parse_expr(s.value)

                # parse the lvalue expression
                if type(s) is pyast.Assign:
                    if len(s.targets) > 1:
                        self.err(s, "expected only one expression " +
                                    "on the left of an assignment")
                    name_node, idxs = self.parse_lvalue(s.targets[0])
                else:
                    name_node, idxs = self.parse_lvalue(s.target)

                # check that the name is valid
                nm = name_node.id
                if nm != '_' and not is_valid_name(nm):
                    self.err(name_node, "expected valid name or _")

                # generate the assignemnt or reduction statement
                if type(s) is pyast.Assign:
                    rstmts.append(PAST.Assign(
                        nm, idxs, rhs, self.getsrcinfo(s)))
                elif type(s) is pyast.AugAssign:
                    if type(s.op) is not pyast.Add:
                        self.err(s, "only += reductions currently supported")
                    rstmts.append(PAST.Reduce(
                        nm, idxs, rhs, self.getsrcinfo(s)))

            # ----- For Loop parsing
            elif type(s) is pyast.For:
                if len(s.orelse) > 0:
                    self.err(s, "else clause on for-loops unsupported")

                self.push()
                if type(s.target) is not pyast.Name:
                    self.err(
                        s.target, "expected simple name for iterator variable")
                itr = s.target.id
                cond = self.parse_loop_cond(s.iter)
                body = self.parse_stmts(s.body)
                self.pop()

                rstmts.append(PAST.ForAll(itr, cond, body, self.getsrcinfo(s)))

            # ----- If statement parsing
            elif type(s) is pyast.If:
                cond = self.parse_expr(s.test)

                self.push()
                body = self.parse_stmts(s.body)
                self.pop()
                self.push()
                orelse = self.parse_stmts(s.orelse)
                self.pop()

                rstmts.append(PAST.If(cond, body, orelse, self.getsrcinfo(s)))

            # ----- Sub-routine call parsing
            elif (type(s) is pyast.Expr and
                  type(s.value) is pyast.Call and
                  type(s.value.func) is pyast.Name):
                fname = s.value.func.id

                if len(s.value.keywords) > 0:
                    self.err(s.value, "cannot call procedure() "+
                                      "with keyword arguments")

                args = [ self.parse_expr(a) for a in s.value.args ]

                rstmts.append(PAST.Call(fname, args,
                                        self.getsrcinfo(s.value)))

            # ----- Stmt Hole parsing
            elif (type(s) is pyast.Expr and
                  type(s.value) is pyast.Name and
                  s.value.id == '_'):
                rstmts.append(PAST.S_Hole(self.getsrcinfo(s.value)))

            # ----- Pass no-op parsing
            elif type(s) is pyast.Pass:
                rstmts.append(PAST.Pass(self.getsrcinfo(s)))
            else:
                self.err(s, "unsupported type of statement")

        return rstmts

    def parse_lvalue(self, node):
        if type(node) is not pyast.Name and type(node) is not pyast.Subscript:
            self.err(tgt, "expected lhs of form 'x' or 'x[...]'")
        else:
            return self.parse_array_indexing(node)

    def parse_array_indexing(self, node):
        if type(node) is pyast.Name:
            return node, []
        elif type(node) is pyast.Subscript:
            if sys.version_info[:3] >= (3, 9):
                # unpack single or multi-arg indexing to list of slices/indices
                if type(node.slice) is pyast.Slice:
                    self.err(node, "index-slicing not allowed")
                else:
                    if type(node.slice) is pyast.Tuple:
                        dims = node.slice.elts
                    else:
                        assert (type(node.slice) is pyast.Name or
                                type(node.slice) is pyast.BinOp)
                        dims = [node.slice]
            else:
                # unpack single or multi-arg indexing to list of slices/indices
                if (type(node.slice) is pyast.Slice or
                    type(node.slice) is pyast.ExtSlice):
                    self.err(node, "index-slicing not allowed")
                else:
                    assert type(node.slice) is pyast.Index
                    if type(node.slice.value) is pyast.Tuple:
                        dims = node.slice.value.elts
                    else:
                        dims = [node.slice.value]

            if type(node.value) is not pyast.Name:
                self.err(node, "expected access to have form 'x' or 'x[...]'")

            idxs    = [ self.parse_expr(e) for e in dims ]

            return node.value, idxs

    def parse_loop_cond(self, cond):
        if type(cond) is pyast.Name and cond.id == '_':
            return self.parse_expr(cond)
        if type(cond) is pyast.Call:
            if type(cond.func) is pyast.Name and cond.func.id == "par":
                if len(cond.keywords) > 0:
                    self.err(cond, "par() does not support named arguments")
                elif len(cond.args) != 2:
                    self.err(cond, "par() expects exactly 2 arguments")
                lo = self.parse_expr(cond.args[0])
                hi = self.parse_expr(cond.args[1])
                if type(lo) is not UAST.Const or lo.val != 0:
                    self.err(cond ,"expected par(0, ...)")
                return hi
        # fall-through error
        self.err(cond, "expected 'par(..., ...)'")

    def parse_expr(self, e):
        if type(e) is pyast.Name or type(e) is pyast.Subscript:
            nm_node, idxs   = self.parse_array_indexing(e)
            nm              = nm_node.id
            if len(idxs) == 0 and nm == '_':
                return PAST.E_Hole( self.getsrcinfo(e) )
            else:
                return PAST.Read(nm, idxs, self.getsrcinfo(e))

        elif type(e) is pyast.Constant:
            return PAST.Const(e.value, self.getsrcinfo(e))

        elif type(e) is pyast.UnaryOp:
            if type(e.op) is pyast.USub:
                arg = self.parse_expr(e.operand)
                return PAST.USub(arg, self.getsrcinfo(e))
            else:
                opnm = ("+" if type(e.op) is pyast.UAdd else
                        "not" if type(e.op) is pyast.Not else
                        "~" if type(e.op) is pyast.Invert else
                        "ERROR-BAD-OP-CASE")
                self.err(e, f"unsupported unary operator: {opnm}")

        elif type(e) is pyast.BinOp:
            lhs = self.parse_expr(e.left)
            rhs = self.parse_expr(e.right)
            if type(e.op) is pyast.Add:
                op = "+"
            elif type(e.op) is pyast.Sub:
                op = "-"
            elif type(e.op) is pyast.Mult:
                op = "*"
            elif type(e.op) is pyast.Div:
                op = "/"
            elif type(e.op) is pyast.FloorDiv:
                op = "//"
            elif type(e.op) is pyast.Mod:
                op = "%"
            elif type(e.op) is pyast.Pow:
                op = "**"
            elif type(e.op) is pyast.LShift:
                op = "<<"
            elif type(e.op) is pyast.RShift:
                op = ">>"
            elif type(e.op) is pyast.BitOr:
                op = "|"
            elif type(e.op) is pyast.BitXor:
                op = "^"
            elif type(e.op) is pyast.BitAnd:
                op = "&"
            elif type(e.op) is pyast.MatMult:
                op = "@"
            else:
                assert False, "unrecognized op"
            if op not in front_ops:
                self.err(e, f"unsupported binary operator: {op}")

            return PAST.BinOp(op, lhs, rhs, self.getsrcinfo(e))

        elif type(e) is pyast.BoolOp:
            assert len(e.values) > 1
            lhs = self.parse_expr(e.values[0])

            if type(e.op) is pyast.And:
                op = "and"
            elif type(e.op) is pyast.Or:
                op = "or"
            else:
                assert False, "unrecognized op"

            for rhs in e.values[1:]:
                lhs = PAST.BinOp(op, lhs, self.parse_expr(
                    rhs), self.getsrcinfo(e))

            return lhs

        elif type(e) is pyast.Compare:
            assert len(e.ops) == len(e.comparators)
            vals = ([self.parse_expr(e.left)] +
                    [self.parse_expr(v) for v in e.comparators])
            srcinfo = self.getsrcinfo(e)

            res = None
            for opnode, lhs, rhs in zip(e.ops, vals[:-1], vals[1:]):
                if type(opnode) is pyast.Eq:
                    op = "=="
                elif type(opnode) is pyast.NotEq:
                    op = "!="
                elif type(opnode) is pyast.Lt:
                    op = "<"
                elif type(opnode) is pyast.LtE:
                    op = "<="
                elif type(opnode) is pyast.Gt:
                    op = ">"
                elif type(opnode) is pyast.GtE:
                    op = ">="
                elif type(opnode) is pyast.Is:
                    op = "is"
                elif type(opnode) is pyast.IsNot:
                    op = "is not"
                elif type(opnode) is pyast.In:
                    op = "in"
                elif type(opnode) is pyast.NotIn:
                    op = "not in"
                else:
                    assert False, "unrecognized op"
                if op not in front_ops:
                    self.err(e, f"unsupported binary operator: {op}")
                c = PAST.BinOp(op, lhs, rhs, self.getsrcinfo(e))
                res = c if res is None else PAST.BinOp("and", res, c, srcinfo)

            return res

        else:
            self.err(e, "unsupported form of expression")
