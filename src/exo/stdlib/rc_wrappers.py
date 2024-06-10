from __future__ import annotations
from dataclasses import dataclass

from exo import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from .inspection import *


# TODO: delete once this feature is supported by the primitives


@dataclass
class fission_cursors:
    scope1: ForCursor | IfCursor
    scope2: ForCursor | IfCursor

    def __iter__(self):
        yield self.scope1
        yield self.scope2


def fission_(proc, gap, rc=False):
    gap = proc.forward(gap)
    stmt = gap.anchor()
    is_after = stmt.after() is gap
    proc = fission(proc, gap)

    scope1 = proc.forward(stmt).parent()
    scope2 = scope1.next()

    if is_after:
        scope1, scope2 = scope2, scope1

    if rc:
        return proc, fission_cursors(scope1, scope2)
    return proc


@dataclass
class divide_loop_cursors:
    outer_loop: ForCursor
    inner_loop: ForCursor
    tail_loop: ForCursor

    def __iter__(self):
        yield self.outer_loop
        yield self.inner_loop
        yield self.tail_loop


def divide_loop_(proc, loop_cursor, div_const, tail="guard", rc=False):
    loop_cursor = proc.forward(loop_cursor)
    loop_iter = loop_cursor.name()
    perfect = tail == "perfect"
    if tail == "perfect":
        tail = "cut"
        perfect = True
    proc = divide_loop(
        proc,
        loop_cursor,
        div_const,
        (loop_iter + "o", loop_iter + "i"),
        tail=tail,
        perfect=perfect,
    )
    if not rc:
        return proc

    outer_loop = proc.forward(loop_cursor)
    inner_loop = outer_loop.body()[0]

    if perfect == True or tail == "guard":
        tail_loop = InvalidCursor()
    else:
        tail_loop = outer_loop.next()

    outer_loop = proc.forward(outer_loop)
    inner_loop = proc.forward(inner_loop)
    if not isinstance(tail_loop, InvalidCursor):
        tail_loop = proc.forward(tail_loop)
    return proc, divide_loop_cursors(outer_loop, inner_loop, tail_loop)


@dataclass
class stage_mem_cursors:
    alloc: AllocCursor
    load_stage: Cursor
    block: BlockCursor
    store_stage: Cursor

    def __iter__(self):
        yield self.alloc
        yield self.load_stage
        yield self.block
        yield self.store_stage


def stage_mem_(proc, block, buff, new_buff_name, accum=False, rc=False):
    if not isinstance(block, BlockCursor):
        block = proc.forward(block)
        block = block.as_block()

    block_first = block[0]
    block_last = block[-1]
    proc = stage_mem(proc, block, buff, new_buff_name, accum)
    block_first = proc.forward(block_first)
    block_last = proc.forward(block_last)
    alloc = block_first.prev().prev()
    load_stage = block_first.prev()
    block = block_first.as_block().expand(0, len(block) - 1)
    store_stage = block_last.next()
    if not rc:
        return proc
    return proc, stage_mem_cursors(alloc, load_stage, block, store_stage)


@dataclass
class cut_loop_cursors:
    loop1: ForCursor
    loop2: ForCursor

    def __iter__(self):
        yield self.loop1
        yield self.loop2


def cut_loop_(proc, loop, expr, rc=False):
    proc = cut_loop(proc, loop, expr)

    if not rc:
        return proc

    loop1 = proc.forward(loop)
    loop2 = loop1.next()
    return proc, cut_loop_cursors(loop1, loop2)


@dataclass
class specialize_cursors:
    if_stmt: Cursor

    def __iter__(self):
        yield self.if_stmt


def specialize_(proc, stmt, cond, rc=False):
    stmt = proc.forward(stmt)
    parent = stmt.parent()
    index = get_index_in_body(proc, stmt)
    proc = specialize(proc, stmt, cond)
    if not rc:
        return proc
    is_else = False
    if (
        isinstance(parent, IfCursor)
        and not isinstance(parent.orelse(), InvalidCursor)
        and index < len(parent.orelse())
        and parent.orelse()[index] == stmt
    ):
        is_else = True
    if not isinstance(parent, InvalidCursor):
        parent = proc.forward(parent)
    else:
        parent = proc

    if_stmt = parent.body()[index] if not is_else else parent.orelse()[index]
    return proc, specialize_cursors(if_stmt)
