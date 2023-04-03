from __future__ import annotations

from exo import proc, DRAM, QAST
from exo.stdlib.scheduling import *


# ------- Reflection tests ---------


def new_sgemm():
    @proc
    def sgemm_full(
        N: size,
        M: size,
        K: size,
        C: f32[N, M] @ DRAM,
        A: f32[N, K] @ DRAM,
        B: f32[K, M] @ DRAM,
    ):
        for i in seq(0, N):
            for j in seq(0, M):
                for k in seq(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    return sgemm_full


def test_proc_name():
    sgemm = new_sgemm()

    proc = sgemm.get_ast()
    assert isinstance(proc, QAST.Proc)
    assert proc.name == "sgemm_full"

    sgemm = rename(sgemm, "sgemm")

    proc = sgemm.get_ast()
    assert isinstance(proc, QAST.Proc)
    assert proc.name == "sgemm"


def test_find_outer_loop():
    sgemm = new_sgemm()

    loops = sgemm.get_ast("for _ in _: _ #0")
    assert isinstance(loops, list) and len(loops) == 1
    assert isinstance(loops[0], QAST.For)

    assert loops[0].name == "i"
    i_body = loops[0].body
    assert isinstance(i_body, list) and len(i_body) == 1
    assert isinstance(i_body[0], QAST.For)

    assert i_body[0].name == "j"
    j_body = i_body[0].body
    assert isinstance(j_body, list) and len(j_body) == 1
    assert isinstance(j_body[0], QAST.For)

    assert j_body[0].name == "k"
    k_body = j_body[0].body
    assert isinstance(k_body, list) and len(k_body) == 1

    assert not isinstance(k_body[0], QAST.For)


def get_loop_nest_info(p, pattern):
    loops = p.get_ast(pattern)
    if loops is None:
        return []
    assert isinstance(loops, list) and len(loops) > 0
    assert isinstance(loops[0], QAST.Stmt), "must call with ... #_ pattern"

    def recurse_loops(loops):
        if len(loops) != 1:
            return []
        stmt = loops[0]
        assert isinstance(stmt, QAST.Stmt)

        if isinstance(stmt, QAST.For):
            return [(stmt.name, stmt.hi)] + recurse_loops(stmt.body)
        else:
            return []

    return recurse_loops(loops)


def test_get_outer_loop_info():
    sgemm = new_sgemm()

    info = get_loop_nest_info(sgemm, "for _ in _: _ #0")

    expect_info = [
        ("i", QAST.Read("N", [], QAST.size())),
        ("j", QAST.Read("M", [], QAST.size())),
        ("k", QAST.Read("K", [], QAST.size())),
    ]
    assert info == expect_info


def test_get_mid_loop_info():
    sgemm = new_sgemm()

    expect_info = [
        ("j", QAST.Read("M", [], QAST.size())),
        ("k", QAST.Read("K", [], QAST.size())),
    ]

    info = get_loop_nest_info(sgemm, "for j in _: _ #0")
    assert info == expect_info

    info = get_loop_nest_info(sgemm, "for _ in _: _ #1")
    assert info == expect_info


def test_get_bottom_loop_info():
    sgemm = new_sgemm()

    expect_info = [
        ("k", QAST.Read("K", [], QAST.size())),
    ]

    info = get_loop_nest_info(sgemm, "for k in _: _ #0")
    assert info == expect_info

    info = get_loop_nest_info(sgemm, "for _ in _: _ #2")
    assert info == expect_info


def test_get_no_loop_info():
    sgemm = new_sgemm()

    info = get_loop_nest_info(sgemm, "for abc in _: _ #0")
    assert info == []


def test_show_effect(golden):
    sgemm = new_sgemm()

    assert sgemm.show_effect("for j in _: _") == golden
