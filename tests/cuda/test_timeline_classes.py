"""Sanity checks on qual-tl implementation classes and proxy classes

This is mostly claude-generated test cases for some internals that could
reasonably change, although test_mbarrier_qual_not_in_data_ram and
test_needs_proxy_fence test some important points not thoroughly
covered by Exo test_cuda_sync tests.

"""

from __future__ import annotations

import pytest

from exo.spork import timelines
from exo.spork.timelines import (
    InstrQuals,
    Qual_tl,
    Sync_tl,
    needs_proxy_fence,
)


def test_no_sync_tl_contains_atomic():
    atomic_bits = Qual_tl.get_all_atomic_bits()
    assert atomic_bits != 0
    for v in vars(timelines).values():
        if isinstance(v, Sync_tl):
            assert not (v.get_timeline_set_bits() & atomic_bits), v


def test_sync_tl_rejects_atomic():
    with pytest.raises(AssertionError):
        Sync_tl("bad", [timelines.cuda_generic_atomic_qual])


def test_precondition_excludes_out_of_order():
    ooo_bits = Qual_tl.get_all_out_of_order_bits()
    assert ooo_bits != 0
    for v in vars(timelines).values():
        if isinstance(v, dict):
            for q in v.values():
                if isinstance(q, InstrQuals):
                    assert not (q.precondition_bits & ooo_bits), q
    with pytest.raises(AssertionError):
        # Out-of-order initial qual-tl can't be the default precondition
        InstrQuals(timelines.tma_to_smem_async_qual)
    with pytest.raises(AssertionError):
        InstrQuals(
            timelines.cuda_in_order_ram_qual,
            precondition=timelines.Sm80_cp_async_qual,
        )


def test_instr_quals_rejects_atomic():
    with pytest.raises(AssertionError):
        InstrQuals(timelines.cuda_generic_atomic_qual)
    with pytest.raises(AssertionError):
        InstrQuals(
            timelines.cuda_in_order_ram_qual,
            precondition=[
                timelines.cuda_in_order_ram_qual,
                timelines.cuda_generic_atomic_qual,
            ],
        )


def test_mbarrier_qual_not_in_data_ram():
    # cuda_mbarrier_qual has no proxy class; it must not be in data RAM masks,
    # otherwise sync-check could carry generic -> async ordering transitively
    # without a codegen'd proxy fence.
    assert timelines.cuda_mbarrier_qual.proxy_class() == "neither"
    mbar_bit = timelines.cuda_mbarrier_qual.as_bit()
    for q in timelines.cuda_ram_qual_tl_dict.values():
        assert not (Qual_tl.make_bits(q) & mbar_bit), q


@pytest.mark.parametrize(
    "L1, L2, expected",
    [
        # TMA load -> wgmma / ld.shared (implicit completion fence)
        ("cuda_mbarrier_only", "cuda_generic_and_async_proxy", False),
        # wgmma read / TMA write -> TMA overwrite
        ("cuda_async_proxy_retired", "cuda_async_proxy_retired", False),
        # ld.shared -> TMA overwrite
        ("cuda_in_order", "cuda_async_proxy_retired", True),
        # generic write -> TMA store / wgmma
        ("cuda_in_order", "cuda_generic_and_async_proxy", True),
        # cp.async (Sm80) -> wgmma
        ("Sm80_cp_async", "cuda_generic_and_async_proxy", True),
        # generic -> generic
        ("Sm80_generic", "cuda_in_order", False),
        # commit groups
        ("wgmma_async", "cuda_generic_and_async_proxy", False),
        ("tma_to_gmem_async", "cuda_generic_and_async_proxy", False),
    ],
)
def test_needs_proxy_fence(L1, L2, expected):
    L1 = getattr(timelines, L1)
    L2 = getattr(timelines, L2)
    assert needs_proxy_fence(L1, L2) == expected
