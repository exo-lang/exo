from __future__ import annotations

import textwrap

import pytest

from SYS_ATL import proc
from SYS_ATL.LoopIR_scheduling import SchedulingError
from SYS_ATL.syntax import *


def test_bad_reorder():
    @proc
    def example(N: size, A: f32[N]):
        for i in par(0, N):
            A[i] = 0.0

    with pytest.raises(SchedulingError) as e:
        example.reorder('i', 'j')

    expected_error = textwrap.dedent('''
reorder: failed to find statement
Pattern: for i in _:
           for j in _: _
''').strip()

    assert str(e.value) == expected_error
