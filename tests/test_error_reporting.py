from __future__ import annotations

import textwrap

import pytest

from SYS_ATL import SchedulingError
from SYS_ATL import proc
from SYS_ATL.syntax import *


def test_bad_reorder():
    @proc
    def example(N: size, A: f32[N]):
        for i in par(0, N):
            A[i] = 0.0

    expected_error = textwrap.dedent('''
    reorder: failed to find statement
    Pattern: for i in _:
               for j in _: _
    ''').strip()

    with pytest.raises(SchedulingError, match=expected_error):
        example.reorder('i', 'j')
