from __future__ import annotations

import textwrap

import pytest

from exo import SchedulingError
from exo import proc
from exo.syntax import *
from exo.stdlib.scheduling import *

# skipping because the API has changed to invalidate this particular
# error message
@pytest.mark.skip()
def test_bad_reorder():
    @proc
    def example(N: size, A: f32[N]):
        for i in seq(0, N):
            A[i] = 0.0

    expected_error = textwrap.dedent(
        """
    reorder: failed to find statement
    Pattern: for i in _:
               for j in _: _
    """
    ).strip()

    with pytest.raises(SchedulingError, match=expected_error):
        reorder_loops(example, "i", "j")
