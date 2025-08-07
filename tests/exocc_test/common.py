from __future__ import annotations

from exo import proc
from exo.API_scheduling import rename


def make_proc(name):
    @proc
    def the_proc():
        pass

    return rename(the_proc, name)
