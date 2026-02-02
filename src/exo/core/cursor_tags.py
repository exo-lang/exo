"""Opaque object used to ID cursors created as a side-effect of scheduling rewrites.

You can create your own instead of using defaults.
Each CursorTag is a unique ID.
We rely on Python object identity, not string names.

"""


class CursorTag:
    __slots__ = []

    def __repr__(self):
        return f"CursorTag()"

    def as_default(self, _or):
        if _or is None:
            return self
        else:
            assert isinstance(_or, CursorTag)
            return _or


class EmptyDict:
    def find(self, key):
        return None


empty_dict = EmptyDict()


# Default tag for allocation stmt added as part of stage_mem.
CursorTag.StageAlloc = CursorTag()

# Default tag for loop nest initializing the new buffer added by stage_mem.
CursorTag.StageLoad = CursorTag()

# Default tag for loop nest writing out the buffer added by stage_mem.
CursorTag.StageStore = CursorTag()

# Default tag for call statement added
CursorTag.Call = CursorTag()

# Default tag for pass inserted
CursorTag.Pass = CursorTag()

# Default tag for Fence inserted
CursorTag.Fence = CursorTag()

# Default tag for Arrive inserted
CursorTag.Arrive = CursorTag()

# Default tag for Await inserted
CursorTag.Await = CursorTag()

# Default tag for barrier allocation inserted
CursorTag.BarrierAlloc = CursorTag()

# Default tag for loop nest created by fission
# containing body statements before the input gap cursor.
CursorTag.FissonBefore = CursorTag()

# Default tag for loop nest created by fission
# containing body statements after the input gap cursor.
CursorTag.FissonAfter = CursorTag()

# Default tag for the loop created by cut_loop
# for iterations before the cut point.
CursorTag.CutBefore = CursorTag()

# Default tag for the loop created by cut_loop
# for iterations including and after the cut point.
CursorTag.CutAfter = CursorTag()
