from dataclasses import dataclass

from ..spork.coll_algebra import CollUnit


@dataclass(slots=True, frozen=True)
class SizeAnnotation:
    def decay(self):
        return None


@dataclass(slots=True, frozen=True)
class NoSizeAnnotation(SizeAnnotation):
    # This should not exist but has to because
    # asdl_adt.validators.ValidationError is brain dead
    # Only used for UAST.

    def __bool__(self):
        return False


@dataclass(slots=True, frozen=True)
class ring_buffer_by(SizeAnnotation):
    depth: int

    def __post_init__(self):
        depth = self.depth
        assert isinstance(depth, int)
        assert depth > 0

    def decay(self):
        return self


# For potential future use...
@dataclass(slots=True, frozen=True)
class CollUnitAnnotation(SizeAnnotation):
    unit: CollUnit

    def __post_init__(self):
        unit = self.unit
        assert isinstance(unit, CollUnit)

    def decay(self):
        return self.unit


def to_size_annotation(obj):
    if obj is None:
        return NoSizeAnnotation()
    if isinstance(obj, ring_buffer_by):
        return obj
    if isinstance(obj, CollUnit):
        return CollUnitAnnotation(obj)
    raise TypeError(f"Unknown size_annotation {obj} of type {type(obj)}")
