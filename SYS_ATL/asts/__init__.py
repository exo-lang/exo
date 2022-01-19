from ..prelude import is_valid_name

OP_STRINGS = frozenset(
    {'%', '*', '+', '-', '/', '<', '<=', '==', '>', '>=', 'and', 'or'})


def _name_validator(_0, _1, name):
    return is_valid_name(name)


__all__ = ['UAST', 'PAST', 'OP_STRINGS']
