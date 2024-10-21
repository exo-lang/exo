# import inspect
# import types
from weakref import WeakKeyDictionary

from collections import defaultdict  # ChainMap, OrderedDict
from itertools import chain
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum

from .LoopIR import LoopIR, T, Alpha_Rename

# This file keeps track of equivalence of procedures due to
# scheduling.  We sometimes refer to this as "provenance" since
# equality is tracked based on the transitive chain of equivalences
# generated through the act of scheduling

# A lot of this is taken from the Wikipedia article
# on disjoint-set data structures (i.e. union-find)

# However, the whole issue is complicated by the idea that we want to
# reason about equivalence "modulo a set of keys"
# (these keys are global/configuration variables)
# For instance, consider two procedures x and y.  We want to say
# that they're equivalent "modulo"/"except for" a set {a,b,c}
#       Write this x =={a,b,c} y.
#
# If L, K are sets s.t. L <= K (is a subset), then
#       x ==L y   ==>   x ==K y
#
# Furthermore, we want to observe a further property between our equivalences.
# If L, K are sets and L^K is their intersection,
#       (x ==L y /\ x ==K y)   ==>   x ==(L^K) y
#
# From these two properties, we can see that we have a semi-lattice defined
# on our entire space of different equivalence relations, where the
# equivalence corresponding to the emptyset is the strictest equality
# (i.e. the fewest things are said to be equal).
#
# If we could somehow enumerate a set of all the "weakest" equalities
# then we could track all of those equivalence relations and reconstruct
# an equivalence query for a particular set of keys by intersecting all
# of the appropriate "basis"/"atomic" relations.  Unfortunately the set
# of possible global variables is not bounded a priori.
#
# Despite this fact, we can track everything by hypothesizing a "universe"
# of all keys called `Unv`.  Then at any point in time where this code module
# has been made aware of keys {x1, x2, ..., xn} we can track n+1 equivalence
# relations, corresponding to
#   Unv, Unv-{x1}, Unv-{x2}, ..., Unv-{xn}
# If we add a new global key `y`, then we can form `Unv-{y}` to begin with
# by simply copying the equivalence relation `Unv`, which must be the same
# as `Unv-{y}` until we start observing equivalences modulo `y` (at which
# point those equivalences WILL be added to `Unv` but not to `Unv-{y}`)
#
# We can always recover the strictest equality `x =={} y` by intersecting
# all currently tracked equivalence relations.  Operationally, this can
# be achieved by checking that an equivalence is in all tracked equivalences
#

# This whole scheme will almost certainly scale empirically as
#  O(a(#P) * #G) where #P is the number of procedures being tracked
# for equivalence, and #G is the number of global variables being tracked
# for the sake of equivalence modulo that global.
# Because #G does not get very large and because a(#P) is
# significantly sublinear, we expect that this scheme will suffice
# for a good long while.  (Although note that memory consumption
# is O(#P * #G))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   basic union find datastructure


class _UnionFind:
    def __init__(self):
        self.lookup = WeakKeyDictionary()

    def new_node(self, val):
        if val not in self.lookup:
            self.lookup[val] = val

    def find(self, val):
        parent = self.lookup[val]
        while val is not parent:
            # path splitting optimization
            grandparent = self.lookup[parent]
            self.lookup[val] = grandparent
            val, parent = parent, grandparent
        return val

    def union(self, val1, val2):
        p1, p2 = self.find(val1), self.find(val2)

        if p1 is p2:
            pass  # then val1 and val2 are already unified
        else:
            self.lookup[p2] = p1

    def check_eqv(self, val1, val2):
        p1, p2 = self.find(val1), self.find(val2)
        return p1 is p2

    def copy_entire_UF(self):
        copy = _UnionFind()
        for v, p in self.lookup.items():
            copy.lookup[v] = p
        return copy


_UF_Unv = _UnionFind()
_UF_Strict = _UnionFind()
_UF_Unv_key = dict()


def new_uf_by_eqv_key(key):
    assert key not in _UF_Unv_key
    _UF_Unv_key[key] = _UF_Unv.copy_entire_UF()


def decl_new_proc(proc):
    # add to all existing union-find data structures
    _UF_Strict.new_node(proc)
    _UF_Unv.new_node(proc)
    for uf in _UF_Unv_key.values():
        uf.new_node(proc)


def derive_proc(orig_proc, new_proc, config_set=frozenset()):
    decl_new_proc(new_proc)
    assert_eqv_proc(orig_proc, new_proc, config_set)


def assert_eqv_proc(proc1, proc2, config_set=frozenset()):
    assert isinstance(config_set, frozenset)
    # First, expand the set of equivalences being tracked if needed
    for key in config_set:
        if key not in _UF_Unv_key:
            new_uf_by_eqv_key(key)
    # then do the appropriate union operations
    if not config_set:
        _UF_Strict.union(proc1, proc2)
    _UF_Unv.union(proc1, proc2)
    for key, uf in _UF_Unv_key.items():
        if key not in config_set:
            uf.union(proc1, proc2)


def check_eqv_proc(proc1, proc2, config_set=frozenset()):
    assert isinstance(config_set, frozenset)
    # if these aren't equal under the weakest assumptions
    # then we can early exit
    if not _UF_Unv.check_eqv(proc1, proc2):
        return False
    # otherwise check intersection of all non-excluded equivalence relations
    return all(
        uf.check_eqv(proc1, proc2)
        for key, uf in _UF_Unv_key.items()
        if key not in config_set
    )


def get_strictest_eqv_proc(proc1, proc2):
    # under the weakest assumptions, are these procedures equivalent?
    is_eqv = _UF_Unv.check_eqv(proc1, proc2)
    # then compute the strongest assumptions under which the equivalence
    # continues to hold.  Note that keys == emptyset() is the strictest
    keys = set()
    if is_eqv:
        keys = {
            key for key, uf in _UF_Unv_key.items() if not uf.check_eqv(proc1, proc2)
        }

    return is_eqv, keys


def get_repr_proc(q_proc):
    proc = _UF_Strict.find(q_proc)
    return proc
