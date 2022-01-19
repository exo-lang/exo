
#import inspect
#import types
from weakref import WeakKeyDictionary

from collections    import defaultdict #ChainMap, OrderedDict
from itertools      import chain
from dataclasses    import dataclass
from typing         import Any, Optional
from enum           import Enum

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
#  O(#P * #G) where #E is the number of procedures being tracked
# for equivalence, and #G is the number of global variables being tracked
# for the sake of equivalence modulo that global.
# This seems like it must be sub-optimal, but it's probably a practical
# enough solution for a long time, so long as the number of globals
# in programs remains small enough to be treated as a constant factor



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   basic union find datastructure

class _UnionFind:
    def __init__(self):
        self.lookup = WeakKeyDictionary()

    def new_node(self, val):
        self.lookup[val] = val
        return val

    def find(self, val):
        parent = self.lookup[val]
        while val is not parent:
            # path splitting optimization
            grandparent         = self.lookup[parent]
            self.lookup[val]    = grandparent
            val, parent         = parent, grandparent
        return val

    def union(self, val1, val2):
        p1, p2 = self.find(val1), self.find(val2)

        if p1 is p2:
            pass
        else:
            self.lookup[p2] = p1

    def check_eqv(self, val1, val2):
        p1, p2 = self.find(val1), self.find(val2)
        return p1 is p2

    def copy_entire_UF(self):
        copy = _UnionFind()
        for v,p in self.lookup.items():
            copy.lookup[v] = p
        return copy


_UF_Unv         = _UnionFind()
_UF_Strict      = _UnionFind()
_UF_Unv_key     = dict()


def new_uf_by_eqv_key(key):
    assert key not in _UF_Unv_key
    _UF_Unv_key[key] = _UF_Unv.copy_entire_UF()

def decl_new_proc(proc):
    # add to all existing union-find data structures
    _UF_Strict.new_node(proc)
    _UF_Unv.new_node(proc)
    for _,uf in _UF_Unv_key.items():
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
    if len(config_set) == 0:
        _UF_Strict.union(proc1, proc2)
    _UF_Unv.union(proc1, proc2)
    for key,uf in _UF_Unv_key.items():
        if key not in config_set:
            uf.union(proc1, proc2)

def check_eqv_proc(proc1,proc2, config_set=frozenset()):
    assert isinstance(config_set, frozenset)
    # if these aren't equal under the weakest assumptions
    # then we can early exit
    if not _UF_Unv.check_eqv(proc1, proc2):
        return False
    # otherwise check intersection of all non-excluded equivalence relations
    return all( uf.check_eqv(proc1, proc2)
                for key,uf in _UF_Unv_key.items()
                if key not in config_set )

def get_strictest_eqv_proc(proc1,proc2):
    # under the weakest assumptions, are these procedures equivalent?
    is_eqv = _UF_Unv.check_eqv(proc1, proc2)
    # then compute the strongest assumptions under which the equivalence
    # continues to hold.  Note that keys == emptyset() is the strictest
    keys = set()
    if is_eqv:
        keys = { key for key,uf in _UF_Unv_key.items()
                     if not uf.check_eqv(proc1,proc2) }

    return is_eqv, keys

def get_repr_proc(q_proc):
    proc = _UF_Strict.find(q_proc)
    return proc


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   proc equivalence classes with representatives

"""
@dataclass
class EqvNode:
    proc        : LoopIR.proc
    parent      : LoopIR.proc
    # This node represents evidence that proc == parent modulo
    # the variables declared in eqv_set
    eqv_set     : frozenset



class _EquivalenceManager_Class:
    def __init__(self):
        self.lookup = WeakKeyDictionary()

    def new_set(self, proc):
        node = EqvNode(proc, proc, frozenset())
        self.lookup[proc] = node
        return node

    # use path splitting, conditional on equivalence
    def find(self, proc, eqv_set):
        node    = self.lookup[proc]
        # we are searching for a representative procedure
        # modulo eqv_set.  is_eqv represents whether or not
        # the equivalence represented by node is sufficiently
        # strong to be traversed "modulo eqv_set"
        is_eqv  = node.eqv_set.issubset(eqv_set)
        while is_eqv and node.parent is not node.proc:
            parent_node = self.lookup[node.parent]
            # only do the path splitting optimization
            # when the two equivalences are modulo the same set
            if node.eqv_set == parent_node.eqv_set:
                node.parent = parent_node.parent
            node        = parent_node
            is_eqv      = node.eqv_set.issubset(eqv_set)

        return node

    # TODO: This is almost certainly subtly incorrect
    def union(self, proc1, proc2, eqv_set):
        node1   = self.find(proc1, eqv_set)
        node2   = self.find(proc2, eqv_set)

        if node1 == node2:
            return # nothing to be done

        # do we need to further union node2's ancestors?
        parent2     = node2.parent
        continue2   = (node2.parent is not node2.proc)
        eqv_set2    = eqv_set.union(node2.eqv_set)

        # regardless, make node1 the new root of node2
        node2.parent    = node1.proc
        node2.eqv_set   = eqv_set

        # now continue with a weaker union operation if needed
        if continue2:
            self.union(node1.proc, parent2, eqv_set2)

    def check_eqv(self, proc1, proc2, eqv_set=frozenset()):
        node1   = self.find(proc1, eqv_set)
        node2   = self.find(proc2, eqv_set)

        if node1.proc is node2.proc:
            return eqv_set
        elif node1.parent is node1.proc and node2.parent is node2.proc:
            return False
        else:
            return self.check_eqv(node1.proc, node2.proc,
                                  node1.eqv_set.union(node2.eqv_set))


_EqvManager = _EquivalenceManager_Class()


def decl_new_proc(orig_proc):
    _EqvManager.new_set(orig_proc)

def derive_proc(orig_proc, new_proc, config_set=frozenset()):
    assert isinstance(config_set, frozenset)
    _EqvManager.new_set(new_proc)
    _EqvManager.union(orig_proc, new_proc, config_set)

def assert_eqv_proc(proc1, proc2, config_set=frozenset()):
    _EqvManager.union(proc1, proc2, config_set)

def check_eqv_proc(proc1,proc2):
    return _EqvManager.check_eqv(proc1,proc2)

def get_repr_proc(proc, config_set=frozenset()):
    node = _EqvManager.find(proc, config_set)
    return node.proc
"""

