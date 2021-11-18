
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

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   proc equivalence classes with representatives

@dataclass
class EqvNode:
    proc        : LoopIR.proc
    parent      : LoopIR.proc
    eqv_set     : frozenset
    # if this is the root, then we also stash
    # an alpha renamed copy
    #uniq_copy   : Optional[LoopIR.proc]



class _EquivalenceManager_Class:
    def __init__(self):
        #self.equiv_tables = defaultdict(WeakKeyDictionary)
        self.lookup = WeakKeyDictionary()

    def new_set(self, proc):
        node = EqvNode(proc, proc, frozenset())
        self.lookup[proc] = node
        return node

    #def new_derivation(orig_proc, new_proc, eqv_set=frozenset()):
    #    node = EqvNode(new_proc, orig_proc, eqv_set, None)

    # use path splitting, conditional on equivalence
    def find(self, proc, eqv_set):
        node    = self.lookup[proc]
        is_eqv  = node.eqv_set.issubset(eqv_set)
        while is_eqv and node.parent is not node.proc:
            parent_node = self.lookup[node.parent]
            if node.eqv_set == parent_node.eqv_set:
                node.parent = parent_node.parent
            node = parent_node

        return node

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


