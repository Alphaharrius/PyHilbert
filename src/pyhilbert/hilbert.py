import types
from dataclasses import dataclass, replace, field
from typing import Tuple
from collections import OrderedDict
from collections.abc import Iterable
from functools import lru_cache
from itertools import chain

from multipledispatch import dispatch

from .abstracts import Updatable
from .utils import FrozenDict
from .spatials import Spatial, ReciprocalLattice, cartes


@dataclass(frozen=True)
class Mode(Spatial, Updatable):
    """
    Mode:
    - r: Real space offset of the mode (unit-cell offset + basis)
    - orb: Symmetry information Orbital class (PointGroupBasis, eigenvalue)
    - spin: Spin information
    """
    count: int
    attr: FrozenDict

    @dispatch(object)
    def __getitem__(self, v):
        raise NotImplementedError(f"Get item of {type(v)} is not supported!")
    
    @dispatch(str)
    def __getitem__(self, name: str):
        return self.attr[name]

    @dispatch(tuple)
    def __getitem__(self, names: Tuple[str, ...]):
        items = {name: self.attr[name] for name in names}
        return replace(self, attr=FrozenDict(items))
    
    @property
    def dim(self) -> int:
        return self.count
    
    def _updated(self, **kwargs) -> 'Mode':
        updated_attr = {**self.attr}
        _MISSING = object()
        for k, v in kwargs.items():
            old = updated_attr.get(k, _MISSING)
            if isinstance(v, types.FunctionType):
                if old is _MISSING: 
                    continue
                updated_attr[k] = v(old)
            else:
                updated_attr[k] = v
        
        return replace(self, attr=FrozenDict(updated_attr))


@dataclass(frozen=True)
class StateSpace(Spatial):
    """
    `StateSpace` is a collection of indices with additional information attached to the elements,
    for the case of TNS there are only two types of state spaces: `MomentumSpace` and `HilbertSpace`.
    `MomentumSpace` is needed because some tensors are better represented in momentum space, e.g. Hamiltonians
    with translational symmetry, while `HilbertSpace` is needed to represent local degrees of freedom, e.g. spin or fermionic modes.

    Attributes
    ----------
    structure : OrderedDict[Spatial, slice]
        An ordered dictionary mapping each spatial component (e.g., `Offset`, `Momentum`, `Mode`) to a slice object that defines its 
        position and the range in the tensor. The slices should be contiguous and ordered.

    dim : int
        The total dimension of the state space, calculated as the count of elements regardless of their lengths.
    """

    structure: OrderedDict[Spatial, slice]
    """
    An ordered dictionary mapping each spatial component (e.g., `Offset`, `Momentum`, `Mode`) to a slice object that defines its 
    position and the range in the tensor. The slices should be contiguous and ordered.
    """
    
    @property
    def dim(self) -> int:
        """ The total dimension of the state space, calculated as the count of elements regardless of their lengths. """
        return len(self.structure)
    
    @property
    def size(self) -> int:
        """ The total size of the vector space (sum of all sector dimensions). """
        if not self.structure:
            return 0
        return self.structure[next(reversed(self.structure))].stop

    def __hash__(self):
        # TODO: Do we need to consider the order of the structure?
        return hash(tuple((k, s.start, s.stop) for k, s in self.structure.items()))


def restructure(structure: OrderedDict[Spatial, slice]) -> OrderedDict[Spatial, slice]:
    """
    Return a new `OrderedDict` with contiguous, ordered slices preserving lengths.

    Parameters
    ----------
    structure : `OrderedDict[Spatial, slice]`
        The original structure with possibly non-contiguous slices.

    Returns
    -------
    `OrderedDict[Spatial, slice]`
        The restructured `OrderedDict` with contiguous, ordered slices.
    """
    new_structure = OrderedDict()
    base = 0
    for k, s in structure.items():
        L = s.stop - s.start
        new_structure[k] = slice(base, base + L)
        base += L
    return new_structure


# TODO: We can put @lru_cache if the hashing of StateSpace is well defined
def permutation_order(src: 'StateSpace', dest: 'StateSpace') -> Tuple[int, ...]:
    """
    Return the permutation of `src` sectors needed to match `dest` sector order.

    This returns a per-sector permutation: each entry corresponds to a key in
    `dest.structure` and gives the index of the same key in `src.structure`.
    It does not expand slices; use `flat_permutation_order` to get element-wise
    indices for reordering a flattened tensor.

    Parameters
    ----------
    src : `StateSpace`
        The source state space defining the original ordering.
    dest : `StateSpace`
        The destination state space defining the target ordering.

    Returns
    -------
    `Tuple[int, ...]`
        Sector indices mapping each key in `dest` to its position in `src`
        (`-1` if missing).
    """
    order_table = {k: n for n, k in enumerate(src.structure.keys())}
    return tuple(order_table.get(k, -1) for k in dest.structure.keys())


# TODO: We can put @lru_cache if the hashing of StateSpace is well defined
def flat_permutation_order(src: 'StateSpace', dest: 'StateSpace') -> Tuple[int, ...]:
    """
    Return the flattened index permutation that reorders `src` to match `dest`.

    This expands each sector slice in `src` into its element indices, then
    concatenates those groups according to `permutation_order(src, dest)`.
    The result can be used to permute a flat vector or tensor axis from `src`
    ordering into `dest` ordering.

    Parameters
    ----------
    src : `StateSpace`
        The source state space defining the original ordering.
    dest : `StateSpace`
        The destination state space defining the target ordering.

    Returns
    -------
    `Tuple[int, ...]`
        Flattened indices that map element positions in `src` to `dest`.
    """
    index_groups = [tuple(range(s.start, s.stop)) for s in src.structure.values()]
    ordered_groups = (index_groups[i] for i in permutation_order(src, dest))
    return tuple(chain.from_iterable(ordered_groups))


# TODO: We can put @lru_cache if the hashing of StateSpace is well defined
def embedding_order(sub: 'StateSpace', sup: 'StateSpace') -> Tuple[int, ...]:
    """
    Return indices mapping `sub` into `sup` (assumes `sub` ⊆ `sup`).

    Parameters
    ----------
    sub : `StateSpace`
        The subspace whose indices are to be embedded.
    sup : `StateSpace`
        The superspace providing the full index set.

    Returns
    -------
    `Tuple[int, ...]`
        Flattened indices mapping `sub` into `sup`.
    """
    indices = []
    sup_slices = sup.structure
    for key, _ in sub.structure.items():
        if key not in sup_slices:
            raise ValueError(f"Key {key} not found in superspace")
        sup_slice = sup_slices[key]
        indices.append(range(sup_slice.start, sup_slice.stop))
    return tuple(chain.from_iterable(indices))


# TODO: We can put @lru_cache if the hashing of StateSpace is well defined
@dispatch(StateSpace, StateSpace)
def same_span(a: StateSpace, b: StateSpace) -> bool:
    return set(a.structure.keys()) == set(b.structure.keys())
    

@dispatch(StateSpace, StateSpace)
def operator_add(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        return ValueError(f'Cannot add StateSpaces of different types: {type(a)} and {type(b)}!')
    new_structure = OrderedDict(
        (*a.structure.items(), *((k, v) for k, v in b.structure.items() if k not in a.structure))
    )
    return type(a)(structure=restructure(new_structure))


@dispatch(StateSpace, StateSpace)
def operator_sub(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        return ValueError(f'Cannot subtract StateSpaces of different types: {type(a)} and {type(b)}!')
    new_structure = OrderedDict(((k, v) for k, v in a.structure.items() if k not in b.structure))
    return type(a)(structure=restructure(new_structure))


@dispatch(StateSpace, StateSpace)
def operator_or(a: StateSpace, b: StateSpace):
    return a + b


@dispatch(StateSpace, StateSpace)
def operator_and(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        return ValueError(f'Cannot intersect StateSpaces of different types: {type(a)} and {type(b)}!')
    new_structure = OrderedDict(((k, v) for k, v in a.structure.items() if k in b.structure))
    return type(a)(structure=restructure(new_structure))


@dataclass(frozen=True)
class MomentumSpace(StateSpace):
    # Ensure that __hash__ is inherited from StateSpace since the hash of StateSpace is specifically
    # designed to account for the structure attribute which is an un-hashable type OrderedDict.
    __hash__ = StateSpace.__hash__
    
    def __str__(self):
        return f'MomentumSpace({self.dim})'
    
    def __repr__(self):
        header = f'{str(self)}:\n'
        body = '\t' + '\n\t'.join([f'{n}: {k}' for n, k in enumerate(self.structure.keys())])
        return header + body


@dataclass(frozen=True)
class HilbertSpace(StateSpace, Updatable):
    __hash__ = StateSpace.__hash__

    def _updated(self, **kwargs):
        updated_structure = {}
        for m, s in self.structure.items():
            if not isinstance(m, Mode):
                raise RuntimeError(
                    f'Implementation error: found {type(m)} in HilbertSpace structure!')
            updated_m = m.update(**kwargs)
            updated_structure[updated_m] = s

        # Don't need StateSpace.restructure here since the slices are unchanged
        return HilbertSpace(structure=updated_structure)


@dispatch(Iterable)
def hilbert(itr: Iterable[Mode]) -> HilbertSpace:
    structure = OrderedDict()
    base = 0
    for mode in itr:
        structure[mode] = slice(base, base + mode.count)
        base += mode.count
    return HilbertSpace(structure=structure)


@lru_cache
def brillouin_zone(lattice: ReciprocalLattice) -> MomentumSpace:
    elements = cartes(lattice)
    structure = OrderedDict((el, slice(n, n + 1)) for n, el in enumerate(elements))
    return MomentumSpace(structure=structure)


@dataclass(frozen=True)
class BroadcastSpace(StateSpace):
    structure: OrderedDict = field(default_factory=OrderedDict)

    # Ensure that __hash__ is inherited from StateSpace since the hash of StateSpace is specifically
    # designed to account for the structure attribute which is an un-hashable type OrderedDict.
    __hash__ = StateSpace.__hash__
    
    def __repr__(self):
        return f'BroadcastSpace'
    
    __str__ = __repr__


@dispatch(BroadcastSpace, BroadcastSpace)
def same_span(a: BroadcastSpace, b: BroadcastSpace) -> bool:
    return True


@dispatch(StateSpace, BroadcastSpace)
def same_span(a: StateSpace, b: BroadcastSpace) -> bool:
    return True


@dispatch(BroadcastSpace, StateSpace)
def same_span(a: BroadcastSpace, b: StateSpace) -> bool:
    return True
