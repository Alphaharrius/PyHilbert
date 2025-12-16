from dataclasses import dataclass
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
    
    def extend(self, **kwargs) -> 'Mode':
        extended_attr = FrozenDict({**self.attr, **kwargs})
        return Mode(count=self.count, attr=extended_attr)
    
    @property
    def dim(self) -> int:
        return self.count
    
    def _updated(self, **kwargs) -> 'Mode':
        updated_attr = {**self.attr}
        for k, v in kwargs.items():
            if callable(v):
                if k not in updated_attr: continue # skip if key not present
                updated_attr[k] = v(updated_attr[k])
                continue

            updated_attr[k] = v # Insert or update directly
        
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
    
    @staticmethod
    def restructure(structure: OrderedDict[Spatial, slice]) -> OrderedDict[Spatial, slice]:
        """ Return a new OrderedDict with contiguous, ordered slices preserving lengths. """
        new_structure = OrderedDict()
        base = 0
        for k, s in structure.items():
            L = s.stop - s.start
            new_structure[k] = slice(base, base + L, 1)
            base += L
        return new_structure
    
    @staticmethod
    @lru_cache
    def permutation_order(src: 'StateSpace', dest: 'StateSpace') -> Tuple[int, ...]:
        order_table = {k: n for n, k in enumerate(src.structure.keys())}
        return tuple(order_table.get(k, -1) for k in dest.structure.keys())
    
    @staticmethod
    @lru_cache
    def flat_permutation_order(src: 'StateSpace', dest: 'StateSpace') -> Tuple[int, ...]:
        index_groups = [tuple(range(s.start, s.stop)) for s in src.structure.values()]
        ordered_groups = (index_groups[i] for i in StateSpace.permutation_order(src, dest))
        return tuple(chain.from_iterable(ordered_groups))

    def __hash__(self):
        # TODO: Do we need to consider the order of the structure?
        return hash(tuple((k, s.start, s.stop) for k, s in self.structure.items()))
    

@dispatch(StateSpace, StateSpace)
def operator_add(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        return ValueError(f'Cannot add StateSpaces of different types: {type(a)} and {type(b)}!')
    new_structure = OrderedDict(
        (*a.structure.items(), *((k, v) for k, v in b.structure.items() if k not in a.structure))
    )
    return type(a)(structure=StateSpace.restructure(new_structure))


@dispatch(StateSpace, StateSpace)
def operator_sub(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        return ValueError(f'Cannot subtract StateSpaces of different types: {type(a)} and {type(b)}!')
    new_structure = OrderedDict(((k, v) for k, v in a.structure.items() if k not in b.structure))
    return type(a)(structure=StateSpace.restructure(new_structure))


@dispatch(StateSpace, StateSpace)
def operator_or(a: StateSpace, b: StateSpace):
    return a + b


@dispatch(StateSpace, StateSpace)
def operator_and(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        return ValueError(f'Cannot intersect StateSpaces of different types: {type(a)} and {type(b)}!')
    new_structure = OrderedDict(((k, v) for k, v in a.structure.items() if k in b.structure))
    return type(a)(structure=StateSpace.restructure(new_structure))


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
