from dataclasses import dataclass
from sympy import ImmutableDenseMatrix
import sympy as sy
from sympy.matrices.normalforms import smith_normal_decomp  # type: ignore[import-untyped]
from functools import lru_cache
from itertools import product
from typing import Tuple, Any, cast, Dict, Callable, ClassVar, Union
from multipledispatch import dispatch  # type: ignore[import-untyped]
from collections import OrderedDict
import torch
import numpy as np
from abc import ABC
from .utils import FrozenDict
from .spatials import Lattice, ReciprocalLattice, Spatial, Offset, Momentum, AffineSpace
from .hilbert import HilbertSpace, MomentumSpace, Mode, restructure, StateSpace
from .tensors import Tensor
from .fourier import fourier_transform

@dataclass(frozen=True)
class AbstractTransform(ABC):
    _register_transform_method: ClassVar[Dict[Tuple[type, type], Callable]] = {}

    @classmethod
    def register_transform_method(cls, obj_type: type):
        """Register a transform method for a specific object type."""
        def decorator(func: Callable):
            key = (obj_type, cls)
            cls._register_transform_method[key] = func
            return func
        return decorator
    
    def transform(self, obj: Any, **kwargs) -> Any:
        transform_class = type(self)
        obj_class = type(obj)
        key = (obj_class, transform_class)
        
        # Use the correct attribute name
        callable = self._register_transform_method.get(key)
        
        if callable is None:
            raise NotImplementedError(
                f"No transform registered for {obj_class.__name__} "
                f"with {transform_class.__name__}"
            )
        
        return callable(self, obj, **kwargs)

    def __call__(self, obj: Any, **kwargs) -> Any:
        return self.transform(obj, **kwargs)

@dataclass(frozen=True)
class BasisTransform(AbstractTransform):
    M: ImmutableDenseMatrix
    def __post_init__(self):
        if self.M.det() == 0:
            raise ValueError("M must have non-zero determinant")

@lru_cache
def _supercell_shifts(
    dim: int, M: ImmutableDenseMatrix
) -> Tuple[ImmutableDenseMatrix, ...]:
    """
    Generate the integer shifts within the supercell defined by M.
    """
    S, U, V = smith_normal_decomp(M, domain=sy.ZZ)
    Q = V.inv()
    ranges = [range(int(S[i, i])) for i in range(dim)]
    shifts = [ImmutableDenseMatrix([n]) @ Q for n in product(*ranges)]
    return tuple(shifts)


@BasisTransform.register_transform_method(AffineSpace)
def affine_transform(t: AbstractTransform, space: AffineSpace) -> AffineSpace:
    """
    Transform an AffineSpace by the basis transformation M.
    """
    new_basis = t.M @ space.basis
    return AffineSpace(basis=new_basis)


@BasisTransform.register_transform_method(Lattice)
def lattice_transform(t: AbstractTransform, lat: Lattice) -> Lattice:
    """
    Generates a Supercell based on the scaling matrix M.
    Automatically populates the new unit cell with original atoms
    to preserve physical density.
    """
    # 1. Validate M
    shifts = _supercell_shifts(lat.dim, t.M)

    # 4. Transform Atoms
    M_inv = t.M.inv()
    new_unit_cell = {}

    # Iterate over existing atoms (or implicit origin)
    items = lat.unit_cell.items() if lat.unit_cell else [("0", [0] * lat.dim)]
    for label, atom in items:
        atom_vec = ImmutableDenseMatrix(atom).reshape(1, lat.dim)
        for i, k in enumerate(shifts):
            # Now both atom_vec and k are 1xN Matrices
            # Formula: new_frac = (old_frac + shift) * M^-1
            new_frac = (atom_vec + k) @ M_inv
            new_frac = new_frac.applyfunc(lambda x: x - sy.floor(x))

            # Generate new label
            new_label = f"{label}_{i}" if len(shifts) > 1 else label
            new_unit_cell[new_label] = new_frac
    new_basis = t.M @ lat.basis
    return Lattice(
        basis=new_basis, shape=lat.shape, unit_cell=FrozenDict(new_unit_cell)
    )

@BasisTransform.register_transform_method(ReciprocalLattice)
def reciprocal_lattice_transform(t: AbstractTransform, lat: ReciprocalLattice) -> ReciprocalLattice:
    """
    Generate the reciprocal lattice corresponding to the transformed direct lattice.
    """
    dual_lat = lat.dual
    transformed_dual_lat = t(dual_lat)
    return transformed_dual_lat.dual

    

@BasisTransform.register_transform_method(Offset)
def offset_transform(t: AbstractTransform, r: Offset) -> Offset:
    """

    Transform an Offset by the basis transformation M.
    """
    new_space = t(r.space)
    return r.rebase(new_space)

@BasisTransform.register_transform_method(Momentum)
def momentum_transform(t: AbstractTransform, momentum: Momentum) -> Momentum:
    """
    Docstring for momentum_transform
    
    Parameters
    ----------
    """
    new_space = t(momentum.space)
    return momentum.rebase(new_space)



def bandfold(M: ImmutableDenseMatrix, tensor: Tensor) -> Tensor:
    """
    make Tensor with (Momentum, Hilbert, Hilbert) to (scaled Momentum, Hilbert, Hilbert)
    Parameters
    ----------
    """
    pass








