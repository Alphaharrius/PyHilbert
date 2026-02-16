import pytest
import sympy as sy
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix

from pyhilbert.hilbert_space import Ket, U1State, U1Span, HilbertSpace, hilbert
from pyhilbert.state_space import MomentumSpace, brillouin_zone
from pyhilbert.spatials import Lattice, Offset


@dataclass(frozen=True)
class Orb:
    name: str


def _state(r: Offset, orb: str = "s", irrep: sy.Expr = sy.Integer(1)) -> U1State:
    return U1State(irrep=irrep, kets=(Ket(r), Ket(Orb(orb))))


def test_u1_state_basic_properties_and_overlap():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)

    psi = _state(r0, "s")
    psi_scaled = _state(r0, "s", sy.Integer(2))

    assert psi.dim == 1
    assert psi.ket(psi_scaled) == sy.Integer(2)
    assert psi.unit() == _state(r0, "s", sy.Integer(1))


def test_u1_state_irrep_access_and_replace():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)

    psi = _state(r0, "p")
    assert psi.irrep_of(Offset) == r0

    replaced = psi.replace(r1)
    assert replaced.irrep_of(Offset) == r1
    assert replaced.kets[1] == Ket(Orb("p"))


def test_u1_state_rejects_non_unity_type_multiplicity():
    with pytest.raises(ValueError, match="unity multiplicity"):
        U1State(irrep=sy.Integer(1), kets=(Ket("a"), Ket("b")))


def test_u1_span_addition_and_deduplication():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)

    a = _state(r0, "s")
    b = _state(r1, "s")

    span = a + a
    assert isinstance(span, U1Span)
    assert span.dim == 1

    span2 = span + b
    assert span2.dim == 2


def test_hilbert_space_creation_and_operations():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(3,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    r2 = Offset(rep=ImmutableDenseMatrix([2]), space=lat.affine)

    s0 = _state(r0, "s")
    s1 = _state(r1, "s")
    s2 = _state(r2, "s")

    hs1 = hilbert([s0, s1])
    hs2 = hilbert([s1, s2])

    assert isinstance(hs1, HilbertSpace)
    assert hs1.dim == 2
    assert len(hs1.elements()) == 2

    union = hs1 + hs2
    assert list(union.structure.keys()) == [s0, s1, s2]

    inter = hs1 & hs2
    assert list(inter.structure.keys()) == [s1]

    diff = hs1 - hs2
    assert list(diff.structure.keys()) == [s0]


def test_statespace_getitem_variants():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(3,))
    states = [
        _state(Offset(rep=ImmutableDenseMatrix([i]), space=lat.affine), "s")
        for i in range(3)
    ]
    hs = hilbert(states)

    assert hs[0] == states[0]
    assert hs[-1] == states[-1]

    hs_slice = hs[1:3]
    assert isinstance(hs_slice, HilbertSpace)
    assert list(hs_slice.structure.keys()) == [states[1], states[2]]
    assert hs_slice.dim == 2

    hs_range = hs[range(0, 2)]
    assert list(hs_range.structure.keys()) == [states[0], states[1]]

    with pytest.raises(IndexError):
        _ = hs[3]

    with pytest.raises(TypeError):
        _ = hs["bad"]


def test_hilbert_space_gram_diagonal_for_identical_basis():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(2,))
    a = _state(
        Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine), "s", sy.Integer(2)
    )
    b = _state(
        Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine), "s", sy.Integer(3)
    )
    hs = hilbert([a, b])

    gram = hs.gram(hs)
    assert gram.data.shape == (2, 2)
    assert gram.data[0, 0] == 4
    assert gram.data[1, 1] == 9
    assert gram.data[0, 1] == 0


def test_hilbert_space_lookup_exact_query_match():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(3,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    r2 = Offset(rep=ImmutableDenseMatrix([2]), space=lat.affine)

    hs = hilbert([_state(r0, "s"), _state(r1, "p"), _state(r2, "s")])
    found = hs.lookup({Offset: r1, Orb: Orb("p")})
    assert found == _state(r1, "p")


def test_hilbert_space_lookup_errors_for_no_or_multiple_matches():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    hs = hilbert([_state(r0, "s"), _state(r1, "s")])

    with pytest.raises(ValueError, match="No state found"):
        hs.lookup({Offset: Offset(rep=ImmutableDenseMatrix([3]), space=lat.affine)})

    with pytest.raises(ValueError, match="Multiple states found"):
        hs.lookup({Orb: Orb("s")})

    with pytest.raises(ValueError, match="cannot be empty"):
        hs.lookup({})


def test_momentum_space_brillouin():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lat = Lattice(basis=basis, shape=(2, 2))
    recip = lat.dual

    ms = brillouin_zone(recip)
    assert isinstance(ms, MomentumSpace)
    assert ms.dim == 4

    assert str(ms) == "MomentumSpace(4)"
    assert "MomentumSpace(4):" in repr(ms)


def test_statespace_type_errors():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(basis=basis, shape=(1,))
    s = _state(Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine), "s")
    hs = hilbert([s])
    ms = MomentumSpace(structure={})

    with pytest.raises(ValueError, match="different types"):
        hs + ms

    with pytest.raises(ValueError, match="different types"):
        hs - ms

    with pytest.raises(ValueError, match="different types"):
        hs & ms
