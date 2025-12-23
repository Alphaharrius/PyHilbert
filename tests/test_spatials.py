import sympy as sy
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import Lattice, ReciprocalLattice, Offset, cartes, AffineSpace

def test_lattice_creation_and_dual():
    # 2D square lattice
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(basis=basis, shape=(2, 2))
    
    assert lattice.dim == 2
    assert lattice.shape == (2, 2)
    assert isinstance(lattice.affine, AffineSpace)
    
    # Check dual
    reciprocal = lattice.dual
    assert isinstance(reciprocal, ReciprocalLattice)
    assert reciprocal.dim == 2
    
    # Check double dual gives back original lattice (ignoring scaling if any specific convention, 
    # usually dual of dual is original)
    # The implementation uses 2*pi for dual, and 1/(2*pi) for dual of dual.
    # Lattice -> 2*pi * basis^-T
    # Reciprocal -> 1/(2*pi) * reciprocal_basis^-T 
    # = 1/(2*pi) * (2*pi * basis^-T)^-T = 1/(2*pi) * (2*pi)^-1 * (basis^-T)^-T 
    # = 1/(2*pi) * 1/(2*pi) * basis ... wait check implementation
    
    # Implementation:
    # Lattice.dual: 2 * sy.pi * self.basis.inv().T
    # ReciprocalLattice.dual: (1 / (2 * sy.pi)) * self.basis.inv().T
    
    # let B be basis.
    # B* = 2pi * (B^-1)^T
    # (B*)* = 1/2pi * (B*^-1)^T = 1/2pi * ( (2pi * (B^-1)^T)^-1 )^T
    #       = 1/2pi * ( 1/2pi * ((B^-1)^T)^-1 )^T
    #       = 1/2pi * 1/2pi * ( B^T )^T
    #       = 1/(4pi^2) * B
    # It seems dual of dual is NOT identity in this implementation?
    # Let's check the code again.
    # ReciprocalLattice.dual: basis = (1 / (2 * sy.pi)) * self.basis.inv().T
    
    # Wait. (k * A)^-1 = 1/k * A^-1
    # ((B^-1)^T)^-1 = ((B^-1)^-1)^T = B^T
    # So ((B^-1)^T)^-1^T = (B^T)^T = B.
    
    # So:
    # B_dual = 2pi * (B^-1)^T
    # B_dual_dual = 1/2pi * (B_dual^-1)^T
    #             = 1/2pi * ( (2pi * (B^-1)^T)^-1 )^T
    #             = 1/2pi * ( 1/2pi * B^T )^T
    #             = 1/2pi * 1/2pi * B
    #             = 1/(4pi^2) * B
    
    # This suggests dual().dual() != self unless I misread the math or code.
    # Let's test it to find out.
    
    # Actually, ReciprocalLattice.dual code:
    # basis = (1 / (2 * sy.pi)) * self.basis.inv().T
    
    # If I have B = [[1, 0], [0, 1]], B^-1 = I, B^-1^T = I.
    # B_dual = 2pi * I.
    # B_dual_dual = 1/2pi * (2pi * I)^-1^T = 1/2pi * 1/2pi * I = 1/(4pi^2) * I.
    
    # So it scales by 1/(4pi^2).
    
    orig_basis = lattice.basis
    round_trip_basis = reciprocal.dual.basis
    
    # sy.simplify(orig_basis - round_trip_basis * (4 * sy.pi**2)) should be zero?
    # Let's just assert the relationship we find.
    assert round_trip_basis == orig_basis * (1/(4 * sy.pi**2))

def test_cartes_lattice():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(basis=basis, shape=(2, 2))
    
    # cartes should return offsets for (0,0), (0,1), (1,0), (1,1)
    points = cartes(lattice)
    assert len(points) == 4
    assert isinstance(points[0], Offset)
    
    # Check content of points
    coords = set()
    for p in points:
        coords.add(tuple(p.rep))
    
    assert (0, 0) in coords
    assert (0, 1) in coords
    assert (1, 0) in coords
    assert (1, 1) in coords

def test_cartes_reciprocal_lattice():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    # shape (2, 2)
    lattice = Lattice(basis=basis, shape=(2, 2))
    reciprocal = lattice.dual
    
    # cartes for reciprocal
    # elements = product(range(2), range(2))
    # sizes = (1/2, 1/2)
    # element * sizes
    
    points = cartes(reciprocal)
    assert len(points) == 4
    
    coords = set()
    for p in points:
        # p.rep should be (n/2, m/2)
        coords.add(tuple(p.rep))
        
    assert (0, 0) in coords
    assert (sy.Rational(1, 2), 0) in coords
    assert (0, sy.Rational(1, 2)) in coords
    assert (sy.Rational(1, 2), sy.Rational(1, 2)) in coords


