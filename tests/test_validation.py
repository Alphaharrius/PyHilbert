import pytest
from sympy import ImmutableDenseMatrix, Rational, Symbol, Integer
from pyhilbert.utils import validate_matrix
from pyhilbert.spatials import AffineSpace, Offset
from pyhilbert.basis_transform import BasisTransform

def test_validate_matrix_valid():
    m1 = ImmutableDenseMatrix([[1, 2], [3, 4]])
    validate_matrix(m1)
    
    m2 = ImmutableDenseMatrix([[Rational(1, 2), 3], [4, Rational(5, 3)]])
    validate_matrix(m2)

def test_validate_matrix_invalid():
    # Float
    m1 = ImmutableDenseMatrix([[1.5, 2], [3, 4]])
    with pytest.raises(TypeError, match="Invalid entry"):
        validate_matrix(m1)
        
    # Symbol
    x = Symbol('x')
    m2 = ImmutableDenseMatrix([[x, 2], [3, 4]])
    validate_matrix(m2)

def test_affine_space_validation():
    # Valid
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    AffineSpace(basis=basis)
    
    # Invalid
    basis_bad = ImmutableDenseMatrix([[1.0, 0], [0, 1]])
    with pytest.raises(TypeError, match="Invalid entry"):
        AffineSpace(basis=basis_bad)

def test_offset_validation():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    space = AffineSpace(basis=basis)
    
    # Valid
    rep = ImmutableDenseMatrix([[1], [2]])
    Offset(rep=rep, space=space)
    
    # Invalid Shape
    rep_bad_shape = ImmutableDenseMatrix([[1, 2]])
    with pytest.raises(ValueError, match="Invalid Shape"):
        Offset(rep=rep_bad_shape, space=space)
        
    # Invalid Type
    rep_bad_type = ImmutableDenseMatrix([[1.5], [2]])
    with pytest.raises(TypeError, match="Invalid entry"):
        Offset(rep=rep_bad_type, space=space)

def test_basis_transform_validation():
    # Valid
    M = ImmutableDenseMatrix([[1, 1], [0, 1]])
    BasisTransform(M=M)
    
    # Invalid Det
    M_det0 = ImmutableDenseMatrix([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="M must have non-zero determinant"):
        BasisTransform(M=M_det0)
        
    # Invalid Type
    M_bad_type = ImmutableDenseMatrix([[1.5, 1], [0, 1]])
    with pytest.raises(TypeError, match="Invalid entry"):
        BasisTransform(M=M_bad_type)
