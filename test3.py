import sys
import sympy as sy
from sympy import ImmutableDenseMatrix, Rational, sqrt, I, cos, sin, pi
from sympy.functions.elementary.trigonometric import acos

# Ideally we would import these from pyhilbert if they were used directly, 
# but here we are doing a custom analysis in the style of the notebook.
# We'll import something to show we are in the environment.
from pyhilbert.spatials import Lattice

# Force UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

def rodrigues_rotation(axis, angle):
    """
    Generates a 3x3 rotation matrix using Rodrigues' rotation formula symbolically.
    """
    # Ensure axis is a unit vector
    axis = ImmutableDenseMatrix(axis)
    norm = sqrt(sum(x**2 for x in axis))
    u = axis / norm
    ux, uy, uz = u[0], u[1], u[2]
    
    # K is the cross-product matrix of u
    K = ImmutableDenseMatrix([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    
    I_mat = sy.eye(3)
    
    # Rodrigues formula: R = I + (sin theta) K + (1 - cos theta) K^2
    R = I_mat + sin(angle) * K + (1 - cos(angle)) * (K @ K)
    return R

def analyze_C3_sympy():
    print("=== Analyzing 8 C3 Rotations of Diamond Lattice (SymPy / PyHilbert Style) ===")

    # 1. Define 4 Bonds
    d_vecs = [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]
    # d[i] represents the i-th bond vector
    d = [ImmutableDenseMatrix(v) for v in d_vecs]
    
    # Define symbolic basis coefficients (b0, b1, b2, b3)
    # These represent the "basis functions" as linear combinations of bonds
    b_syms = sy.symbols('b0 b1 b2 b3')
    
    # 2. Define Operations
    operations = []
    
    # (A) Identity
    operations.append(("Identity", sy.eye(3)))
    
    # (B) 8 C3 Rotations
    theta_c3 = 2 * pi / 3
    
    for i in range(4):
        axis = d[i]
        # +120 degrees
        R_plus = rodrigues_rotation(axis, theta_c3)
        operations.append((f"C3 (Axis {i}, +120 deg)", R_plus))
        
        # -120 degrees (or +240)
        R_minus = rodrigues_rotation(axis, -theta_c3)
        operations.append((f"C3 (Axis {i}, -120 deg)", R_minus))

    # 3. Calculate Permutations and Eigenvalues/Eigenvectors
    for name, R in operations:
        print(f"\n--- Operation: {name} ---")
        
        R = sy.simplify(R)
        
        # Build Permutation Matrix P
        P_rows = []
        mapping = []
        
        for j in range(4): 
            rotated_vec = R @ d[j]
            rotated_vec = sy.simplify(rotated_vec)
            
            row = [0] * 4
            found = False
            for i in range(4):
                if rotated_vec == d[i]:
                    row[i] = 1
                    mapping.append(f"{j}->{i}")
                    found = True
                    break
            if not found:
                 print(f"  Warning: Bond {j} rotated to {rotated_vec.T} is not in the original set")
            P_rows.append(row)

        P = ImmutableDenseMatrix(P_rows).T 
        
        print(f"  Permutation Mapping: {', '.join(mapping)}")
        print(f"  Permutation Matrix P:")
        rows = P.tolist()
        for row in rows:
            print(f"    {row}")
        
        # Calculate Eigenvectors
        # P.eigenvects() returns a list of tuples: (eigenvalue, multiplicity, [eigenvectors])
        eigenvectors = P.eigenvects()
        
        print("  Basis Functions (Eigenvectors of P):")
        omega = sy.exp(2*pi*I/3)
        
        for val, mult, vectors in eigenvectors:
            val_simp = sy.simplify(val)
            
            note = ""
            if val_simp == 1: note = "(1)"
            elif sy.simplify(val_simp - omega) == 0: note = "(ω)"
            elif sy.simplify(val_simp - omega**2) == 0: note = "(ω^2)"
            
            print(f"    Eigenvalue: {(val_simp)} {note}")
            
            for v in vectors:
                # v is a column vector of coefficients [c0, c1, c2, c3]
                # The basis function is c0*b0 + c1*b1 + c2*b2 + c3*b3
                basis_func = sum(v[k] * b_syms[k] for k in range(4))
                # Simplify the expression to make it readable
                basis_func = sy.simplify(basis_func)
                print(f"      -> {(basis_func)}")

if __name__ == "__main__":
    analyze_C3_sympy()
