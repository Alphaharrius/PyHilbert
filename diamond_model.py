
import sys
import os

# Add src to path to allow importing pyhilbert
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix
import matplotlib.pyplot as plt

# PyHilbert imports
from pyhilbert.spatials import Lattice, Offset, PointGroupBasis
from pyhilbert.hilbert import Mode, hilbert, brillouin_zone
from pyhilbert.utils import FrozenDict

def create_diamond_model():
    """
    Creates a tight-binding model for a Diamond lattice structure.
    Diamond = FCC lattice + 2-atom basis (A and B).
    """
    
    print("Step 1: Define the FCC Lattice")
    # Define lattice constant
    a = sy.Symbol('a', real=True, positive=True)

    # FCC lattice vectors (Primitive)
    # a1 = (0, 0.5, 0.5)a
    # a2 = (0.5, 0, 0.5)a
    # a3 = (0.5, 0.5, 0)a
    basis_matrix = ImmutableDenseMatrix([
        [0, a/2, a/2],      
        [a/2, 0, a/2],      
        [a/2, a/2, 0]       
    ])

    # Create real space lattice
    N = 4
    fcc_lattice = Lattice(basis=basis_matrix, shape=(N, N, N))
    print(f"Lattice created: {fcc_lattice}")

    print("\nStep 2: Define Sublattices (2-atom basis)")
    # Sublattice A at origin (0, 0, 0)
    sublattice_A = Offset(
        rep=ImmutableDenseMatrix([0, 0, 0]),
        space=fcc_lattice.affine
    )
    
    # Sublattice B at (1/4, 1/4, 1/4) in terms of cubic conventional cell
    # In terms of primitive vectors: 
    # (1/4)(a1 + a2 + a3) = (1/4)(1, 1, 1)a (Cartesian)
    # So we can just define it relative to the affine space origin or basis.
    # Let's express it in the basis vectors for clarity if needed, 
    # or just Cartesian coordinates relative to a.
    # 0.25 * (a1 + a2 + a3)
    vec_B = 0.25 * (basis_matrix.col(0) + basis_matrix.col(1) + basis_matrix.col(2))
    
    sublattice_B = Offset(
        rep=vec_B,
        space=fcc_lattice.affine
    )

    print("\nStep 3: Build Hilbert Space (Spinless for clarity)")
    # We will use just 1 orbital per site (s-orbital) and ignore spin 
    # to highlight the A-B neighbor interaction.
    
    x, y, z = sy.symbols('x y z', real=True)
    s_orbital = PointGroupBasis(
        expr=sy.Integer(1),
        axes=(x, y, z),
        order=0,
        rep=ImmutableDenseMatrix([1])
    )
    
    modes = []
    # Create modes for Sublattice A and B
    for label, offset in [('A', sublattice_A), ('B', sublattice_B)]:
        mode = Mode(
            count=1,
            attr=FrozenDict({
                'sublattice': label,
                'position': offset,
                'orbital': s_orbital,
                'spin': 'none' # Spinless
            })
        )
        modes.append(mode)

    H_space = hilbert(modes)
    print(f"Hilbert space dimension: {H_space.dim}") # Should be 2 (A and B)

    print("\nStep 4: Get Reciprocal Lattice")
    reciprocal_lattice = fcc_lattice.dual
    BZ = brillouin_zone(reciprocal_lattice)

    return fcc_lattice, reciprocal_lattice, H_space, BZ

def compute_diamond_bands(H_space, params):
    """
    Computes band structure for Diamond lattice (Nearest Neighbor).
    """
    # High symmetry points
    high_sym_points = {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'X': np.array([ 0.5, 0.0, 0.5]), # In primitive basis? Need to check convention.
        # Standard FCC path often used: Gamma -> X -> W -> K -> Gamma -> L
        # Let's use simple path: L -> Gamma -> X -> W -> K
        'L': np.array([0.5, 0.5, 0.5]),
        'K': np.array([0.375,0.375, 0.75]),
        'W': np.array([0.5,0.25,0.75]) ,
        'U': np.array([0.625, 0.25, 0.625])
    }
    
    # Let's use a simpler path for demo: L - Gamma - X
    path_labels = ['Gamma', 'X', 'W', 'K', 'Gamma', 'L', 'U', 'W', 'L', 'K']
    # Coordinates in terms of Reciprocal Basis vectors (b1, b2, b3)
    # Note: High sym points definition depends heavily on basis definition.
    # Using standard labels for now.
    
    # Create path
    k_path = []
    k_distances = [0]
    current_distance = 0
    num_points = 50
    
    boundaries = [0]
    
    for i in range(len(path_labels)-1):
        start = high_sym_points[path_labels[i]]
        end = high_sym_points[path_labels[i+1]]
        
        segment_k = np.linspace(start, end, num_points)
        for j, k_pt in enumerate(segment_k):
            if j == 0 and i > 0: continue # Avoid duplicate points
            
            k_path.append(k_pt)
            if len(k_path) > 1:
                # Approximate distance in reciprocal space units
                dist = np.linalg.norm(k_path[-1] - k_path[-2])
                current_distance += dist
                k_distances.append(current_distance)
        
        boundaries.append(len(k_path)-1)

    k_path = np.array(k_path)
    eigenvalues = []
    
    # Interaction Parameter (Hopping energy)
    t = params.get('t', 1.0)
    
    # 4 Nearest Neighbors in Diamond:
    # 1. Inside same unit cell (0,0,0) -> connect A to B
    # 2. In cell (-1,0,0) -> connect A to B
    # 3. In cell (0,-1,0) -> connect A to B
    # 4. In cell (0,0,-1) -> connect A to B
    # (Indices refer to primitive lattice vectors a1, a2, a3)
    
    for k_vec in k_path:
        kx, ky, kz = k_vec # Fractional coords in b1, b2, b3 basis
        
        # Phase factors for the 4 neighbors
        # Neighbor 1: Same cell => phase 1
        # Neighbor 2: -a1 direction => phase exp(-i * k * a1) = exp(-i * 2pi * kx)
        # Neighbor 3: -a2 direction => phase exp(-i * 2pi * ky)
        # Neighbor 4: -a3 direction => phase exp(-i * 2pi * kz)
        
        p1 = 1.0
        p2 = np.exp(-1j * 2 * np.pi * kx)
        p3 = np.exp(-1j * 2 * np.pi * ky)
        p4 = np.exp(-1j * 2 * np.pi * kz)
        
        # Off-diagonal hopping term
        # H_AB = t * (sum of phase factors)
        h_ab = t * (p1 + p2 + p3 + p4)
        
        # Hamiltonian Matrix (2x2)
        # | E_A    H_AB |
        # | H_BA   E_B  |
        # Assume E_A = E_B = 0
        
        H = np.array([
            [0, h_ab],
            [np.conj(h_ab), 0]
        ], dtype=complex)
        
        evals = np.linalg.eigvalsh(H)
        eigenvalues.append(evals)
        
    return np.array(k_distances), np.array(eigenvalues), boundaries, path_labels

if __name__ == "__main__":
    lattice, recip, H_space, BZ = create_diamond_model()
    
    print("\nComputing bands...")
    params = {'t': 1.0}
    k_dist, evals, boundaries, labels = compute_diamond_bands(H_space, params)
    
    plt.figure(figsize=(10, 6))
    for i in range(H_space.dim):
        plt.plot(k_dist, evals[:, i], label=f'Band {i+1}')
        
    for b in boundaries:
        plt.axvline(k_dist[b], color='gray', linestyle=':', alpha=0.5)
        
    # Set xticks
    tick_locs = [k_dist[b] for b in boundaries]
    plt.xticks(tick_locs, labels)
    
    plt.ylabel("Energy")
    plt.title("Diamond Lattice Tight-Binding (2 Bands, 4 Neighbors)")
    plt.grid(True, alpha=0.3)
    plt.show()






