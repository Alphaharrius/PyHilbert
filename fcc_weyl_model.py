
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
from pyhilbert.spatials import Lattice, Offset, PointGroupBasis, cartes
from pyhilbert.hilbert import Mode, hilbert, brillouin_zone
from pyhilbert.utils import FrozenDict

def create_fcc_weyl_model():
    """
    Creates a tight-binding model for a Weyl semimetal on an FCC lattice (1-atom basis).
    """
    
    print("Step 1: Define the FCC Lattice")
    # Define lattice constant
    a = sy.Symbol('a', real=True, positive=True)

    # FCC lattice vectors (columns are basis vectors)
    basis_matrix = ImmutableDenseMatrix([
        [0, a/2, a/2],      # a1
        [a/2, 0, a/2],      # a2
        [a/2, a/2, 0]       # a3
    ])

    # Create real space lattice (N x N x N unit cells)
    N = 4
    fcc_lattice = Lattice(basis=basis_matrix, shape=(N, N, N))
    print(f"Lattice created: {fcc_lattice}")

    print("\nStep 2: Define Sublattice (1-atom basis)")
    # Sublattice A at origin
    sublattice_A = Offset(
        rep=ImmutableDenseMatrix([0, 0, 0]),
        space=fcc_lattice.affine
    )

    print("\nStep 3: Define Degrees of Freedom (Orbital & Spin)")
    x, y, z = sy.symbols('x y z', real=True)
    # s-orbital
    s_orbital = PointGroupBasis(
        expr=sy.Integer(1),
        axes=(x, y, z),
        order=0,
        rep=ImmutableDenseMatrix([1])
    )
    
    # Spin states
    spin_up = FrozenDict({'label': 'up', 'sz': sy.Rational(1, 2)})
    spin_down = FrozenDict({'label': 'down', 'sz': sy.Rational(-1, 2)})

    print("\nStep 4: Build Hilbert Space")
    modes = []
    # Only one sublattice now
    for spin_name, spin_data in [('up', spin_up), ('down', spin_down)]:
        mode = Mode(
            count=1,
            attr=FrozenDict({
                'sublattice': 'A',
                'position': sublattice_A,
                'orbital': s_orbital,
                'spin': spin_data['label'],
                'sz': spin_data['sz']
            })
        )
        modes.append(mode)

    H_space = hilbert(modes)
    print(f"Hilbert space dimension: {H_space.dim}") # Should be 2

    print("\nStep 5: Get Reciprocal Lattice and Brillouin Zone")
    reciprocal_lattice = fcc_lattice.dual
    BZ = brillouin_zone(reciprocal_lattice)
    print(f"Reciprocal lattice basis:\n{reciprocal_lattice.basis}")

    return fcc_lattice, reciprocal_lattice, H_space, BZ

def build_hamiltonian(BZ, H_space, params):
    """
    Constructs the Hamiltonian for the 1-atom FCC Weyl semimetal model.
    """
    t = params.get('t', 1.0)
    m = params.get('m', 2.5) # Parameter to control Weyl point phase
    
    # Pauli matrices for spin
    sigma_0 = torch.eye(2, dtype=torch.complex128)
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    H_data = torch.zeros((BZ.dim, H_space.dim, H_space.dim), dtype=torch.complex128)
    
    # Placeholder for full BZ Hamiltonian construction if needed
    pass

def compute_band_structure_path(reciprocal_lattice, H_space, params):
    """
    Computes band structure along a high-symmetry path.
    """
    # High symmetry points for FCC (BCC reciprocal)
    high_sym_points = {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'X': np.array([0.0, 0.5, 0.0]),
        'L': np.array([0.5, 0.5, 0.5]),
        'W': np.array([0.25, 0.75, 0.5]),
        'K': np.array([0.375, 0.75, 0.375]),
        'U': np.array([0.25, 0.625, 0.625])
    }
    
    path_labels = ['Gamma', 'K', 'W', 'K', 'Gamma', 'L', 'U', 'W', 'L', 'K']
    num_points_per_segment = 30
    
    k_path = []
    k_distances = [0]
    current_distance = 0
    segment_boundaries = [0]
    
    for i in range(len(path_labels) - 1):
        start_pt = high_sym_points[path_labels[i]]
        end_pt = high_sym_points[path_labels[i+1]]
        
        for j in range(num_points_per_segment):
            t = j / num_points_per_segment
            k_point = (1 - t) * start_pt + t * end_pt
            k_path.append(k_point)
            
            if len(k_path) > 1:
                dk = np.linalg.norm(k_point - k_path[-2])
                current_distance += dk
                k_distances.append(current_distance)
        segment_boundaries.append(len(k_path))
        
    k_path.append(high_sym_points[path_labels[-1]])
    if len(k_path) > 1:
        current_distance += np.linalg.norm(k_path[-1] - k_path[-2])
    k_distances.append(current_distance)
    
    k_path = np.array(k_path)
    
    # Construct Hamiltonian and diagonalize
    eigenvalues = []
    
    t_val = params.get('t', 1.0)
    m = params.get('m', 2.0)
    
    # Pauli matrices for spin (2x2)
    # H_space dim is 2 (Spin Up, Spin Down)
    sigma_0 = torch.eye(2, dtype=torch.complex128)
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    
    for k_vec in k_path:
        # k_vec are fractional coordinates in reciprocal basis (b1, b2, b3)
        # k = kx*b1 + ky*b2 + kz*b3
        # In this basis, terms like sin(2*pi*kx) correspond to hoppings in real space a1 direction
        kx, ky, kz = k_vec
        
        # Model: H = t * [ sin(kx)sx + sin(ky)sy + (cos(kz) + cos(kx) + cos(ky) - m)sz ]
        # This is a standard 2-band Weyl model (often used on cubic, here mapped to FCC basis)
        # We assume nearest neighbor hoppings in a1, a2, a3 directions.
        
        arg_x = torch.tensor(2*np.pi*kx)
        arg_y = torch.tensor(2*np.pi*ky)
        arg_z = torch.tensor(2*np.pi*kz)
        
        hx = t_val * torch.sin(arg_x)
        hy = t_val * torch.sin(arg_y)
        hz = t_val * (torch.cos(arg_z) + torch.cos(arg_x) + torch.cos(arg_y) - m)
        
        H_k = hx * sigma_x + hy * sigma_y + hz * sigma_z
               
        evals, _ = torch.linalg.eigh(H_k)
        eigenvalues.append(evals.numpy())
        
    return np.array(k_distances), np.array(eigenvalues), path_labels, segment_boundaries

if __name__ == "__main__":
    lattice, recip_lattice, H_space, BZ = create_fcc_weyl_model()
    
    # Parameters for Weyl points
    # For the model H = sin(x)sx + sin(y)sy + (cos(z)+cos(x)+cos(y)-m)sz
    # Gap closes when sin(x)=sin(y)=0 => x,y in {0, pi}
    # and term_z = 0.
    # Try m = 2.0. Then if x=0, y=0, term_z = 1+1+1-2 = 1 != 0
    # If x=0, y=pi? term_z = cos(z) + 1 - 1 - 2 = cos(z) - 2 != 0
    # Let's adjust m to be around 2.5 or 1.5 to find crossings.
    # If m=2.0: x=0, y=pi/2 (sin!=0).
    # Need sin(x)=0, sin(y)=0.
    # Points: (0,0,z), (pi,0,z), (0,pi,z), (pi,pi,z)
    # (0,0,z): Mz = 1 + 1 + cos(z) - m = 2 + cos(z) - m. 
    # If m=2.5, 2+cos(z)-2.5 = cos(z)-0.5 = 0 => cos(z)=0.5 => z=pi/3.
    # So m=2.5 should give Weyl points.
    
    params = {'t': 1.0, 'm': 2.5}
    print(f"\nComputing band structure with parameters: {params}")
    
    k_dist, evals, labels, boundaries = compute_band_structure_path(recip_lattice, H_space, params)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(H_space.dim):
        plt.plot(k_dist, evals[:, i], label=f'Band {i+1}')
        
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    # Vertical lines for symmetry points
    for b_idx in boundaries:
        if b_idx < len(k_dist):
            plt.axvline(k_dist[b_idx], color='gray', alpha=0.3)
    
    # Label symmetry points
    # Need to map boundary indices to labels
    # b_idx corresponds to path_labels[i] (start of segment)
    # boundaries has len(path_labels) entries? No, len(path_labels) points in path_labels
    # segments = len(path_labels)-1. boundaries has start indices of segments + end.
    
    xticks = [k_dist[i] for i in boundaries]
    # path_labels has one more than segments
    # boundaries has indices for 0, seg1_end, seg2_end...
    # Actually my boundary logic in function:
    # boundaries = [0]
    # ... append(len) ...
    # So boundaries has N+1 entries for N segments.
    # path_labels has N+1 entries.
    
    if len(xticks) == len(labels):
        plt.xticks(xticks, labels)
    
    plt.ylabel("Energy (eV)")
    plt.title("Band Structure of 1-Atom FCC Weyl Semimetal Model")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Band structure plotted.")
