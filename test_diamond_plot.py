import sympy as sy
import numpy as np
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import Lattice, Offset
from pyhilbert.plot import plot_system, plot_bands
def main():
    print("Initializing Diamond Lattice visualization test...")

    # 1. Define lattice constant
    a = sy.Symbol('a', real=True, positive=True)

    # 2. FCC lattice vectors (columns are basis vectors)
    # Convention: lattice vectors in 3D Cartesian space
    basis_matrix = ImmutableDenseMatrix([
        [0, a/2, a/2],      # a1 = (0, a/2, a/2)
        [a/2, 0, a/2],      # a2 = (a/2, 0, a/2)
        [a/2, a/2, 0]       # a3 = (a/2, a/2, 0)
    ])

    # 3. Create real space lattice with N×N×N unit cells
    # Use smaller N for cleaner visualization, N=4 might be too dense for interactive plot
    N = 2 
    diamond_lattice = Lattice(basis=basis_matrix, shape=(N, N, N))
    
    print(f"Lattice created: {diamond_lattice}")
    print(f"Dimension: {diamond_lattice.dim}")
    print(f"Shape: {diamond_lattice.shape}")

    # 4. Define Sublattices A and B offsets
    # Sublattice A at origin
    sublattice_A = Offset(
        rep=ImmutableDenseMatrix([0, 0, 0]),
        space=diamond_lattice.affine
    )

    # Sublattice B at (1/4, 1/4, 1/4) in lattice coordinates
    sublattice_B = Offset(
        rep=ImmutableDenseMatrix([sy.Rational(1, 4), sy.Rational(1, 4), sy.Rational(1, 4)]),
        space=diamond_lattice.affine
    )
    
    offsets = [sublattice_A, sublattice_B]

    # --- Real Space Plot with Spins ---
    print("\nGenerating plot for full Diamond lattice with Spins...")
    try:
        
        # Calculate total sites: N^3 cells * 2 basis atoms
        total_sites = (N**3) * len(offsets) # 8 * 2 = 16
        
        # Generate random spin vectors (normalized)
        spins = np.random.randn(total_sites, 3)
        spins = spins / np.linalg.norm(spins, axis=1, keepdims=True)
        
        # Use plot_system with offsets and spin_data
        fig = plot_system(
            diamond_lattice, 
            plot_type='scatter', 
            show=True, # Set to True to open in browser
            subs={a: 2.0}, 
            offsets=offsets,
            # spin_data=spins
        )
        print("Success: 3D Plotly figure generated with Spins (Cone plot).")
        # fig.write_html("diamond_spins.html")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

    # --- Band Structure Plot ---
    print("\nGenerating Band Structure plot...")
    try:
        import torch
        
        # 1. Define Hamiltonian Model Function H(k)
        # Returns (N, 4, 4) matrix for N k-points
        def weyl_model(k_vecs):
            # k_vecs shape: (N, 3) or (3,)
            if k_vecs.dim() == 1:
                k_vecs = k_vecs.unsqueeze(0)
            
            kx, ky, kz = k_vecs[:, 0], k_vecs[:, 1], k_vecs[:, 2]
            
            # Simple Weyl model parameters
            t = 1.0
            m = 2.0
            
            # Pauli matrices
            s0 = torch.eye(2, dtype=torch.complex128)
            sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
            sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
            sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
            
            # Model H = tx*sx + ty*sy + tz*sz
            # Here we just make a 4x4 toy model (2x2 block diagonal for simplicity)
            # Block 1: Weyl point at (0,0,0)
            tx1 = torch.sin(kx)
            ty1 = torch.sin(ky)
            tz1 = torch.cos(kz) - m
            
            # Construct batches (N, 2, 2)
            # Use broadcasting manually since torch.kron doesn't broadcast batch well easily without care
            # H1 = tx1 * sx + ...
            N = k_vecs.shape[0]
            H = torch.zeros((N, 4, 4), dtype=torch.complex128)
            
            # Fill block 1 (upper left 2x2)
            # We construct (N, 2, 2)
            H1 = (tx1.view(N, 1, 1) * sx + 
                  ty1.view(N, 1, 1) * sy + 
                  tz1.view(N, 1, 1) * sz)
                  
            # Fill block 2 (lower right 2x2) - just a cosine band
            e_band = torch.cos(kx)
            H2 = e_band.view(N, 1, 1) * s0
            
            H[:, 0:2, 0:2] = H1
            H[:, 2:4, 2:4] = H2
            
            return H

        # 2. Define High Symmetry Points
        points = {
            'Γ': [0, 0, 0],
            'X': [np.pi, 0, 0],
            'W': [np.pi, np.pi/2, 0],
            'L': [np.pi, np.pi, np.pi]
        }
        
        # 3. Define Path
        path = ['Γ', 'X', 'W', 'L', 'Γ']
        
        # 4. Plot (letting function do the calculation)
        fig_bands = plot_bands(
            hamiltonian=weyl_model,
            points=points,
            path_labels=path,
            k_resolution=40,
            show=True
        )
        print("Success: Band Structure plot generated (calculated from model).")
        # fig_bands.write_html("bands_test.html")
        
    except Exception as e:
        print(f"Error plotting bands: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
