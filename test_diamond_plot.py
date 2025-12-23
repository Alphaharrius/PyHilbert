import sympy as sy
import numpy as np
import torch
from sympy import ImmutableDenseMatrix
from pyhilbert.spatials import Lattice, Offset
from pyhilbert.plot import LatticePlotter

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
    print("\n[1/5] Generating plot for full Diamond lattice with Spins...")
    try:
        # Create Plotter Instance ONCE
        plotter = LatticePlotter(diamond_lattice, subs={a: 2.0}, offsets=offsets)

        # Calculate total sites: N^3 cells * 2 basis atoms
        total_sites = (N**3) * len(offsets) # 8 * 2 = 16
        
        # Generate random spin vectors (normalized)
        spins = np.random.randn(total_sites, 3)
        spins = spins / np.linalg.norm(spins, axis=1, keepdims=True)
        
        # Use plotter.plot_system
        fig = plotter.plot_system(
            plot_type='edge-and-node', 
            show=True, 
            spin_data=spins
        )
        print("Success: 3D Plotly figure generated with Spins (Cone plot).")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

    # --- Wavefunction Bubble Plot ---
    print("\n[2/5] Generating Wavefunction Bubble Plot...")
    try:
        # Reusing plotter instance
        coords = plotter.coords.numpy()
        
        # Gaussian wavepacket centered at first atom
        center = coords[0]
        sigma = 1.0
        dist = np.linalg.norm(coords - center, axis=1)
        
        # Add some phase variation: e^(i * k * r)
        k_vec = np.array([1.0, 1.0, 1.0])
        phase = np.exp(1j * np.dot(coords, k_vec))
        
        psi = np.exp(-dist**2 / (2*sigma**2)) * phase
        
        fig_wf = plotter.plot_wavefunction(psi, scale=2.0, show=True, title="Localized Wavepacket")
        print("Success: Wavefunction plot generated.")
        
    except Exception as e:
        print(f"Error plotting wavefunction: {e}")
        import traceback
        traceback.print_exc()

    # --- Hamiltonian Model Definition ---
    # Returns (N, 4, 4) matrix for N k-points
    def weyl_model(k_vecs):
        # ... (same model code) ...
        # k_vecs shape: (N, 3) or (3,)
        if k_vecs.dim() == 1:
            k_vecs = k_vecs.unsqueeze(0)
        
        kx, ky, kz = k_vecs[:, 0], k_vecs[:, 1], k_vecs[:, 2]
        
        # Simple Weyl model parameters
        m = 2.0
        
        # Pauli matrices
        s0 = torch.eye(2, dtype=torch.complex128)
        sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        
        # Model H = tx*sx + ty*sy + tz*sz
        tx1 = torch.sin(kx)
        ty1 = torch.sin(ky)
        tz1 = torch.cos(kz) - m
        
        # Construct batches (N, 2, 2)
        N_k = k_vecs.shape[0]
        H = torch.zeros((N_k, 4, 4), dtype=torch.complex128)
        
        # Fill block 1 (upper left 2x2)
        H1 = (tx1.view(N_k, 1, 1) * sx + 
              ty1.view(N_k, 1, 1) * sy + 
              tz1.view(N_k, 1, 1) * sz)
              
        # Fill block 2 (lower right 2x2) - just a cosine band
        e_band = torch.cos(kx)
        H2 = e_band.view(N_k, 1, 1) * s0
        
        H[:, 0:2, 0:2] = H1
        H[:, 2:4, 2:4] = H2
        
        return H

    # --- Band Structure Path Plot ---
    print("\n[3/5] Generating Band Structure Path plot...")
    try:
        # Define High Symmetry Points
        points = {
            'Γ': [0, 0, 0],
            'X': [np.pi, 0, 0],
            'W': [np.pi, np.pi/2, 0],
            'L': [np.pi, np.pi, np.pi]
        }
        
        # Define Path
        path = ['Γ', 'X', 'W', 'L', 'Γ']
        
        # Using plotter to call plot_bands (even though bands don't strictly require lattice if H provided)
        # But conceptually it fits "Plotter for this system"
        fig_bands = plotter.plot_bands(
            hamiltonian=weyl_model,
            points=points,
            path_labels=path,
            k_resolution=40,
            show=True
        )
        print("Success: Band Structure Path plot generated.")
        
    except Exception as e:
        print(f"Error plotting bands: {e}")
        import traceback
        traceback.print_exc()

    # --- 2D Band Surface Plot ---
    print("\n[4/5] Generating 2D Band Surface plot...")
    try:
        # We'll just plot kx-ky plane at kz=0
        def plane_model(k_vecs_2d):
            # k_vecs_2d is (N, 2)
            # Pad with kz=0
            zeros = torch.zeros(k_vecs_2d.shape[0], 1, dtype=k_vecs_2d.dtype)
            k_3d = torch.cat([k_vecs_2d, zeros], dim=1)
            return weyl_model(k_3d)

        fig_surf = plotter.plot_band_surface(
            hamiltonian=plane_model,
            k_range_x=(-np.pi, np.pi),
            k_range_y=(-np.pi, np.pi),
            resolution=30, # Low res for speed
            show=True
        )
        print("Success: Band Surface plot generated.")

    except Exception as e:
        print(f"Error plotting surface: {e}")
        traceback.print_exc()

    # --- Matrix & Spectrum Plot ---
    print("\n[5/5] Generating Matrix & Spectrum plots...")
    try:
        # Generate a random Hermitian matrix
        size = 20
        H_rand = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        H_rand = H_rand + H_rand.conj().T
        
        # Plot Matrix Heatmap
        fig_mat = LatticePlotter.plot_matrix(H_rand, title="Random Hermitian Matrix", show=True)
        print("Success: Matrix Heatmap generated.")

        # Plot Spectrum
        evals = np.linalg.eigvalsh(H_rand)
        fig_spec = LatticePlotter.plot_spectrum(evals, title="Eigenvalue Spectrum", show=True)
        print("Success: Spectrum plot generated.")

    except Exception as e:
        print(f"Error plotting matrix/spectrum: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
