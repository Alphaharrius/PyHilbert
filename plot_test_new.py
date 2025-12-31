import sympy as sy
import numpy as np
import torch
from sympy import ImmutableDenseMatrix
from collections import OrderedDict
from pyhilbert.spatials import Lattice, Offset
from pyhilbert.tensors import Tensor
from pyhilbert.hilbert import StateSpace

def main():
    print("Initializing Diamond Lattice visualization test (New Architecture)...")

    # 1. Define lattice constant
    a = sy.Symbol('a', real=True, positive=True)

    # 2. FCC lattice vectors
    basis_matrix = ImmutableDenseMatrix([
        [0, a/2, a/2],      # a1
        [a/2, 0, a/2],      # a2
        [a/2, a/2, 0]       # a3
    ])

    # 3. Create real space lattice with N=2
    N = 2 
    diamond_lattice = Lattice(basis=basis_matrix, shape=(N, N, N))
    
    print(f"Lattice created: {diamond_lattice}")

    # 4. Define Sublattices
    sublattice_A = Offset(
        rep=ImmutableDenseMatrix([0, 0, 0]),
        space=diamond_lattice.affine
    )

    sublattice_B = Offset(
        rep=ImmutableDenseMatrix([sy.Rational(1, 4), sy.Rational(1, 4), sy.Rational(1, 4)]),
        space=diamond_lattice.affine
    )
    
    offsets = [sublattice_A, sublattice_B]
    subs = {a: 2.0}

    # --- Test 1: Plot Structure (using .plot on Lattice) ---
    print("\n[1/5] Plotting Crystal Structure...")
    try:
        # Calculate total sites for spins
        total_sites = (N**3) * len(offsets)
        spins = np.random.randn(total_sites, 3)
        spins = spins / np.linalg.norm(spins, axis=1, keepdims=True)
        
        # New API Call
        diamond_lattice.plot('structure', 
                           offsets=offsets, 
                           subs=subs, 
                           spin_data=spins,
                           plot_type='scatter',
                           show=True)
        print("Success: Structure plot generated.")
    except Exception as e:
        print(f"Error plotting structure: {e}")
        import traceback
        traceback.print_exc()

    # --- Test 2: Plot Wavefunction (using .plot on Tensor) ---
    print("\n[2/5] Plotting Wavefunction...")
    psi_tensor = None # Initialize variable for later use in try block
    try:
        # Need coordinates to generate dummy wavefunction
        from src.pyhilbert.plot import compute_coordinates
        coords = compute_coordinates(diamond_lattice, offsets, subs).numpy()
        
        # Gaussian wavepacket
        center = coords[0]
        dist = np.linalg.norm(coords - center, axis=1)
        phase = np.exp(1j * np.dot(coords, np.array([1.0, 1.0, 1.0])))
        psi_data = np.exp(-dist**2 / 2.0) * phase
        
        # Create a dummy Tensor holding this data
        # Correctly initializing StateSpace with an empty OrderedDict
        psi_tensor = Tensor(data=torch.from_numpy(psi_data), dims=(StateSpace(structure=OrderedDict()),))
        
        # New API Call
        psi_tensor.plot('wavefunction', 
                        lattice=diamond_lattice, 
                        offsets=offsets, 
                        subs=subs,
                        title="New Wavefunction Plot",
                        show=True)
        print("Success: Wavefunction plot generated.")
    except Exception as e:
        print(f"Error plotting wavefunction: {e}")
        import traceback
        traceback.print_exc()

    # --- Hamiltonian Model ---
    def weyl_model(k_vecs):
        if k_vecs.dim() == 1:
            k_vecs = k_vecs.unsqueeze(0)
        kx, ky, kz = k_vecs[:, 0], k_vecs[:, 1], k_vecs[:, 2]
        m = 2.0
        s0 = torch.eye(2, dtype=torch.complex128)
        sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        
        tx1 = torch.sin(kx); ty1 = torch.sin(ky); tz1 = torch.cos(kz) - m
        N_k = k_vecs.shape[0]
        H = torch.zeros((N_k, 4, 4), dtype=torch.complex128)
        
        H1 = (tx1.view(N_k, 1, 1) * sx + ty1.view(N_k, 1, 1) * sy + tz1.view(N_k, 1, 1) * sz)
        H2 = torch.cos(kx).view(N_k, 1, 1) * s0
        H[:, 0:2, 0:2] = H1; H[:, 2:4, 2:4] = H2
        return H

    # --- Test 3: Plot Bands (using .plot on Plottable/Tensor) ---
    print("\n[3/5] Plotting Bands...")
    try:
        points = {'Γ': [0,0,0], 'X': [np.pi,0,0], 'W': [np.pi,np.pi/2,0], 'L': [np.pi,np.pi,np.pi]}
        path = ['Γ', 'X', 'W', 'L', 'Γ']
        
        # We can call plot on ANY Plottable object, or create a dummy one
        # Here we reuse psi_tensor just as a carrier if it exists, otherwise use lattice
        carrier = psi_tensor if psi_tensor is not None else diamond_lattice
        
        carrier.plot('bands', 
                        hamiltonian=weyl_model,
                        points=points,
                        path_labels=path,
                        show=True)
        print("Success: Band plot generated.")
    except Exception as e:
        print(f"Error plotting bands: {e}")
        import traceback
        traceback.print_exc()

    # --- Test 4: Plot Band Surface ---
    print("\n[4/5] Plotting Band Surface...")
    try:
        def plane_model(k_vecs_2d):
            zeros = torch.zeros(k_vecs_2d.shape[0], 1, dtype=k_vecs_2d.dtype)
            k_3d = torch.cat([k_vecs_2d, zeros], dim=1)
            return weyl_model(k_3d)

        carrier = psi_tensor if psi_tensor is not None else diamond_lattice
        carrier.plot('band_surface', 
                        hamiltonian=plane_model, 
                        resolution=20, 
                        show=True)
        print("Success: Surface plot generated.")
    except Exception as e:
        print(f"Error plotting surface: {e}")
        import traceback
        traceback.print_exc()

    # --- Test 5: Plot Heatmap & Spectrum ---
    print("\n[5/5] Plotting Matrix Heatmap & Spectrum...")
    try:
        H_rand = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
        H_rand = H_rand + H_rand.conj().T
        H_tensor = Tensor(data=torch.from_numpy(H_rand), dims=(StateSpace(structure=OrderedDict()),))
        
        # Heatmap
        H_tensor.plot('heatmap', title="New Matrix Plot", show=True)
        
        # Spectrum
        # Note: If H_tensor contains the matrix, 'spectrum' plotter should ideally calculate eigenvalues
        # But our current implementation expects eigenvalues directly or in .data
        # Let's compute them first
        evals = np.linalg.eigvalsh(H_rand)
        evals_tensor = Tensor(data=torch.from_numpy(evals), dims=(StateSpace(structure=OrderedDict()),))
        evals_tensor.plot('spectrum', title="New Spectrum Plot", show=True)
        
        print("Success: Matrix/Spectrum plots generated.")
    except Exception as e:
        print(f"Error plotting matrix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
