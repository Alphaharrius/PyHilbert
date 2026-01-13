import torch
import numpy as np
import sympy as sy
import os

from pyhilbert.spatials import Lattice
from pyhilbert.tensors import Tensor
from pyhilbert.hilbert import hilbert, Mode
from pyhilbert.utils import FrozenDict, generate_k_path
# Plotting backends are now registered automatically via pyhilbert import in __init__.py

def create_dummy_tensor(data_np):
    """Creates a valid pyhilbert.tensors.Tensor from numpy data."""
    if isinstance(data_np, np.ndarray):
        data = torch.from_numpy(data_np)
    else:
        data = data_np
    
    # Create dummy dimensions to satisfy dataclass
    # We just need any valid StateSpace
    m = Mode(count=data.shape[0], attr=FrozenDict({'label': 'dummy'}))
    hspace = hilbert([m])
    
    # Assuming square matrix or similar for simplicity in demo
    if data.ndim == 2:
        dims = (hspace, hspace)
    else:
        dims = (hspace,) * data.ndim
        
    return Tensor(data=data, dims=dims)

def run_demo():
    print("=== Running Plotting Demo (Object-Oriented API) ===")
    
    # Ensure img directory exists
    img_dir = './tests/img'
    os.makedirs(img_dir, exist_ok=True)
    print(f"   Created directory '{img_dir}' for output images.")
    
    # --- Heatmap Demo ---
    print("\n1. Heatmap (Complex Random Matrix)")
    mat = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    tensor_obj = create_dummy_tensor(mat)
    
    # Using .plot() method
    fig = tensor_obj.plot('heatmap', title="Random Complex Matrix", show=True)
    print("   Generated Figure:", fig.layout.title.text)

    # --- Heatmap Realistic Demo ---
    print("\n1b. Heatmap (Hermitian Matrix)")
    dim = 20
    M = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = M + M.conj().T
    tensor_h = create_dummy_tensor(H)
    
    fig_herm = tensor_h.plot('heatmap', title="Hermitian Matrix (Real=Sym, Imag=AntiSym)", show=True)
    print("   Generated Hermitian Matrix Heatmap")

    # --- Structure 2D Demo ---
    print("\n2. Structure (2D Square Lattice)")
    a = sy.Symbol('a')
    basis = sy.ImmutableDenseMatrix([[a, 0], [0, a]])
    lattice = Lattice(basis=basis, shape=(3, 3))
    
    # Using .plot() method on Lattice
    fig = lattice.plot('structure', subs={a: 1.5}, show=True)
    print("   Generated 2D Structure Plot with traces:", [t.name for t in fig.data])

    # --- Structure 3D Demo ---
    print("\n3. Structure (3D Cubic Lattice)")
    basis_3d = sy.ImmutableDenseMatrix([[a, 0, 0], [0, a, 0], [0, 0, a]])
    lattice_3d = Lattice(basis=basis_3d, shape=(2, 2, 2))
    
    fig_3d = lattice_3d.plot('structure', subs={a: 1.0}, show=True)
    print("   Generated 3D Structure Plot with traces:", [t.name for t in fig_3d.data])

    # --- Spectrum Demo ---
    print("\n4. Spectrum")
    
    # Case A: Hermitian
    print("   4a. Hermitian Spectrum (Sorted Real Eigenvalues)")
    # tensor_h created above
    fig_spec_h = tensor_h.plot('spectrum', title="Hermitian Spectrum", show=True)
    
    # Case B: Non-Hermitian
    print("   4b. Non-Hermitian Spectrum (Complex Plane)")
    tensor_m = create_dummy_tensor(M)
    fig_spec_nh = tensor_m.plot('spectrum', title="Non-Hermitian Spectrum", show=True)

    # --- Band Structure Demo ---
    print("\n5. Band Structure (2D Square Lattice Tight Binding)")
    
    # 1. Define Hamiltonian Function
    def tight_binding_2d(k_vecs):
        kx = k_vecs[:, 0]
        ky = k_vecs[:, 1]
        t = 1.0
        E = -2.0 * t * (torch.cos(kx) + torch.cos(ky))
        return E.unsqueeze(1) # (N, 1) Band

    # 2. Define Path
    points = {
        'G': [0, 0],
        'X': [np.pi, 0],
        'M': [np.pi, np.pi]
    }
    path = ['G', 'X', 'M', 'G']
    
    # 3. Generate Path Vectors (Helper function still needed)
    k_vecs, k_dists, nodes = generate_k_path(points, path, resolution=50)
    
    # 4. Calculate Energies
    energies = tight_binding_2d(k_vecs)
    
    # Wrap energies in a Tensor object to use .plot()
    # Energies shape is (N_k, N_bands)
    energies_tensor = create_dummy_tensor(energies)
    
    # 5. Plot
    fig_bands = energies_tensor.plot('bandstructure', 
                                     k_distances=k_dists, 
                                     k_node_indices=nodes, 
                                     k_node_labels=path,
                                     title="Square Lattice Bands (G-X-M-G)", 
                                     show=True)
    print("   Generated Square Lattice Band Structure Plot")

    print("\n--- Testing Matplotlib Backend (Saving to ./img/) ---")
    
    # MPL Heatmap
    print("\n6. MPL Heatmap")
    tensor_h.plot('heatmap', backend='matplotlib', title="Hermitian Matrix (MPL)", 
                  save_path=os.path.join(img_dir, 'heatmap_hermitian.png'))
    print("   Saved heatmap_hermitian.png")
    
    # MPL Structure 2D
    print("\n7. MPL Structure 2D")
    lattice.plot('structure', backend='matplotlib', subs={a: 1.5},
                 save_path=os.path.join(img_dir, 'structure_2d.png'))
    print("   Saved structure_2d.png")
    
    # MPL Structure 3D (with custom angles)
    print("\n8. MPL Structure 3D")
    lattice_3d.plot('structure', backend='matplotlib', subs={a: 1.0}, elev=20, azim=45,
                    save_path=os.path.join(img_dir, 'structure_3d.png'))
    print("   Saved structure_3d.png")
    
    # MPL Spectrum
    print("\n9. MPL Spectrum")
    tensor_m.plot('spectrum', backend='matplotlib', title="Non-Hermitian Spectrum (MPL)",
                  save_path=os.path.join(img_dir, 'spectrum_complex.png'))
    print("   Saved spectrum_complex.png")
    
    # MPL Band Structure
    print("\n10. MPL Band Structure")
    energies_tensor.plot('bandstructure', backend='matplotlib',
                         k_distances=k_dists, 
                         k_node_indices=nodes, 
                         k_node_labels=path,
                         title="Square Lattice Bands (MPL)",
                         save_path=os.path.join(img_dir, 'bandstructure.svg'))
    print("   Saved bandstructure.svg")

    print(f"\nAll Matplotlib plots saved to {os.path.abspath(img_dir)}")
    print("\n=== Demo Completed ===")

if __name__ == "__main__":
    run_demo()
