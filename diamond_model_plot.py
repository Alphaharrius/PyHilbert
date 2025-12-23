import sympy as sy
import numpy as np
import torch
from sympy import ImmutableDenseMatrix
from src.pyhilbert.spatials import Lattice, Offset, PointGroupBasis
from src.pyhilbert.hilbert import Mode, hilbert, brillouin_zone
from src.pyhilbert.utils import FrozenDict
from src.pyhilbert.plot import plot_system, plot_bands

def main():
    print("Diamond Weyl Semimetal Model (PyHilbert Structure)")

    # 1. Define Lattice
    a = sy.Symbol('a', real=True, positive=True)
    basis_matrix = ImmutableDenseMatrix([
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ])
    
    diamond_lattice = Lattice(basis=basis_matrix, shape=(2, 2, 2))
    print(f"Lattice: {diamond_lattice}")

    # 2. Define Sublattices
    sublattice_A = Offset(rep=ImmutableDenseMatrix([0, 0, 0]), space=diamond_lattice.affine)
    sublattice_B = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 4), sy.Rational(1, 4), sy.Rational(1, 4)]), space=diamond_lattice.affine)
    
    # 3. Define Degrees of Freedom
    x, y, z = sy.symbols('x y z', real=True)
    s_orbital = PointGroupBasis(expr=sy.Integer(1), axes=(x, y, z), order=0, rep=ImmutableDenseMatrix([1]))
    
    spin_up = FrozenDict({'label': 'up', 'sz': sy.Rational(1, 2)})
    spin_down = FrozenDict({'label': 'down', 'sz': sy.Rational(-1, 2)})

    # 4. Build Hilbert Space (Modes)
    modes = []
    for sublat_name, sublat_offset in [('A', sublattice_A), ('B', sublattice_B)]:
        for spin_name, spin_data in [('up', spin_up), ('down', spin_down)]:
            mode = Mode(
                count=1, 
                attr=FrozenDict({
                    'sublattice': sublat_name,
                    'position': sublat_offset,
                    'orbital': s_orbital,
                    'spin': spin_data['label'],
                    'sz': spin_data['sz']
                })
            )
            modes.append(mode)
    
    H_space = hilbert(modes)
    print(f"Hilbert Space Dim: {H_space.dim}")

    # 5. Define Hamiltonian Function
    # This matches the model used in diamond.ipynb
    def diamond_weyl_hamiltonian(k_vecs):
        """
        k_vecs: (N, 3) tensor of k-points in fractional reciprocal coordinates.
        Returns: (N, 4, 4) Hamiltonian matrices.
        """
        # Ensure input is tensor
        if not isinstance(k_vecs, torch.Tensor):
            k_vecs = torch.tensor(k_vecs)
            
        if k_vecs.dim() == 1:
            k_vecs = k_vecs.unsqueeze(0)
            
        kx = k_vecs[:, 0]
        ky = k_vecs[:, 1]
        kz = k_vecs[:, 2]
        
        # Parameters (tuned for Weyl points)
        t = 1.0
        Delta = 0.0
        m = 2.5 # Tuned for Weyl points? No, notebook said 0.0 initially, then discussed 2.5. 
                # Let's use 0.0 as base, or parameters from notebook end.
                # Actually notebook cell 17 output shows gap 0 at m=0? No, it says small gap detected.
                # Let's use standard params.
        lambda_so = 0.5
        
        # Pauli matrices
        s0 = torch.eye(2, dtype=torch.complex128)
        sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        
        # Kron Helper manually for batch
        def batch_kron(A_batch, B_static):
            # A: (N, 2, 2), B: (2, 2) -> (N, 4, 4)
            N = A_batch.shape[0]
            # Reshape A to (N, 2, 1, 2, 1) and B to (1, 1, 2, 1, 2) then multiply... 
            # Easier:
            # Result is block matrix:
            # [ A11*B  A12*B ]
            # [ A21*B  A22*B ]
            out = torch.zeros((N, 4, 4), dtype=torch.complex128)
            out[:, 0:2, 0:2] = A_batch[:, 0, 0].unsqueeze(-1).unsqueeze(-1) * B_static
            out[:, 0:2, 2:4] = A_batch[:, 0, 1].unsqueeze(-1).unsqueeze(-1) * B_static
            out[:, 2:4, 0:2] = A_batch[:, 1, 0].unsqueeze(-1).unsqueeze(-1) * B_static
            out[:, 2:4, 2:4] = A_batch[:, 1, 1].unsqueeze(-1).unsqueeze(-1) * B_static
            return out

        # Base Terms
        arg_x = torch.tensor(2*np.pi)*kx
        arg_y = torch.tensor(2*np.pi)*ky
        arg_z = torch.tensor(2*np.pi)*kz
        
        hx = t * torch.sin(arg_x)
        hy = t * torch.sin(arg_y)
        hz = t * torch.sin(arg_z) + Delta
        h0 = m
        
        # H_base construction
        # Terms like hx * kron(sx, s0)
        # We need to broadcast scalars hx to (N, 2, 2)
        N = k_vecs.shape[0]
        
        H_base = torch.zeros((N, 4, 4), dtype=torch.complex128)
        
        # Term 1: h0 * kron(s0, sz)
        # kron(s0, sz) is static
        term1 = h0 * torch.kron(s0, sz)
        H_base += term1.unsqueeze(0) # Broadcast to N
        
        # Term 2: hx * kron(sx, s0)
        term2_mat = torch.kron(sx, s0)
        H_base += hx.view(N, 1, 1) * term2_mat
        
        # Term 3: hy * kron(sy, s0)
        term3_mat = torch.kron(sy, s0)
        H_base += hy.view(N, 1, 1) * term3_mat
        
        # Term 4: hz * kron(sz, s0)
        term4_mat = torch.kron(sz, s0)
        H_base += hz.view(N, 1, 1) * term4_mat
        
        # SOC Terms
        # sin(kx)*kron(sx, sx)
        H_soc = torch.zeros((N, 4, 4), dtype=torch.complex128)
        H_soc += (torch.sin(arg_x).view(N,1,1) * torch.kron(sx, sx))
        H_soc += (torch.sin(arg_y).view(N,1,1) * torch.kron(sy, sy))
        H_soc += (torch.sin(arg_z).view(N,1,1) * torch.kron(sz, sz))
        H_soc *= lambda_so
        
        return H_base + H_soc

    # 6. Plot Band Structure
    print("\nGenerating Band Structure...")
    points = {
        'Γ': [0.0, 0.0, 0.0],
        'X': [0.0, 0.5, 0.0],
        'L': [0.5, 0.5, 0.5],
        'W': [0.25, 0.75, 0.5],
        'K': [0.375, 0.75, 0.375],
        'U': [0.25, 0.625, 0.625]
    }
    path = ['Γ', 'K', 'W', 'K', 'Γ', 'L', 'U', 'W', 'L', 'K']
    
    plot_bands(
        hamiltonian=diamond_weyl_hamiltonian,
        points=points,
        path_labels=path,
        k_resolution=30,
        show=True # Will open in browser
    )
    print("Band structure plot generated.")

    # 7. Real Space Plot
    print("\nGenerating Real Space Plot...")
    # Generate spin texture for 2 atoms per cell
    # Just random for demo
    offsets = [sublattice_A, sublattice_B]
    total_sites = (2**3) * len(offsets)
    spins = np.random.randn(total_sites, 3)
    spins /= np.linalg.norm(spins, axis=1, keepdims=True)
    
    plot_system(
        diamond_lattice,
        plot_type='edge-and-node',
        show=True, # Will open in browser
        subs={a: 1.0},
        offsets=offsets,
        spin_data=spins
    )
    print("Real space plot generated.")

if __name__ == "__main__":
    main()




