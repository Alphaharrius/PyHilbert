import torch
import numpy as np
import sympy as sy

from pyhilbert.spatials import Lattice
from pyhilbert.tensors import Tensor
from pyhilbert.hilbert import hilbert, Mode
from pyhilbert.utils import FrozenDict, generate_k_path


def create_dummy_tensor(data_np):
    """Creates a valid pyhilbert.tensors.Tensor from numpy data."""
    if isinstance(data_np, np.ndarray):
        data = torch.from_numpy(data_np)
    else:
        data = data_np

    m = Mode(count=data.shape[0], attr=FrozenDict({"label": "dummy"}))
    hspace = hilbert([m])

    if data.ndim == 2:
        dims = (hspace, hspace)
    else:
        dims = (hspace,) * data.ndim

    return Tensor(data=data, dims=dims)


def test_plots():
    """
    Smoke test for plotting capabilities.
    Ensures all plot methods run without errors (headless).
    """
    print("=== Running Plotting Smoke Test ===")

    # Disable showing plots for tests
    SHOW_PLOTS = False

    # --- Heatmap Demo ---
    mat = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    tensor_obj = create_dummy_tensor(mat)

    # Plotly Backend
    fig = tensor_obj.plot("heatmap", title="Random Complex Matrix", show=SHOW_PLOTS)
    assert fig is not None

    # --- Heatmap Realistic Demo ---
    dim = 20
    M = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = M + M.conj().T
    tensor_h = create_dummy_tensor(H)

    tensor_h.plot("heatmap", title="Hermitian Matrix", show=SHOW_PLOTS)

    # --- Structure 2D Demo ---
    a = sy.Symbol("a")
    basis = sy.ImmutableDenseMatrix([[a, 0], [0, a]])
    lattice = Lattice(basis=basis, shape=(3, 3))

    lattice.plot("structure", subs={a: 1.5}, show=SHOW_PLOTS)

    # --- Structure 3D Demo ---
    basis_3d = sy.ImmutableDenseMatrix([[a, 0, 0], [0, a, 0], [0, 0, a]])
    lattice_3d = Lattice(basis=basis_3d, shape=(2, 2, 2))

    lattice_3d.plot("structure", subs={a: 1.0}, show=SHOW_PLOTS)

    # --- Spectrum Demo ---
    # Hermitian
    tensor_h.plot("spectrum", title="Hermitian Spectrum", show=SHOW_PLOTS)

    # Non-Hermitian
    tensor_m = create_dummy_tensor(M)
    tensor_m.plot("spectrum", title="Non-Hermitian Spectrum", show=SHOW_PLOTS)

    # --- Band Structure Demo ---
    def tight_binding_2d(k_vecs):
        kx = k_vecs[:, 0]
        ky = k_vecs[:, 1]
        t = 1.0
        E = -2.0 * t * (torch.cos(kx) + torch.cos(ky))
        return E.unsqueeze(1)

    points = {"G": [0, 0], "X": [np.pi, 0], "M": [np.pi, np.pi]}
    path = ["G", "X", "M", "G"]
    k_vecs, k_dists, nodes = generate_k_path(points, path, resolution=50)
    energies = tight_binding_2d(k_vecs)
    energies_tensor = create_dummy_tensor(energies)

    energies_tensor.plot(
        "bandstructure",
        k_distances=k_dists,
        k_node_indices=nodes,
        k_node_labels=path,
        title="Square Lattice Bands",
        show=SHOW_PLOTS,
    )

    print("\n--- Testing Matplotlib Backend (No Save) ---")

    # MPL Heatmap
    tensor_h.plot("heatmap", backend="matplotlib", title="Hermitian Matrix (MPL)")

    # MPL Structure 2D
    lattice.plot("structure", backend="matplotlib", subs={a: 1.5})

    # MPL Structure 3D
    lattice_3d.plot("structure", backend="matplotlib", subs={a: 1.0}, elev=20, azim=45)

    # MPL Spectrum
    tensor_m.plot(
        "spectrum", backend="matplotlib", title="Non-Hermitian Spectrum (MPL)"
    )

    # MPL Band Structure
    energies_tensor.plot(
        "bandstructure",
        backend="matplotlib",
        k_distances=k_dists,
        k_node_indices=nodes,
        k_node_labels=path,
        title="Square Lattice Bands (MPL)",
    )

    print("=== Plotting Smoke Test Passed ===")


if __name__ == "__main__":
    # Allow running directly with python tests/test_plots.py
    test_plots()
