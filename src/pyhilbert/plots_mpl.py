import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict, Any, cast, Tuple
from collections import OrderedDict
from .spatials import Lattice, ReciprocalLattice
from .tensors import Tensor
from .utils import compute_bonds
from .hilbert import HilbertSpace, generate_k_path
from .fourier import fourier_transform

# --- Registered Plot Methods (Matplotlib Backend) ---


@Lattice.register_plot_method("structure", backend="matplotlib")
def plot_structure_mpl(
    obj: Lattice,
    subs: Optional[Dict] = None,
    spin_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    plot_type: str = "edge-and-node",
    elev: float = 30,
    azim: float = -60,
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
    **kwargs,
) -> plt.Figure:
    """
    Visualize the lattice structure (sites, bonds, spins) using Matplotlib.

    Parameters
    ----------
    obj : Lattice
        The lattice instance to plot.
    subs : dict, optional
        Dictionary of symbolic substitutions for lattice parameters (e.g., {'a': 1.0}).
    spin_data : array-like, optional
        (N_sites, 3) array containing spin vectors for each site.
    plot_type : {'edge-and-node', 'scatter'}, default 'edge-and-node'
        Visualization style. 'edge-and-node' draws bonds between nearest neighbors;
        'scatter' draws only the sites.
    elev : float, default 30
        Elevation angle (in degrees) for 3D plots.
    azim : float, default -60
        Azimuth angle (in degrees) for 3D plots.
    save_path : str, optional
        If provided, saves the figure to this path. File format is inferred from the extension.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    **kwargs
        Additional keyword arguments passed to `plt.figure` (e.g., `figsize`).

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    valid_types = ["edge-and-node", "scatter"]
    if plot_type not in valid_types:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Options: {valid_types}")

    coords = obj.coords(subs)
    coords_np = coords.numpy()

    x = coords_np[:, 0]
    y = coords_np[:, 1]

    is_3d = obj.dim == 3
    z = coords_np[:, 2] if is_3d else None

    if ax is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title("3D Lattice System")
        else:
            ax = fig.add_subplot(111)
            ax.set_title("2D Lattice System")
            ax.set_aspect("equal")
    else:
        fig = ax.get_figure()

    # Bonds
    if plot_type == "edge-and-node":
        x_lines, y_lines, z_lines = compute_bonds(coords, obj.dim)
        if x_lines:
            # Remove None values for matplotlib plot if needed, but plot handles nan/None usually by breaking line
            # Matplotlib handles np.nan as break. None might cause issues.
            # Convert None to np.nan in lists
            x_l = [val if val is not None else np.nan for val in x_lines]
            y_l = [val if val is not None else np.nan for val in y_lines]

            if is_3d and z_lines is not None:
                z_l = [val if val is not None else np.nan for val in z_lines]
                ax.plot(x_l, y_l, z_l, color="black", linewidth=1, label="Bonds")
            else:
                ax.plot(x_l, y_l, color="black", linewidth=1.5, label="Bonds")

    # Sites
    num_basis = len(obj.unit_cell) if obj.unit_cell else 1
    num_cells = coords.shape[0] // num_basis

    # Basis colors
    basis_colors = ["blue", "red", "green", "orange", "purple"]
    colors = []
    for _ in range(num_cells):
        for b in range(num_basis):
            colors.append(basis_colors[b % len(basis_colors)])

    if is_3d:
        cast(Any, ax).scatter(x, y, z, c=colors, s=20, label="Sites")
    else:
        ax.scatter(x, y, c=colors, s=50, zorder=5, label="Sites")

    # Spins
    if spin_data is not None:
        if isinstance(spin_data, np.ndarray):
            spin_data = torch.from_numpy(spin_data)

        if spin_data.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Spin data shape {spin_data.shape} does not match sites {coords.shape[0]}"
            )

        spin_np = spin_data.numpy()
        u = spin_np[:, 0]
        v = spin_np[:, 1]

        if is_3d:
            w = spin_np[:, 2]
            # Quiver in 3D: x, y, z, u, v, w
            ax.quiver(
                x, y, z, u, v, w, length=0.5, normalize=True, color="red", label="Spins"
            )
        else:
            # Quiver in 2D
            ax.quiver(x, y, u, v, color="red", scale=20, width=0.005, label="Spins")

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if is_3d:
        cast(Any, ax).set_zlabel("Z")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@Tensor.register_plot_method("heatmap", backend="matplotlib")
def plot_heatmap_mpl(
    obj: Tensor,
    title: str = "Matrix Visualization",
    save_path: Optional[str] = None,
    fixed_indices: Optional[Tuple[int, ...]] = None,
    axes: Tuple[int, int] = (-2, -1),
    **kwargs,
) -> plt.Figure:
    """
    Plot a heatmap of a matrix using Matplotlib.

    If the matrix is complex, displays Real and Imaginary parts in side-by-side subplots.

    Parameters
    ----------
    obj : Tensor
        Tensor to visualize as a 2D heatmap.
    title : str, default "Matrix Visualization"
        Title of the figure.
    save_path : str, optional
        If provided, saves the figure to this path.
    fixed_indices : tuple of int, optional
        Indices used to fix non-heatmap dimensions. For an N-dimensional tensor,
        this must provide N-2 indices after selecting `axes`.
    axes : tuple of int, default (-2, -1)
        Pair of dimensions used as (row_axis, col_axis) in the heatmap.
    **kwargs
        Additional keyword arguments passed to `plt.subplots` (e.g., `figsize`).

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """

    tensor = obj.data.detach().cpu()
    rank = tensor.ndim
    if rank < 2:
        raise ValueError(
            f"Heatmap requires rank >= 2 tensor, got shape {tuple(tensor.shape)}"
        )

    if len(axes) != 2:
        raise ValueError(f"`axes` must have length 2, got {axes}")

    normalized_axes = []
    for ax in axes:
        ax_norm = ax + rank if ax < 0 else ax
        if not (0 <= ax_norm < rank):
            raise ValueError(f"Axis {ax} is out of bounds for tensor with rank {rank}")
        normalized_axes.append(ax_norm)
    row_axis, col_axis = normalized_axes
    if row_axis == col_axis:
        raise ValueError(f"`axes` must reference two different dimensions, got {axes}")

    permute_order = [i for i in range(rank) if i not in (row_axis, col_axis)] + [
        row_axis,
        col_axis,
    ]
    tensor = tensor.permute(*permute_order)

    expected_fixed = rank - 2
    if fixed_indices is None:
        if expected_fixed == 0:
            fixed_indices = ()
        else:
            raise ValueError(
                f"Heatmap for shape {tuple(obj.data.shape)} with axes={axes} requires "
                f"`fixed_indices` of length {expected_fixed}."
            )
    elif len(fixed_indices) != expected_fixed:
        raise ValueError(
            f"`fixed_indices` length must be {expected_fixed} for shape "
            f"{tuple(obj.data.shape)} with axes={axes}, got {len(fixed_indices)}."
        )

    try:
        tensor = tensor[(*fixed_indices, slice(None), slice(None))]
    except IndexError as exc:
        raise IndexError(
            f"`fixed_indices` {fixed_indices} is out of bounds for shape "
            f"{tuple(obj.data.shape)} with axes={axes}."
        ) from exc

    is_complex = tensor.is_complex()

    if is_complex:
        real_part = tensor.real.numpy()
        imag_part = tensor.imag.numpy()

        limit = max(np.abs(real_part).max(), np.abs(imag_part).max())

        fig, subplot_axes = plt.subplots(1, 2, figsize=kwargs.get("figsize", (12, 5)))

        im1 = subplot_axes[0].imshow(real_part, cmap="RdBu", vmin=-limit, vmax=limit)
        subplot_axes[0].set_title("Real Part")
        fig.colorbar(im1, ax=subplot_axes[0])

        im2 = subplot_axes[1].imshow(imag_part, cmap="RdBu", vmin=-limit, vmax=limit)
        subplot_axes[1].set_title("Imaginary Part")
        fig.colorbar(im2, ax=subplot_axes[1])

        fig.suptitle(title)
    else:
        data = tensor.numpy()
        limit = np.abs(data).max()

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 5)))
        im = ax.imshow(data, cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@Tensor.register_plot_method("spectrum", backend="matplotlib")
def plot_spectrum_mpl(
    obj: Tensor,
    title: str = "Spectrum Visualization",
    save_path: Optional[str] = None,
    fixed_indices: Optional[Tuple[int, ...]] = None,
    axes: Tuple[int, int] = (-2, -1),
    **kwargs,
) -> plt.Figure:
    """
    Plot the eigenvalue spectrum of a matrix using Matplotlib.

    - If Hermitian/Symmetric: Plots sorted real eigenvalues.
    - If Non-Hermitian: Plots eigenvalues in the complex plane.

    Parameters
    ----------
    obj : Tensor
        Matrix/tensor to analyze as a 2D operator.
    title : str, default "Spectrum Visualization"
        Title of the figure.
    save_path : str, optional
        If provided, saves the figure to this path.
    fixed_indices : tuple of int, optional
        Indices used to fix non-matrix dimensions. For an N-dimensional tensor,
        this must provide N-2 indices after selecting `axes`.
    axes : tuple of int, default (-2, -1)
        Pair of dimensions used as (row_axis, col_axis) for spectrum analysis.
    **kwargs
        Additional keyword arguments passed to `plt.subplots`.

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    tensor = obj.data.detach().cpu()

    rank = tensor.ndim
    if rank < 2:
        raise ValueError(
            f"Spectrum requires rank >= 2 tensor, got shape {tuple(tensor.shape)}"
        )

    if len(axes) != 2:
        raise ValueError(f"`axes` must have length 2, got {axes}")

    normalized_axes = []
    for ax in axes:
        ax_norm = ax + rank if ax < 0 else ax
        if not (0 <= ax_norm < rank):
            raise ValueError(f"Axis {ax} is out of bounds for tensor with rank {rank}")
        normalized_axes.append(ax_norm)
    row_axis, col_axis = normalized_axes
    if row_axis == col_axis:
        raise ValueError(f"`axes` must reference two different dimensions, got {axes}")

    permute_order = [i for i in range(rank) if i not in (row_axis, col_axis)] + [
        row_axis,
        col_axis,
    ]
    tensor = tensor.permute(*permute_order)

    expected_fixed = rank - 2
    if fixed_indices is None:
        if expected_fixed == 0:
            fixed_indices = ()
        else:
            raise ValueError(
                f"Spectrum for shape {tuple(tensor.shape)} with axes={axes} requires "
                f"`fixed_indices` of length {expected_fixed}."
            )
    elif len(fixed_indices) != expected_fixed:
        raise ValueError(
            f"`fixed_indices` length must be {expected_fixed} for shape "
            f"{tuple(tensor.shape)} with axes={axes}, got {len(fixed_indices)}."
        )

    try:
        tensor = tensor[(*fixed_indices, slice(None), slice(None))]
    except IndexError as exc:
        raise IndexError(
            f"`fixed_indices` {fixed_indices} is out of bounds for shape "
            f"{tuple(tensor.shape)} with axes={axes}."
        ) from exc

    if tensor.shape[-2] != tensor.shape[-1]:
        raise ValueError(
            f"Spectrum requires a square matrix after slicing, got shape {tuple(tensor.shape)}"
        )

    # Check Hermiticity
    is_complex = tensor.is_complex()
    is_hermitian = False
    norm = torch.norm(tensor)
    if norm > 0:
        if is_complex:
            diff = torch.norm(tensor - tensor.conj().T)
        else:
            diff = torch.norm(tensor - tensor.T)
        if diff / norm < 1e-5:
            is_hermitian = True
    else:
        is_hermitian = True

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 5)))

    if is_hermitian:
        evals = torch.linalg.eigvalsh(tensor).numpy()
        ax.plot(evals, "b.-", markersize=10)
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")
        ax.grid(True)
    else:
        evals = torch.linalg.eigvals(tensor).numpy()
        ax.scatter(evals.real, evals.imag, c="r", s=30)
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.grid(True)
        ax.set_aspect("equal")

    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@Lattice.register_plot_method("bandstructure", backend="matplotlib")
@ReciprocalLattice.register_plot_method("bandstructure", backend="matplotlib")
def plot_bandstructure_mpl(
    obj: Union[Lattice, ReciprocalLattice],
    hamiltonian: Tensor,
    k_path: List[Union[List[float], Tuple[float, ...]]],
    n_points: int = 100,
    title: str = "Band Structure",
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot the band structure of a Hamiltonian along a k-path using Matplotlib.

    Parameters
    ----------
    obj : Union[Lattice, ReciprocalLattice]
        The lattice object (or its reciprocal).
    hamiltonian : Tensor
        The real-space Hamiltonian tensor. Must have rank >= 2.
    k_path : List[List[float]]
        List of fractional coordinates for high-symmetry points defining the path.
    n_points : int
        Approximate number of points along the path.
    title : str
        Plot title.
    save_path : str, optional
        If provided, saves the figure to this path.
    **kwargs
        Additional arguments passed to plt.subplots.
    """
    # 1. Resolve ReciprocalLattice
    if isinstance(obj, Lattice):
        recip = obj.dual
    else:
        recip = obj

    # 2. Generate k-path
    k_space, x_vals, tick_vals = generate_k_path(recip, k_path, n_points)

    # 3. Identify Bloch Space from Hamiltonian
    region_space = hamiltonian.dims[-1]
    if not isinstance(region_space, HilbertSpace):
        raise ValueError("Hamiltonian last dimension must be HilbertSpace")

    unique_modes = {}
    for mode in region_space:
        frac_offset = mode["r"].fractional()
        bloch_mode = mode.update(r=frac_offset)

        if bloch_mode not in unique_modes:
            unique_modes[bloch_mode] = mode.count

    sorted_bloch_modes = sorted(unique_modes.keys(), key=lambda m: tuple(m["r"].rep))

    bloch_structure = OrderedDict()
    base = 0
    for m in sorted_bloch_modes:
        c = unique_modes[m]
        bloch_structure[m] = slice(base, base + c)
        base += c

    bloch_space = HilbertSpace(structure=bloch_structure)

    # 4. Fourier Transform
    f = fourier_transform(k_space, bloch_space, region_space)
    f_dag = f.h(-2, -1)

    # H(k) = f @ hamiltonian @ f_dag
    h_k = f @ hamiltonian @ f_dag

    # 5. Diagonalize
    hk_data = h_k.data

    # Assume Hermitian for now (usual for band structure)
    eigvals = torch.linalg.eigvalsh(hk_data)  # (K, N_bands)
    eigvals_np = eigvals.detach().cpu().numpy()

    # 6. Plot
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

    n_bands = eigvals_np.shape[1]

    for b in range(n_bands):
        ax.plot(x_vals, eigvals_np[:, b], "b-", linewidth=1.5)

    # High-symmetry lines
    for tick in tick_vals:
        ax.axvline(x=tick, color="gray", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Wave Vector")
    ax.set_ylabel("Energy")
    ax.set_xticks(tick_vals)
    # We can't set labels easily as they are not passed
    ax.set_xticklabels([])

    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig
