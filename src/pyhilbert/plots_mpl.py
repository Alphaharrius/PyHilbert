import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict
from .abstracts import Plottable
from .spatials import Lattice, Offset
from .utils import compute_bonds

# --- Registered Plot Methods (Matplotlib Backend) ---


@Plottable.register_plot_method("structure", backend="matplotlib")
def plot_structure_mpl(
    obj: Lattice,
    subs: Optional[Dict] = None,
    basis_offsets: Optional[List[Offset]] = None,
    spin_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    plot_type: str = "edge-and-node",
    elev: float = 30,
    azim: float = -60,
    save_path: Optional[str] = None,
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
    basis_offsets : list of Offset, optional
        List of offsets defining basis atoms within the unit cell.
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

    coords = obj.compute_coords(basis_offsets, subs)
    coords_np = coords.numpy()

    x = coords_np[:, 0]
    y = coords_np[:, 1]

    is_3d = obj.dim == 3
    z = coords_np[:, 2] if is_3d else None

    fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))

    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("3D Lattice System")
    else:
        ax = fig.add_subplot(111)
        ax.set_title("2D Lattice System")
        ax.set_aspect("equal")

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
    num_basis = len(basis_offsets) if basis_offsets else 1
    num_cells = coords.shape[0] // num_basis

    # Basis colors
    basis_colors = ["blue", "red", "green", "orange", "purple"]
    colors = []
    for _ in range(num_cells):
        for b in range(num_basis):
            colors.append(basis_colors[b % len(basis_colors)])

    if is_3d:
        ax.scatter(x, y, z, c=colors, s=20, label="Sites")
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
        ax.set_zlabel("Z")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


@Plottable.register_plot_method("heatmap", backend="matplotlib")
def plot_heatmap_mpl(
    obj: Union[np.ndarray, torch.Tensor, object],
    title: str = "Matrix Visualization",
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot a heatmap of a matrix using Matplotlib.

    If the matrix is complex, displays Real and Imaginary parts in side-by-side subplots.

    Parameters
    ----------
    obj : array-like or Tensor
        2D matrix to visualize.
    title : str, default "Matrix Visualization"
        Title of the figure.
    save_path : str, optional
        If provided, saves the figure to this path.
    **kwargs
        Additional keyword arguments passed to `plt.subplots` (e.g., `figsize`).

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    # Standardize input
    if hasattr(obj, "data") and isinstance(obj.data, torch.Tensor):
        tensor = obj.data.detach().cpu()
    elif isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
    else:
        tensor = torch.from_numpy(np.array(obj))

    if tensor.ndim != 2:
        raise ValueError(f"Heatmap requires a 2D matrix, got shape {tensor.shape}")

    is_complex = tensor.is_complex()

    if is_complex:
        real_part = tensor.real.numpy()
        imag_part = tensor.imag.numpy()

        limit = max(np.abs(real_part).max(), np.abs(imag_part).max())

        fig, axes = plt.subplots(1, 2, figsize=kwargs.get("figsize", (12, 5)))

        im1 = axes[0].imshow(real_part, cmap="RdBu", vmin=-limit, vmax=limit)
        axes[0].set_title("Real Part")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(imag_part, cmap="RdBu", vmin=-limit, vmax=limit)
        axes[1].set_title("Imaginary Part")
        fig.colorbar(im2, ax=axes[1])

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


@Plottable.register_plot_method("spectrum", backend="matplotlib")
def plot_spectrum_mpl(
    obj: Union[np.ndarray, torch.Tensor, object],
    title: str = "Spectrum Visualization",
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot the eigenvalue spectrum of a matrix using Matplotlib.

    - If Hermitian/Symmetric: Plots sorted real eigenvalues.
    - If Non-Hermitian: Plots eigenvalues in the complex plane.

    Parameters
    ----------
    obj : array-like or Tensor
        2D matrix to analyze.
    title : str, default "Spectrum Visualization"
        Title of the figure.
    save_path : str, optional
        If provided, saves the figure to this path.
    **kwargs
        Additional keyword arguments passed to `plt.subplots`.

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    # Standardize
    if hasattr(obj, "data") and isinstance(obj.data, torch.Tensor):
        tensor = obj.data.detach().cpu()
    elif isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
    else:
        tensor = torch.from_numpy(np.array(obj))

    if tensor.ndim != 2:
        raise ValueError(f"Spectrum requires a 2D matrix, got shape {tensor.shape}")

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


@Plottable.register_plot_method("bandstructure", backend="matplotlib")
def plot_bandstructure_mpl(
    obj: Union[np.ndarray, torch.Tensor, object],
    k_distances: Optional[Union[np.ndarray, torch.Tensor]] = None,
    k_node_indices: Optional[List[int]] = None,
    k_node_labels: Optional[List[str]] = None,
    title: str = "Band Structure",
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot electronic band structure using Matplotlib.

    Parameters
    ----------
    obj : array-like or Tensor
        (N_k, N_bands) array containing energy eigenvalues for each k-point.
    k_distances : array-like, optional
        (N_k,) array of cumulative distances along the k-path.
        If None, defaults to indices.
    k_node_indices : list of int, optional
        Indices of high-symmetry points (ticks) in `k_distances`.
    k_node_labels : list of str, optional
        Labels for the high-symmetry points (e.g., ['G', 'M', 'K']).
    title : str, default "Band Structure"
        Title of the plot.
    save_path : str, optional
        If provided, saves the figure to this path.
    **kwargs
        Additional keyword arguments passed to `plt.subplots`.

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    # Standardize
    if hasattr(obj, "data") and isinstance(obj.data, torch.Tensor):
        energies = obj.data.detach().cpu().numpy()
    elif isinstance(obj, torch.Tensor):
        energies = obj.detach().cpu().numpy()
    else:
        energies = np.array(obj)

    if isinstance(k_distances, torch.Tensor):
        k_distances = k_distances.detach().cpu().numpy()

    if energies.ndim != 2:
        raise ValueError(f"Energies must be 2D, got {energies.shape}")

    num_k, num_bands = energies.shape

    if k_distances is None:
        k_distances = np.arange(num_k)

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

    # Plot bands
    ax.plot(k_distances, energies, "k-", linewidth=1.5)

    # Vertical lines
    if k_node_indices:
        tick_vals = []
        tick_text = []
        labels = k_node_labels if k_node_labels else [str(i) for i in k_node_indices]

        for idx, label in zip(k_node_indices, labels):
            if 0 <= idx < len(k_distances):
                x_val = k_distances[idx]
                tick_vals.append(x_val)
                tick_text.append(label)
                ax.axvline(x=x_val, color="grey", linestyle="--", linewidth=1)

        if tick_vals:
            ax.set_xticks(tick_vals)
            ax.set_xticklabels(tick_text)

    ax.set_title(title)
    ax.set_xlabel("Wave Vector")
    ax.set_ylabel("Energy")
    ax.set_xlim(k_distances[0], k_distances[-1])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig
