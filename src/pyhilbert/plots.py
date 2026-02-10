import torch
import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.figure_factory as ff  # type: ignore[import-untyped]
from typing import Optional, List, Union, Dict
from .abstracts import Plottable
from .spatials import Lattice
from .tensors import Tensor
from .utils import compute_bonds
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

# --- Registered Plot Methods ---


@Plottable.register_plot_method("structure", backend="plotly")
def plot_structure(
    obj: Lattice,
    subs: Optional[Dict] = None,
    spin_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    plot_type: str = "edge-and-node",
    show: bool = True,
    fig: Optional[go.Figure] = None,
    **kwargs,
) -> go.Figure:
    """
    Visualize the lattice structure using Plotly.

    This function creates an interactive 3D or 2D plot of the lattice structure,
    including sites, bonds, and optional spin vectors.

    Parameters
    ----------
    obj : Lattice
        The lattice instance to visualize.
    subs : dict, optional
        Dictionary of symbol substitutions for lattice parameters.
    spin_data : array-like, optional
        (N_sites, 3) array of spin vectors.
    plot_type : {'edge-and-node', 'scatter'}, default 'edge-and-node'
        Visualization style.
    show : bool, default True
        If True, calls `fig.show()` to display the plot immediately.
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add traces to.
    **kwargs
        Additional keyword arguments passed to `go.Figure`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    valid_types = ["edge-and-node", "scatter"]
    if plot_type not in valid_types:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Options: {valid_types}")

    # Use method on Lattice object
    coords = obj.coords(subs)
    coords_np = coords.numpy()

    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2] if obj.dim == 3 else None

    if fig is None:
        fig = go.Figure()

    # Bonds (Only for 'edge-and-node')
    if plot_type == "edge-and-node":
        x_lines, y_lines, z_lines = compute_bonds(coords, obj.dim)
        if x_lines:
            if obj.dim == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_lines,
                        y=y_lines,
                        z=z_lines,
                        mode="lines",
                        line=dict(color="black", width=1),
                        name="Bonds",
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_lines,
                        y=y_lines,
                        mode="lines",
                        line=dict(color="black", width=2),
                        name="Bonds",
                        showlegend=False,
                    )
                )

    # Sites
    num_basis = len(obj.unit_cell) if obj.unit_cell else 1
    num_cells = coords.shape[0] // num_basis

    basis_colors = ["blue", "red", "green", "orange", "purple"]
    colors = []
    for _ in range(num_cells):
        for b in range(num_basis):
            colors.append(basis_colors[b % len(basis_colors)])

    if obj.dim == 3:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=5, color=colors),
                name="Sites",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=10, color=colors, symbol="circle"),
                name="Sites",
            )
        )

    # Spins (Optional)
    if spin_data is not None:
        if isinstance(spin_data, np.ndarray):
            spin_data = torch.from_numpy(spin_data)

        if spin_data.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Spin data shape {spin_data.shape} does not match sites {coords.shape[0]}. "
                f"Did you forget to provide spin data for all basis atoms?"
            )

        spin_np = spin_data.numpy()

        if obj.dim == 3:
            fig.add_trace(
                go.Cone(
                    x=x,
                    y=y,
                    z=z,
                    u=spin_np[:, 0],
                    v=spin_np[:, 1],
                    w=spin_np[:, 2],
                    sizemode="absolute",
                    sizeref=0.5,
                    anchor="tail",
                    colorscale="Viridis",
                    name="Spins",
                )
            )
        else:
            quiver = ff.create_quiver(
                x,
                y,
                spin_np[:, 0],
                spin_np[:, 1],
                scale=0.2,
                arrow_scale=0.3,
                name="Spins",
                line=dict(color="red"),
            )
            fig.add_traces(quiver.data)

    # Layout
    if obj.dim == 3:
        fig.update_layout(title="3D Lattice System", scene=dict(aspectmode="data"))
    else:
        fig.update_layout(
            title="2D Lattice System", yaxis=dict(scaleanchor="x", scaleratio=1)
        )

    if show:
        fig.show()
    return fig


@Plottable.register_plot_method("heatmap", backend="plotly")
def plot_heatmap(
    obj: Tensor,
    title: str = "Matrix Visualization",
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """
    Plot a heatmap of a matrix using Plotly.

    Handles complex matrices by showing Real and Imaginary parts side-by-side
    with a shared symmetric color scale.

    Parameters
    ----------
    obj : Tensor
        2D Tensor to visualize.
    title : str, default "Matrix Visualization"
        Title of the plot.
    show : bool, default True
        Whether to show the plot immediately.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure.
    """
    tensor = obj.data.detach().cpu()

    if tensor.ndim != 2:
        raise ValueError(
            f"Heatmap requires a 2D Tensor, got shape {tuple(tensor.shape)}"
        )

    is_complex = tensor.is_complex()

    if is_complex:
        real_part = tensor.real
        imag_part = tensor.imag

        # Calculate global range using torch
        limit = max(torch.abs(real_part).max(), torch.abs(imag_part).max()).item()

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Real Part", "Imaginary Part")
        )

        # Convert to numpy only for Plotly
        fig.add_trace(
            go.Heatmap(
                z=real_part.numpy(),
                colorscale="RdBu",
                zmin=-limit,
                zmax=limit,
                showscale=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=imag_part.numpy(),
                colorscale="RdBu",
                zmin=-limit,
                zmax=limit,
                showscale=True,
            ),
            row=1,
            col=2,
        )
    else:
        # Real matrix
        limit = torch.abs(tensor).max().item()
        fig = go.Figure(
            data=go.Heatmap(
                z=tensor.numpy(),
                colorscale="RdBu",
                zmin=-limit,
                zmax=limit,
                showscale=True,
            )
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)

    if show:
        fig.show()
    return fig


@Plottable.register_plot_method("spectrum", backend="plotly")
def plot_spectrum(
    obj: Union[np.ndarray, torch.Tensor, object],
    title: str = "Spectrum Visualization",
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """
    Plot the eigenvalue spectrum using Plotly.

    Parameters
    ----------
    obj : array-like or Tensor
        2D matrix to analyze.
    title : str, default "Spectrum Visualization"
        Title of the plot.
    show : bool, default True
        Whether to show the plot immediately.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure.
    """
    # 1. Standardize to PyTorch Tensor
    if hasattr(obj, "data") and isinstance(obj.data, torch.Tensor):
        tensor = obj.data.detach().cpu()
    elif isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
    else:
        tensor = torch.from_numpy(np.array(obj))

    if tensor.ndim != 2:
        raise ValueError(f"Spectrum requires a 2D matrix, got shape {tensor.shape}")

    # 2. Check for Hermiticity (M == M.H)
    is_complex = tensor.is_complex()
    is_hermitian = False

    # Calculate Frobenius norm once
    norm = torch.norm(tensor)
    if norm > 0:
        if is_complex:
            diff = torch.norm(tensor - tensor.conj().T)
        else:
            diff = torch.norm(tensor - tensor.T)

        if diff / norm < 1e-5:
            is_hermitian = True
    else:
        is_hermitian = True  # Zero matrix is Hermitian

    fig = go.Figure()

    # 3. Calculate and Plot
    if is_hermitian:
        # Returns real values sorted in ascending order
        evals = torch.linalg.eigvalsh(tensor)
        y_vals = evals.numpy()
        x_vals = np.arange(len(y_vals))

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+lines",
                marker=dict(size=6, color="blue"),
                name="Eigenvalues",
            )
        )
        fig.update_layout(xaxis_title="Index", yaxis_title="Eigenvalue")
    else:
        # General case (Complex plane)
        evals = torch.linalg.eigvals(tensor)
        real_parts = evals.real.numpy()
        imag_parts = evals.imag.numpy()

        fig.add_trace(
            go.Scatter(
                x=real_parts,
                y=imag_parts,
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Eigenvalues",
            )
        )
        fig.update_layout(xaxis_title="Real Part", yaxis_title="Imaginary Part")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(title=title)

    if show:
        fig.show()
    return fig


@Plottable.register_plot_method("bandstructure", backend="plotly")
def plot_bandstructure(
    obj: Union[np.ndarray, torch.Tensor, object],
    k_distances: Optional[Union[np.ndarray, torch.Tensor]] = None,
    k_node_indices: Optional[List[int]] = None,
    k_node_labels: Optional[List[str]] = None,
    title: str = "Band Structure",
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """
    Plot electronic band structure using Plotly.

    Parameters
    ----------
    obj : array-like or Tensor
        (N_k, N_bands) array of energy eigenvalues.
    k_distances : array-like, optional
        (N_k,) array of cumulative distances.
    k_node_indices : list of int, optional
        Indices of high-symmetry points.
    k_node_labels : list of str, optional
        Labels for high-symmetry points.
    title : str, default "Band Structure"
        Title of the plot.
    show : bool, default True
        Whether to show the plot immediately.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure.
    """
    # Standardize inputs
    if hasattr(obj, "data") and isinstance(obj.data, torch.Tensor):
        energies = obj.data.detach().cpu().numpy()
    elif isinstance(obj, torch.Tensor):
        energies = obj.detach().cpu().numpy()
    else:
        energies = np.array(obj)

    if isinstance(k_distances, torch.Tensor):
        k_distances = k_distances.detach().cpu().numpy()

    if energies.ndim != 2:
        raise ValueError(f"Energies must be 2D (N_k, N_bands), got {energies.shape}")

    num_k, num_bands = energies.shape

    if k_distances is None:
        k_distances = np.arange(num_k)

    fig = go.Figure()

    # Plot bands
    for b in range(num_bands):
        fig.add_trace(
            go.Scatter(
                x=k_distances,
                y=energies[:, b],
                mode="lines",
                line=dict(color="black", width=1.5),
                name=f"Band {b}",
                showlegend=False,
            )
        )

    # Vertical lines and ticks
    if k_node_indices:
        tick_vals = []
        tick_text = []
        labels = k_node_labels if k_node_labels else [str(i) for i in k_node_indices]

        for idx, label in zip(k_node_indices, labels):
            if 0 <= idx < len(k_distances):
                x_val = k_distances[idx]
                tick_vals.append(x_val)
                tick_text.append(label)

                # Vertical line
                fig.add_vline(
                    x=x_val, line_width=1, line_dash="dash", line_color="grey"
                )

        if tick_vals:
            fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text)

    fig.update_layout(
        title=title,
        xaxis_title="Wave Vector",
        yaxis_title="Energy",
        template="simple_white",
    )

    if show:
        fig.show()
    return fig
