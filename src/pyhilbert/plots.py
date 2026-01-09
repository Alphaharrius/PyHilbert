import torch
import numpy as np
import sympy as sy
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Optional, List, Union, Dict, Callable
from .abstracts import Plottable
from .spatials import Lattice, cartes, Offset
from plotly.subplots import make_subplots
# --- Helper Functions ---

def _compute_all_coords(lattice: Lattice, basis_offsets: Optional[List[Offset]] = None, subs: Optional[Dict] = None) -> torch.Tensor:
    """
    Vectorized calculation of all site coordinates.
    Avoids running SymPy substitution inside the loop.
    """
    basis_sym = lattice.basis
    if subs:
        basis_eval = basis_sym.subs(subs)
    else:
        basis_eval = basis_sym.subs({s: 1.0 for s in basis_sym.free_symbols})
    
    try:
        basis_mat = torch.tensor(np.array(basis_eval).astype(np.float64), dtype=torch.float64)
    except Exception as e:
         raise ValueError(f"Basis matrix contains unresolved symbols: {basis_eval.free_symbols}") from e

    lat_offsets = cartes(lattice) 
    
    lat_reps = []
    for off in lat_offsets:
        lat_reps.append(np.array(off.rep).flatten().astype(np.float64))
    
    if not lat_reps:
        return torch.empty((0, lattice.dim))

    lat_tensor = torch.tensor(np.array(lat_reps), dtype=torch.float64) # (N_cells, Dim)

    if basis_offsets is None:
         basis_offsets = [Offset(rep=sy.ImmutableDenseMatrix([0]*lattice.dim), space=lattice.affine)]
    
    basis_reps = []
    for off in basis_offsets:
        rep = off.rep
        if subs:
             rep = rep.subs(subs)
        basis_reps.append(np.array(rep).flatten().astype(np.float64))
        
    basis_tensor = torch.tensor(np.array(basis_reps), dtype=torch.float64) # (N_basis, Dim)

    total_crystal = lat_tensor.unsqueeze(1) + basis_tensor.unsqueeze(0)
    
    total_crystal_flat = total_crystal.view(-1, lattice.dim)
    
    coords = total_crystal_flat @ basis_mat.T
    
    return coords

def _generate_bonds_trace(coords: torch.Tensor, dim: int) -> Optional[Union[go.Scatter, go.Scatter3d]]:
    """Generate bond lines connecting nearest neighbors using PyTorch."""
    if coords.size(0) < 2:
        return None
        
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dists = torch.norm(diff, dim=-1)
    
    dists.fill_diagonal_(float('inf'))
    
    min_dist = torch.min(dists)
    if torch.isinf(min_dist):
        return None
        
    tol = 1e-4
    pairs = torch.nonzero(dists <= min_dist + tol)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    
    if pairs.size(0) == 0:
        return None

    p1 = coords[pairs[:, 0]]
    p2 = coords[pairs[:, 1]]
    
    p1_np = p1.numpy()
    p2_np = p2.numpy()
    
    x_lines = []
    y_lines = []
    z_lines = []
    nan = None
    
    for i in range(len(p1_np)):
        x_lines.extend([p1_np[i, 0], p2_np[i, 0], nan])
        y_lines.extend([p1_np[i, 1], p2_np[i, 1], nan])
        if dim == 3:
            z_lines.extend([p1_np[i, 2], p2_np[i, 2], nan])
    
    if dim == 3:
        return go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color='black', width=1),
            name='Bonds',
            showlegend=False
        )
    else:
        return go.Scatter(
            x=x_lines, y=y_lines,
            mode='lines',
            line=dict(color='black', width=2),
            name='Bonds',
            showlegend=False
        )

def generate_k_path(points: Dict[str, Union[List, np.ndarray, torch.Tensor]], 
                    path_labels: List[str], 
                    resolution: int = 30) -> tuple:
    """
    Generates a k-path through high-symmetry points.
    
    Args:
        points: Dictionary mapping labels to coordinates (e.g. {'G': [0,0], 'M': [0.5, 0.5]})
        path_labels: List of labels defining the path (e.g. ['G', 'M', 'K', 'G'])
        resolution: Number of points per segment.
        
    Returns:
        (k_vecs, k_dist, node_indices)
        k_vecs: Tensor of k-vectors (N, D)
        k_dist: Tensor of cumulative distances (N,)
        node_indices: List of indices for the high-symmetry points.
    """
    k_vecs_list = []
    node_indices = [0]
    
    # Convert points to numpy for easier math
    pts_np = {}
    for k, v in points.items():
        if isinstance(v, torch.Tensor):
            pts_np[k] = v.detach().cpu().numpy().astype(float)
        else:
            pts_np[k] = np.array(v, dtype=float)
    
    for i in range(len(path_labels) - 1):
        start_label = path_labels[i]
        end_label = path_labels[i+1]
        
        start_vec = pts_np[start_label]
        end_vec = pts_np[end_label]
        
        # Determine number of points
        # If it's the last segment, include the end point
        is_last = (i == len(path_labels) - 2)
        num = resolution + 1 if is_last else resolution
        
        t = np.linspace(0, 1, num, endpoint=is_last)
        
        for ti in t:
            vec = (1 - ti) * start_vec + ti * end_vec
            k_vecs_list.append(vec)
            
        # The next node is at the current total length of list
        # Note: k_vecs_list length grows by 'resolution' each time (except last)
        next_idx = len(k_vecs_list) - 1
        node_indices.append(next_idx)

    k_vecs = torch.tensor(np.array(k_vecs_list), dtype=torch.float64)
    
    # Recalculate distances precisely from the vectors
    if len(k_vecs) > 0:
        diffs = torch.norm(k_vecs[1:] - k_vecs[:-1], dim=1)
        k_dist = torch.cat([torch.tensor([0.0], dtype=torch.float64), torch.cumsum(diffs, dim=0)])
    else:
        k_dist = torch.tensor([], dtype=torch.float64)
    
    return k_vecs, k_dist, node_indices

# --- Registered Plot Methods ---

@Plottable.register_plot_method('structure', backend='plotly')
def plot_structure(obj: Lattice, 
                   subs: Optional[Dict] = None, 
                   basis_offsets: Optional[List[Offset]] = None,
                   spin_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   plot_type: str = 'edge-and-node',
                   show: bool = True,
                   **kwargs) -> go.Figure:
    """
    Plots the structure of the lattice (sites, bonds, spins).
    
    Parameters
    ----------
    obj : `Lattice`
        The lattice to plot.
    subs : `Optional[Dict]`
        Dictionary of symbol substitutions.
    basis_offsets : `Optional[List[Offset]]`
        List of basis atom offsets.
    spin_data : `Optional[Union[np.ndarray, torch.Tensor]]`
        Array of spin vectors for each site.
    plot_type : `str`
        The type of plot to generate.
    show : `bool`
        Whether to display the plot immediately.
    **kwargs : `Any`
        Additional keyword arguments.
        
    Returns
    -------
    `go.Figure`
        The plot figure.
    """
    valid_types = ['edge-and-node', 'scatter']
    if plot_type not in valid_types:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Options: {valid_types}")

    coords = _compute_all_coords(obj, basis_offsets, subs)
    coords_np = coords.numpy()
    
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2] if obj.dim == 3 else None

    fig = go.Figure()

    # Bonds (Only for 'edge-and-node')
    if plot_type == 'edge-and-node':
        bonds_trace = _generate_bonds_trace(coords, obj.dim)
        if bonds_trace:
            fig.add_trace(bonds_trace)

    # Sites
    num_basis = len(basis_offsets) if basis_offsets else 1
    num_cells = coords.shape[0] // num_basis
    
    basis_colors = ['blue', 'red', 'green', 'orange', 'purple']
    colors = []
    for _ in range(num_cells):
        for b in range(num_basis):
            colors.append(basis_colors[b % len(basis_colors)])

    if obj.dim == 3:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color=colors),
            name='Sites'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=10, color=colors, symbol='circle'),
            name='Sites'
        ))

    # Spins (Optional)
    if spin_data is not None:
        if isinstance(spin_data, np.ndarray):
            spin_data = torch.from_numpy(spin_data)
        
        if spin_data.shape[0] != coords.shape[0]:
             raise ValueError(f"Spin data shape {spin_data.shape} does not match sites {coords.shape[0]}. "
                              f"Did you forget to provide spin data for all basis atoms?")

        spin_np = spin_data.numpy()
        
        if obj.dim == 3:
            fig.add_trace(go.Cone(
                x=x, y=y, z=z,
                u=spin_np[:, 0], v=spin_np[:, 1], w=spin_np[:, 2],
                sizemode="absolute", sizeref=0.5, anchor="tail",
                colorscale='Viridis', name='Spins'
            ))
        else:
            quiver = ff.create_quiver(x, y, spin_np[:, 0], spin_np[:, 1],
                                      scale=0.2, arrow_scale=0.3, name='Spins',
                                      line=dict(color='red'))
            fig.add_traces(quiver.data)

    # Layout
    if obj.dim == 3:
        fig.update_layout(title="3D Lattice System", scene=dict(aspectmode='data'))
    else:
        fig.update_layout(title="2D Lattice System", yaxis=dict(scaleanchor="x", scaleratio=1))

    if show:
        fig.show()
    return fig


@Plottable.register_plot_method('heatmap', backend='plotly')
def plot_heatmap(obj: Union[np.ndarray, torch.Tensor, object], 
                 title: str = "Matrix Visualization", 
                 show: bool = True,
                 **kwargs) -> go.Figure:
    """
    Plots a heatmap of the tensor (matrix). 

    Parameters
    ----------
    obj : `Union[np.ndarray, torch.Tensor, object]`
        The matrix to plot.
    title : `str`
        The title of the plot.
    show : `bool`
        Whether to display the plot immediately.

    Returns
    -------
    `go.Figure`
        The plot figure.

    If complex, plots Real and Imaginary parts side-by-side sharing the same color scale.
    """
    # 1. Standardize to PyTorch Tensor on CPU
    if hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor):
        tensor = obj.data.detach().cpu()
    elif isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
    else:
        tensor = torch.from_numpy(np.array(obj))
        
    if tensor.ndim != 2:
        raise ValueError(f"Heatmap requires a 2D matrix, got shape {tensor.shape}")

    is_complex = tensor.is_complex()
    
    if is_complex:
        real_part = tensor.real
        imag_part = tensor.imag
        
        # Calculate global range using torch
        limit = max(torch.abs(real_part).max(), torch.abs(imag_part).max()).item()
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Real Part", "Imaginary Part"))
        
        # Convert to numpy only for Plotly
        fig.add_trace(go.Heatmap(z=real_part.numpy(), 
                                 colorscale='RdBu', 
                                 zmin=-limit, zmax=limit,
                                 showscale=False), 
                      row=1, col=1)
                      
        fig.add_trace(go.Heatmap(z=imag_part.numpy(), 
                                 colorscale='RdBu', 
                                 zmin=-limit, zmax=limit,
                                 showscale=True), 
                      row=1, col=2)
    else:
        # Real matrix
        limit = torch.abs(tensor).max().item()
        fig = go.Figure(data=go.Heatmap(z=tensor.numpy(), 
                                        colorscale='RdBu', 
                                        zmin=-limit, zmax=limit,
                                        showscale=True))
        
    fig.update_yaxes(autorange="reversed") 
    fig.update_layout(title=title)
    
    if show:
        fig.show()
    return fig

@Plottable.register_plot_method('spectrum', backend='plotly')
def plot_spectrum(obj: Union[np.ndarray, torch.Tensor, object], 
                 title: str = "Spectrum Visualization", 
                 show: bool = True,
                 **kwargs) -> go.Figure:
    """
    Plots the eigenvalue spectrum of the tensor (matrix).
    If Hermitian/Symmetric, plots sorted real eigenvalues.
    If non-Hermitian, plots eigenvalues in the complex plane.
    
    Parameters
    ----------
    obj : `Union[np.ndarray, torch.Tensor, object]`
        The matrix to plot.
    title : `str`
        The title of the plot.
    show : `bool`
        Whether to display the plot immediately.
    **kwargs : `Any`
        Additional keyword arguments.

    Returns
    -------
    `go.Figure`
    """
    # 1. Standardize to PyTorch Tensor
    if hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor):
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
        is_hermitian = True # Zero matrix is Hermitian

    fig = go.Figure()

    # 3. Calculate and Plot
    if is_hermitian:
        # Returns real values sorted in ascending order
        # If matrix was real but just symmetric, eigvalsh still works fine
        evals = torch.linalg.eigvalsh(tensor)
        y_vals = evals.numpy()
        x_vals = np.arange(len(y_vals))
        
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, 
                                 mode='markers+lines', 
                                 marker=dict(size=6, color='blue'),
                                 name='Eigenvalues'))
        fig.update_layout(xaxis_title="Index", yaxis_title="Eigenvalue")
    else:
        # General case (Complex plane)
        evals = torch.linalg.eigvals(tensor)
        real_parts = evals.real.numpy()
        imag_parts = evals.imag.numpy()
        
        fig.add_trace(go.Scatter(x=real_parts, y=imag_parts, 
                                 mode='markers', 
                                 marker=dict(size=8, color='red'),
                                 name='Eigenvalues'))
        fig.update_layout(xaxis_title="Real Part", yaxis_title="Imaginary Part")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(title=title)
    
    if show:
        fig.show()
    return fig

@Plottable.register_plot_method('bandstructure', backend='plotly')
def plot_bandstructure(obj: Union[np.ndarray, torch.Tensor, object],
                       k_distances: Optional[Union[np.ndarray, torch.Tensor]] = None,
                       k_node_indices: Optional[List[int]] = None,
                       k_node_labels: Optional[List[str]] = None,
                       title: str = "Band Structure",
                       show: bool = True,
                       **kwargs) -> go.Figure:
    """
    Plots the band structure using provided energies and k-path data.
    
    Parameters
    ----------
    obj : `Union[np.ndarray, torch.Tensor, object]`
        (N_k, N_bands) array of eigenvalues or Tensor object.
    k_distances : `Optional[Union[np.ndarray, torch.Tensor]]`
        (N_k,) array of cumulative distances along the k-path.
    k_node_indices : `Optional[List[int]]`
        Indices of high-symmetry points in the path.
    k_node_labels : `Optional[List[str]]`
        Labels for the high-symmetry points.
    title : `str`
        Plot title.
    show : `bool`
        Whether to show the plot.
        
    Returns
    -------
    `go.Figure`
    """
    # Standardize inputs
    if hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor):
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
        fig.add_trace(go.Scatter(
            x=k_distances,
            y=energies[:, b],
            mode='lines',
            line=dict(color='black', width=1.5),
            name=f'Band {b}',
            showlegend=False
        ))
        
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
                fig.add_vline(x=x_val, line_width=1, line_dash="dash", line_color="grey")
                
        if tick_vals:
            fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text)
            
    fig.update_layout(
        title=title,
        xaxis_title="Wave Vector",
        yaxis_title="Energy",
        template="simple_white"
    )
    
    if show:
        fig.show()
    return fig
