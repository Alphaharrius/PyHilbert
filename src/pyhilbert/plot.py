import torch
import numpy as np
import sympy as sy
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Optional, List, Union, Dict, Callable
from .spatials import Lattice, cartes, Offset

class LatticePlotter:
    def __init__(self, lattice: Lattice, subs: Optional[Dict] = None, offsets: Optional[List[Offset]] = None):
        """
        Initialize the plotter.
        
        Args:
            lattice: The Lattice object to plot.
            subs: Dictionary of symbol substitutions for lattice constants (e.g. {a: 1.5}).
                  If None, attempts to substitute all free symbols with 1.0.
            offsets: Optional list of basis atom offsets. If None, plots only lattice points (Offset=0).
                     To visualize a crystal with a basis (e.g. Diamond), pass the basis offsets here.
        """
        self.lattice = lattice
        self.subs = subs
        # Default to a single atom at origin if no offsets provided
        if offsets is None:
            self.basis_offsets = [Offset(rep=sy.ImmutableDenseMatrix([0]*lattice.dim), space=lattice.affine)]
        else:
            self.basis_offsets = offsets
            
        self._coords: Optional[torch.Tensor] = None

    @property
    def coords(self) -> torch.Tensor:
        if self._coords is None:
            self._coords = self._compute_coordinates()
        return self._coords

    def _to_tensor_vector(self, lattice_point_offset: Offset, basis_offset: Optional[Offset] = None) -> torch.Tensor:
        """
        Convert a lattice point + basis offset to a torch tensor.
        R = R_lattice + R_basis
        """
        basis_matrix = lattice_point_offset.space.basis
        
        # Lattice point position in crystal coords (integers)
        rep_lattice = lattice_point_offset.rep
        
        # Basis position in crystal coords (fractions)
        if basis_offset:
            rep_basis = basis_offset.rep
        else:
            rep_basis = sy.zeros(rep_lattice.rows, 1)

        # Total position in crystal coords
        rep_total = rep_lattice + rep_basis
        
        # Cartesian position: basis_matrix * rep_total
        pos = basis_matrix @ rep_total
        
        if self.subs is not None:
            pos_eval = pos.subs(self.subs)
        else:
            subs_dict = {s: 1.0 for s in pos.free_symbols}
            pos_eval = pos.subs(subs_dict)
        
        try:
            return torch.tensor(np.array(pos_eval).astype(np.float64).flatten(), dtype=torch.float64)
        except Exception as e:
            raise ValueError(f"Could not convert coordinates to float. Remaining symbols: {pos_eval.free_symbols}. Please provide 'subs' dictionary.") from e

    def _compute_coordinates(self) -> torch.Tensor:
        """Get all site coordinates (lattice points + basis atoms)."""
        # Get all Bravais lattice points
        lattice_offsets = cartes(self.lattice)
        
        all_coords_list = []
        
        # For each lattice point, add all basis atoms
        for lat_off in lattice_offsets:
            for basis_off in self.basis_offsets:
                # We need to be careful: if basis_off is None or zero, it's just the lattice point.
                # basis_off should be an Offset object.
                coord = self._to_tensor_vector(lat_off, basis_off)
                all_coords_list.append(coord)
                
        if not all_coords_list:
            return torch.empty((0, self.lattice.dim))
            
        return torch.stack(all_coords_list)

    def _generate_bonds_traces(self) -> Optional[Union[go.Scatter, go.Scatter3d]]:
        """Generate bond lines connecting nearest neighbors using PyTorch."""
        coords = self.coords
        if coords.size(0) < 2:
            return None
            
        # Compute distance matrix efficiently with torch
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dists = torch.norm(diff, dim=-1)
        
        # Ignore self-distance
        dists.fill_diagonal_(float('inf'))
        
        min_dist = torch.min(dists)
        if torch.isinf(min_dist):
            return None
            
        # connect neighbors within tolerance
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
            if self.lattice.dim == 3:
                z_lines.extend([p1_np[i, 2], p2_np[i, 2], nan])
        
        if self.lattice.dim == 3:
            return go.Scatter3d(
                x=x_lines, y=y_lines, z=z_lines,
                mode='lines',
                line=dict(color='rgba(100,100,100,0.5)', width=2),
                name='Bonds',
                showlegend=False
            )
        else:
            return go.Scatter(
                x=x_lines, y=y_lines,
                mode='lines',
                line=dict(color='rgba(100,100,100,0.5)', width=1),
                name='Bonds',
                showlegend=False
            )

    def plot_system(self, spin_data: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                    show: bool = True, plot_type: str = 'edge-and-node') -> go.Figure:
        """
        Main plotting method for the lattice system (sites, bonds, spins).
        """
        valid_types = ['edge-and-node', 'scatter']
        if plot_type not in valid_types:
            raise ValueError(f"Invalid plot_type '{plot_type}'. Options: {valid_types}")

        coords = self.coords
        coords_np = coords.numpy()
        x = coords_np[:, 0]
        y = coords_np[:, 1]
        z = coords_np[:, 2] if self.lattice.dim == 3 else None

        fig = go.Figure()

        # Bonds
        if plot_type == 'edge-and-node':
            bonds_trace = self._generate_bonds_traces()
            if bonds_trace:
                fig.add_trace(bonds_trace)

        # Sites
        # Color atoms by sublattice if there are multiple basis atoms
        num_basis = len(self.basis_offsets)
        num_cells = coords.shape[0] // num_basis
        
        # Generate colors for basis atoms
        # e.g. Atom 0: Blue, Atom 1: Red...
        basis_colors = ['blue', 'red', 'green', 'orange', 'purple']
        colors = []
        for _ in range(num_cells):
            for b in range(num_basis):
                colors.append(basis_colors[b % len(basis_colors)])
        
        if self.lattice.dim == 3:
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

        # Spins
        if spin_data is not None:
            if isinstance(spin_data, np.ndarray):
                spin_data = torch.from_numpy(spin_data)
            
            if spin_data.shape[0] != coords.shape[0]:
                raise ValueError(f"Spin data shape {spin_data.shape} does not match sites {coords.shape[0]}. "
                                 f"Did you forget to provide spin data for all basis atoms?")
            
            spin_np = spin_data.numpy()
            
            if self.lattice.dim == 3:
                fig.add_trace(go.Cone(
                    x=x, y=y, z=z,
                    u=spin_np[:, 0], v=spin_np[:, 1], w=spin_np[:, 2],
                    sizemode="absolute",
                    sizeref=0.5,
                    anchor="tail",
                    colorscale='Viridis',
                    name='Spins'
                ))
            else: # 2D
                quiver = ff.create_quiver(x, y, spin_np[:, 0], spin_np[:, 1],
                                          scale=0.2, arrow_scale=0.3, name='Spins', 
                                          line=dict(color='red'))
                fig.add_traces(quiver.data)

        # Layout
        if self.lattice.dim == 3:
            fig.update_layout(
                title=f"3D Lattice System",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="z",
                    aspectmode='data'
                )
            )
        else:
            fig.update_layout(
                title=f"2D Lattice System",
                xaxis_title="x",
                yaxis_title="y",
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

        if show:
            fig.show()
        return fig

    def plot_wavefunction(self, psi: Union[np.ndarray, torch.Tensor], 
                          scale: float = 1.0, show: bool = True,
                          title: str = "Wavefunction Visualization") -> go.Figure:
        """
        Visualizes a complex scalar field (wavefunction) on the lattice.
        
        Args:
            psi: Complex vector of shape (num_sites,).
            scale: Scaling factor for marker sizes.
            show: Whether to show the plot immediately.
            title: Plot title.
        """
        if isinstance(psi, torch.Tensor):
            psi = psi.detach().cpu().numpy()
        elif not isinstance(psi, np.ndarray):
             psi = np.array(psi)

        coords = self.coords
        coords_np = coords.numpy()
        
        if psi.shape[0] != coords_np.shape[0]:
             # Try to handle basis atoms if psi is provided per unit cell but flattened? 
             # Or just raise error.
             raise ValueError(f"Wavefunction shape {psi.shape} does not match number of sites {coords_np.shape[0]}.")

        x = coords_np[:, 0]
        y = coords_np[:, 1]
        z = coords_np[:, 2] if self.lattice.dim == 3 else None
        
        # Amplitude -> Size
        amp = np.abs(psi)
        max_amp = amp.max()
        if max_amp < 1e-9:
            sizes = np.zeros_like(amp)
        else:
            # Normalize and scale. 
            sizes = 20 * (amp / max_amp) * scale
        
        # Phase -> Color (Hue)
        # Angle is in (-pi, pi]
        phase = np.angle(psi) 
        
        fig = go.Figure()
        
        # Add faint bonds for context
        bonds_trace = self._generate_bonds_traces()
        if bonds_trace:
            if isinstance(bonds_trace, go.Scatter3d):
                 bonds_trace.line.color = 'rgba(200,200,200,0.2)'
            else:
                 bonds_trace.line.color = 'rgba(200,200,200,0.2)'
            fig.add_trace(bonds_trace)

        marker_dict = dict(
            size=sizes,
            color=phase, 
            colorscale='HSV', 
            colorbar=dict(
                title="Phase", 
                tickvals=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                ticktext=["-π", "-π/2", "0", "π/2", "π"]
            ),
            cmin=-np.pi, 
            cmax=np.pi,
            opacity=0.9,
            line=dict(width=0)
        )

        if self.lattice.dim == 3:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, 
                mode='markers', 
                marker=marker_dict,
                name='Amplitude/Phase',
                text=[f"Amp: {a:.3f}, Phase: {p:.3f}" for a, p in zip(amp, phase)]
            ))
            fig.update_layout(scene=dict(aspectmode='data'))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, 
                mode='markers', 
                marker=marker_dict,
                name='Amplitude/Phase',
                text=[f"Amp: {a:.3f}, Phase: {p:.3f}" for a, p in zip(amp, phase)]
            ))
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            
        fig.update_layout(title=title)
            
        if show:
            fig.show()
        return fig

    def _generate_k_path(self, points: Dict[str, Union[List, np.ndarray, torch.Tensor]], 
                        path_labels: List[str], 
                        resolution: int) -> tuple:
        """Helper to generate k-path vectors and distances."""
        k_path_vecs = []
        k_dist = [0.0]
        current_dist = 0.0
        indices = [0]
        
        # Ensure points are numpy/torch compatible
        pts = {}
        for k, v in points.items():
            if isinstance(v, torch.Tensor):
                pts[k] = v.numpy()
            else:
                pts[k] = np.array(v)
                
        for i in range(len(path_labels) - 1):
            start_label = path_labels[i]
            end_label = path_labels[i+1]
            
            start_vec = pts[start_label]
            end_vec = pts[end_label]
            
            segment_len = np.linalg.norm(end_vec - start_vec)
            
            # Generate points for this segment
            if i == len(path_labels) - 2:
                num = resolution + 1 # Include end
            else:
                num = resolution # Exclude end
                
            t = np.linspace(0, 1, num, endpoint=(i == len(path_labels) - 2))
            
            for ti in t:
                # Vector
                vec = (1 - ti) * start_vec + ti * end_vec
                k_path_vecs.append(vec)
                
            # Update current distance for next segment
            current_dist += segment_len
            
            indices.append(indices[-1] + resolution)
            
        k_path_vecs = np.array(k_path_vecs)
        
        diffs = np.linalg.norm(k_path_vecs[1:] - k_path_vecs[:-1], axis=1)
        dists = np.concatenate(([0.0], np.cumsum(diffs)))
        
        return k_path_vecs, dists, indices

    def plot_bands(self, energies: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   k_path_distances: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   k_node_indices: Optional[List[int]] = None, 
                   k_node_labels: Optional[List[str]] = None,
                   hamiltonian: Optional[Callable] = None,
                   points: Optional[Dict[str, Union[List, np.ndarray, torch.Tensor]]] = None,
                   path_labels: Optional[List[str]] = None,
                   k_resolution: int = 30,
                   show: bool = True) -> go.Figure:
        """
        Plot band structure.
        
        Can be used in two modes:
        1. Data mode: Provide `energies` (and optionally `k_node_indices`, `k_node_labels`).
        2. Calculation mode: Provide `hamiltonian`, `points`, and `path_labels`.
           The function will generate the path, calculate energies, and plot.
        """
        
        # Auto-calculation logic
        if hamiltonian is not None:
            if points is None or path_labels is None:
                raise ValueError("If 'hamiltonian' is provided, 'points' and 'path_labels' must also be provided.")
                
            # Generate path
            k_vecs, k_dists, indices = self._generate_k_path(points, path_labels, k_resolution)
            
            # Calculate energies
            k_vecs_t = torch.tensor(k_vecs)
            
            try:
                H_k = hamiltonian(k_vecs_t) # (N, D, D)
                evals = torch.linalg.eigvalsh(H_k) # (N, D)
                energies = evals.detach().numpy() if isinstance(evals, torch.Tensor) else evals
            except Exception:
                # Fallback to loop
                evals_list = []
                for k in k_vecs_t:
                    H = hamiltonian(k)
                    e = torch.linalg.eigvalsh(H)
                    evals_list.append(e)
                energies = torch.stack(evals_list).detach().numpy()
                
            # Set variables for plotting
            k_path_distances = k_dists
            k_node_indices = indices
            k_node_labels = path_labels
            
        # Validation
        if energies is None:
            raise ValueError("Must provide either 'energies' or 'hamiltonian' (with 'points' and 'path_labels').")

        if isinstance(energies, torch.Tensor):
            energies = energies.numpy()
            
        num_k_points = energies.shape[0]
        
        if k_path_distances is None:
            k_path_distances = np.arange(num_k_points)
        elif isinstance(k_path_distances, torch.Tensor):
            k_path_distances = k_path_distances.numpy()

        fig = go.Figure()
        
        num_bands = energies.shape[1]
        
        for b in range(num_bands):
            fig.add_trace(go.Scatter(
                x=k_path_distances,
                y=energies[:, b],
                mode='lines',
                line=dict(color='black', width=1.5),
                name=f'Band {b}',
                showlegend=False
            ))
            
        # Add vertical lines for high-symmetry points
        if k_node_indices:
            for idx in k_node_indices:
                if 0 <= idx < len(k_path_distances):
                    x_val = k_path_distances[idx]
                    fig.add_vline(x=x_val, line_width=1, line_dash="dash", line_color="grey")
                
        # Set x-axis ticks to labels
        if k_node_indices and k_node_labels and len(k_node_indices) == len(k_node_labels):
            tick_vals = [k_path_distances[i] for i in k_node_indices if 0 <= i < len(k_path_distances)]
            valid_labels = [label for i, label in zip(k_node_indices, k_node_labels) if 0 <= i < len(k_path_distances)]
            
            fig.update_xaxes(
                ticktext=valid_labels,
                tickvals=tick_vals
            )
            
        fig.update_layout(
            title="Band Structure",
            xaxis_title="Wave Vector",
            yaxis_title="Energy",
            template="simple_white"
        )
        
        if show:
            fig.show()
        return fig

    def plot_band_surface(self, hamiltonian: Callable, 
                          k_range_x: tuple = (-np.pi, np.pi), 
                          k_range_y: tuple = (-np.pi, np.pi),
                          resolution: int = 50,
                          show: bool = True) -> go.Figure:
        """
        Plots the 2D band structure as 3D surfaces.
        
        Args:
            hamiltonian: Function k_vec (2,) -> H (D,D) (or batch version).
            k_range_x: (min, max) for kx.
            k_range_y: (min, max) for ky.
            resolution: Grid resolution.
        """
        kx = np.linspace(k_range_x[0], k_range_x[1], resolution)
        ky = np.linspace(k_range_y[0], k_range_y[1], resolution)
        KX, KY = np.meshgrid(kx, ky)
        
        # Flatten for calculation
        k_vecs = np.stack([KX.flatten(), KY.flatten()], axis=1) # (N, 2)
        
        k_vecs_t = torch.tensor(k_vecs, dtype=torch.float64) # Ensure float64 or whatever H expects
        
        try:
            # Try batch
            H_k = hamiltonian(k_vecs_t)
            evals = torch.linalg.eigvalsh(H_k)
            energies = evals.detach().numpy()
        except Exception:
            # Loop fallback
            evals_list = []
            for k in k_vecs_t:
                H = hamiltonian(k)
                e = torch.linalg.eigvalsh(H)
                evals_list.append(e)
            energies = torch.stack(evals_list).detach().numpy()
            
        num_bands = energies.shape[1]
        
        fig = go.Figure()
        
        for b in range(num_bands):
            E_grid = energies[:, b].reshape(resolution, resolution)
            fig.add_trace(go.Surface(
                x=KX, y=KY, z=E_grid,
                colorscale='Viridis',
                opacity=0.9,
                name=f'Band {b}',
                showscale=(b==0) # Only show one colorbar
            ))
            
        fig.update_layout(
            title="2D Band Structure Surface",
            scene=dict(
                xaxis_title="kx",
                yaxis_title="ky",
                zaxis_title="Energy"
            )
        )
        
        if show:
            fig.show()
        return fig

    @staticmethod
    def plot_matrix(matrix: Union[np.ndarray, torch.Tensor], title: str = "Matrix Visualization", show: bool = True) -> go.Figure:
        """
        Plots Real and Imaginary parts of a matrix side-by-side.
        """
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
            
        real_part = np.real(matrix)
        imag_part = np.imag(matrix)
        
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Real Part", "Imaginary Part"))
        
        fig.add_trace(go.Heatmap(z=real_part, colorscale='RdBu', zmid=0, showscale=True), row=1, col=1)
        fig.add_trace(go.Heatmap(z=imag_part, colorscale='RdBu', zmid=0, showscale=True), row=1, col=2)
        
        fig.update_yaxes(autorange="reversed") # Matrix convention
        
        fig.update_layout(title=title)
        
        if show:
            fig.show()
        return fig

    @staticmethod
    def plot_spectrum(eigenvalues: Union[np.ndarray, torch.Tensor, List], 
                      title: str = "Eigenvalue Spectrum", 
                      show: bool = True) -> go.Figure:
        """
        Plots sorted eigenvalues. Useful for entanglement spectrum.
        """
        if isinstance(eigenvalues, torch.Tensor):
            ev = eigenvalues.detach().cpu().numpy()
        else:
            ev = np.array(eigenvalues)
            
        ev_sorted = np.sort(ev)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=ev_sorted,
            mode='markers+lines',
            marker=dict(size=6),
            name='Eigenvalues'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Index",
            yaxis_title="Eigenvalue"
        )
        
        if show:
            fig.show()
        return fig
