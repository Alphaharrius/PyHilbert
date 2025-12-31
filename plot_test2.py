import sympy as sy
import torch
import itertools
import numpy as np
import plotly.graph_objects as go
from pyhilbert.spatials import Lattice, Offset
from pyhilbert.old_version_plot import LatticePlotter

def test_diamond_ab_modes():
    print("Setting up 2x2x2 Stack with A/B Sublattice Coloring...")
    a = sy.Symbol('a')
    a_val = 3.57
    
    basis_vectors = sy.ImmutableDenseMatrix([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]).T * a 
    
    super_lattice = Lattice(basis=basis_vectors, shape=(2, 2, 2))
    
    offsets = []
    # Order: A, B, A, B...
    for n1, n2, n3 in itertools.product([0, 1], repeat=3):
        frac_a = [sy.Rational(n1, 2), sy.Rational(n2, 2), sy.Rational(n3, 2)]
        offsets.append(Offset(rep=sy.ImmutableDenseMatrix(frac_a), space=super_lattice.affine))
        frac_b = [x + sy.Rational(1, 8) for x in frac_a]
        offsets.append(Offset(rep=sy.ImmutableDenseMatrix(frac_b), space=super_lattice.affine))
        
    plotter = LatticePlotter(
        lattice=super_lattice,
        subs={a: a_val}, 
        offsets=offsets
    )
    
    fig = plotter.plot_system(show=False)
    
    # --- SHIFT LOGIC ---
    atom_coords = plotter.coords.numpy()
    global_atom_centroid = np.mean(atom_coords, axis=0)
    basis_num = np.array(super_lattice.affine.basis.subs({a: a_val})).astype(float)
    grid_corners_frac = list(itertools.product([0, 2], repeat=3)) 
    grid_corners = [basis_num @ np.array(c) for c in grid_corners_frac]
    global_box_centroid = np.mean(grid_corners, axis=0)
    shift_vec = global_atom_centroid - global_box_centroid
    
    # --- COLOR BY SUBLATTICE A/B ---
    # The coords are generated as (Cell0_Offset0, Cell0_Offset1, ..., Cell0_Offset15, Cell1_Offset0...)
    # In our offsets list, even indices (0, 2, ...) are Sublattice A. Odd are Sublattice B.
    # Total offsets = 16.
    # So the pattern A, B, A, B... repeats for every atom.
    
    colors = []
    for i in range(len(atom_coords)):
        if i % 2 == 0:
            colors.append('blue') # Sublattice A
        else:
            colors.append('red')  # Sublattice B
            
    # Update Sites Trace
    for trace in fig.data:
        if trace.name == 'Sites':
            trace.marker.color = colors
            trace.marker.size = 8
            # Add legend group to distinguish? Hard with single trace.
            break

    # --- DRAW 8 STACKED HULLS ---
    coeffs_list = list(itertools.product([0, 1], repeat=3))
    base_corners = []
    for coeffs in coeffs_list:
        base_corners.append(basis_num @ np.array(coeffs))
    base_corners = np.array(base_corners)
    
    for idx, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
        cell_offset = basis_num @ np.array([i, j, k])
        curr_corners = base_corners + cell_offset + shift_vec
        
        # Hull Volume
        fig.add_trace(go.Mesh3d(
            x=curr_corners[:, 0], y=curr_corners[:, 1], z=curr_corners[:, 2],
            alphahull=0, opacity=0.05, color='cyan', name=f'Hull {idx}', flatshading=True,
            showlegend=False
        ))
        
        # Hull Edges
        x_lines, y_lines, z_lines = [], [], []
        for c1_idx, c1 in enumerate(coeffs_list):
            for c2_idx, c2 in enumerate(coeffs_list):
                if c2_idx > c1_idx:
                    dist = sum(abs(x-y) for x,y in zip(c1, c2))
                    if dist == 1:
                        p1, p2 = curr_corners[c1_idx], curr_corners[c2_idx]
                        x_lines.extend([p1[0], p2[0], None])
                        y_lines.extend([p1[1], p2[1], None])
                        z_lines.extend([p1[2], p2[2], None])

        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines', line=dict(color='black', width=1), name=f'Edges {idx}',
            showlegend=False
        ))

    # Center Corner Marker
    center_corner = (basis_num @ np.array([1.0, 1.0, 1.0])) + shift_vec
    fig.add_trace(go.Scatter3d(
        x=[center_corner[0]], y=[center_corner[1]], z=[center_corner[2]],
        mode='markers', marker=dict(size=10, color='gold', symbol='x'),
        name='Center Corner'
    ))

    fig.update_layout(title="Diamond Lattice: Sublattice A (Blue) vs B (Red)")
    fig.show()

if __name__ == "__main__":
    test_diamond_ab_modes()