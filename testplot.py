import sympy as sy
import torch
import itertools
import numpy as np
import plotly.graph_objects as go
from src.pyhilbert.spatials import Lattice, Offset
from src.pyhilbert.plot import LatticePlotter

def test_stacked_hulls_shifted():
    print("Setting up 2x2x2 Stack of Blocked Diamond Cells (Shifted)...")
    a = sy.Symbol('a')
    a_val = 3.57
    
    basis_vectors = sy.ImmutableDenseMatrix([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]).T * a 
    
    super_lattice = Lattice(basis=basis_vectors, shape=(2, 2, 2))
    
    offsets = []
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
    
    # --- CALCULATE SHIFT VECTOR (From Single Cell Logic) ---
    # We need to calculate the shift based on a single unit cell's atom distribution vs its box
    # Let's generate coordinates for just one cell to compute the shift
    # (Or just use the known relative shift from previous attempts)
    
    # 1. Get coords of atoms in the first unit cell (approx)
    # Actually, simpler: compute centroid of ALL atoms and centroid of the 2x2x2 BOX grid.
    # If the shift is consistent, this global shift is the same as the local shift.
    
    atom_coords = plotter.coords.numpy()
    global_atom_centroid = np.mean(atom_coords, axis=0)
    
    basis_num = np.array(super_lattice.affine.basis.subs({a: a_val})).astype(float)
    
    # Calculate corners of the full 2x2x2 grid to find its center
    # Grid goes from (0,0,0) to (2,2,2) in basis coords
    grid_corners_frac = list(itertools.product([0, 2], repeat=3)) 
    grid_corners = [basis_num @ np.array(c) for c in grid_corners_frac]
    global_box_centroid = np.mean(grid_corners, axis=0)
    
    shift_vec = global_atom_centroid - global_box_centroid
    print(f"Applying shift vector: {shift_vec}")

    # --- CENTER CORNER ---
    # The geometric center of the 2x2x2 grid is at (1,1,1)
    # We must shift this point too!
    center_corner_unshifted = basis_num @ np.array([1.0, 1.0, 1.0])
    center_corner = center_corner_unshifted + shift_vec # The "Center Corner" relative to the atoms
    
    # --- HIGHLIGHT 14 NEAREST ATOMS ---
    dists = np.linalg.norm(atom_coords - center_corner, axis=1)
    k = 14
    closest_indices = np.argpartition(dists, k)[:k]
    
    colors = np.array(['rgba(100, 100, 255, 0.3)'] * len(atom_coords), dtype=object)
    colors[closest_indices] = 'red'
    
    for trace in fig.data:
        if trace.name == 'Sites':
            trace.marker.color = colors
            sizes = np.full(len(atom_coords), 6)
            sizes[closest_indices] = 12
            trace.marker.size = sizes
            break

    # --- DRAW 8 STACKED HULLS (SHIFTED) ---
    coeffs_list = list(itertools.product([0, 1], repeat=3))
    base_corners = []
    for coeffs in coeffs_list:
        base_corners.append(basis_num @ np.array(coeffs))
    base_corners = np.array(base_corners)
    
    for idx, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
        # Calculate unshifted position
        cell_offset = basis_num @ np.array([i, j, k])
        curr_corners = base_corners + cell_offset
        
        # APPLY SHIFT
        curr_corners += shift_vec
        
        fig.add_trace(go.Mesh3d(
            x=curr_corners[:, 0], y=curr_corners[:, 1], z=curr_corners[:, 2],
            alphahull=0, opacity=0.05, color='cyan', name=f'Hull {idx}', flatshading=True,
            showlegend=False
        ))
        
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
            mode='lines', line=dict(color='black', width=2), name=f'Edges {idx}',
            showlegend=False
        ))

    # Center Marker
    fig.add_trace(go.Scatter3d(
        x=[center_corner[0]], y=[center_corner[1]], z=[center_corner[2]],
        mode='markers', marker=dict(size=10, color='gold', symbol='x'),
        name='Shifted Center Corner'
    ))

    fig.update_layout(title="8 Stacked Hulls (Shifted) with 14 Closest Sites")
    fig.show()
    fig.write_html("my_interactive_plot2.html")

if __name__ == "__main__":
    test_stacked_hulls_shifted()