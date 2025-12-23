import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import sympy as sy
import itertools
from src.pyhilbert.spatials import Lattice, Offset
from src.pyhilbert.plot import LatticePlotter

print("Setting up geometry...")
a = sy.Symbol('a')
a_val = 3.57

basis_vectors = sy.ImmutableDenseMatrix([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]).T * a 

super_lattice = Lattice(basis_vectors, shape=(2, 2, 2))

offsets = []
for n1, n2, n3 in itertools.product([0, 1], repeat=3):
    frac_a = [sy.Rational(n1, 2), sy.Rational(n2, 2), sy.Rational(n3, 2)]
    offsets.append(Offset(sy.ImmutableDenseMatrix(frac_a), super_lattice.affine))
    frac_b = [x + sy.Rational(1, 8) for x in frac_a]
    offsets.append(Offset(sy.ImmutableDenseMatrix(frac_b), super_lattice.affine))

plotter = LatticePlotter(super_lattice, subs={a: a_val}, offsets=offsets)

atom_coords = plotter.coords.numpy()
global_atom_centroid = np.mean(atom_coords, axis=0)
basis_num = np.array(super_lattice.affine.basis.subs({a: a_val})).astype(float)
grid_corners = [basis_num @ np.array(c) for c in list(itertools.product([0, 2], repeat=3))]
global_box_centroid = np.mean(grid_corners, axis=0)
shift_vec = global_atom_centroid - global_box_centroid

# --- REBUILD FIGURE MANUALLY FOR CONTROL ---
fig = go.Figure()

# 1. Add Stacked Hulls (Background)
coeffs_list = list(itertools.product([0, 1], repeat=3))
base_corners = np.array([basis_num @ np.array(c) for c in coeffs_list])

for idx, (i, j, k) in enumerate(itertools.product([0, 1], repeat=3)):
    cell_offset = basis_num @ np.array([i, j, k])
    curr_corners = base_corners + cell_offset + shift_vec
    
    # Volume (Disable hover/click)
    fig.add_trace(go.Mesh3d(
        x=curr_corners[:,0], y=curr_corners[:,1], z=curr_corners[:,2],
        alphahull=0, opacity=0.05, color='cyan', flatshading=True, 
        hoverinfo='skip'
    ))
    
    # Edges (Disable hover/click)
    x_l, y_l, z_l = [], [], []
    for c1_i, c1 in enumerate(coeffs_list):
        for c2_i, c2 in enumerate(coeffs_list):
            if c2_i > c1_i and sum(abs(x-y) for x,y in zip(c1,c2)) == 1:
                p1, p2 = curr_corners[c1_i], curr_corners[c2_i]
                x_l.extend([p1[0], p2[0], None])
                y_l.extend([p1[1], p2[1], None])
                z_l.extend([p1[2], p2[2], None])
    fig.add_trace(go.Scatter3d(
        x=x_l, y=y_l, z=z_l, mode='lines', line=dict(color='black', width=1), 
        hoverinfo='skip', showlegend=False
    ))

# 2. Add Bonds (From plotter logic, simplified)
# We can skip bonds or re-implement if needed, but for now let's focus on clickable atoms.
# If you need bonds, call plotter._generate_bonds_traces() and add it here.
bonds_trace = plotter._generate_bonds_traces()
if bonds_trace:
    bonds_trace.hoverinfo = 'skip'
    fig.add_trace(bonds_trace)

# 3. Add Sites (LAST so they are on top)
initial_colors = []
for i in range(len(atom_coords)):
    initial_colors.append('blue' if i % 2 == 0 else 'red')

fig.add_trace(go.Scatter3d(
    x=atom_coords[:, 0],
    y=atom_coords[:, 1],
    z=atom_coords[:, 2],
    mode='markers',
    marker=dict(size=12, color=initial_colors),
    name='Sites',
    hoverinfo='text',
    text=[f"Atom {i}" for i in range(len(atom_coords))]
))

fig.update_layout(
    title="Interactive: Click Atom to Cycle Color",
    scene=dict(aspectmode='data'),
    clickmode='event', # Enable click events
    uirevision='constant' # Important: Keeps camera view constant after update
)

# --- DASH APP ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Interactive Lattice Coloring"),
    dcc.Graph(id='lattice-graph', figure=fig, style={'height': '90vh'})
])

@app.callback(
    Output('lattice-graph', 'figure'),
    Input('lattice-graph', 'clickData'),
    State('lattice-graph', 'figure')
)
def update_color(clickData, fig_json):
    if not clickData:
        return dash.no_update
    
    # Debug print to terminal to verify click is received
    print(f"Click received: {clickData['points'][0]['pointIndex']}")
    
    point = clickData['points'][0]
    curve_idx = point['curveNumber']
    point_idx = point['pointIndex']
    
    # Identify Sites trace (It's the last one)
    # But safer to check name
    sites_trace_idx = None
    for i, data in enumerate(fig_json['data']):
        if data.get('name') == 'Sites':
            sites_trace_idx = i
            break
            
    if sites_trace_idx is None or curve_idx != sites_trace_idx:
        return dash.no_update

    current_colors = fig_json['data'][sites_trace_idx]['marker']['color']
    
    # Handle single color string vs list
    if not isinstance(current_colors, list):
        current_colors = [current_colors] * len(atom_coords)
        
    clicked_color = current_colors[point_idx]
    
    if clicked_color in ['blue', 'red']:
        new_color = 'green'
    elif clicked_color == 'green':
        new_color = 'gold'
    else:
        new_color = 'blue' if point_idx % 2 == 0 else 'red'
        
    current_colors[point_idx] = new_color
    fig_json['data'][sites_trace_idx]['marker']['color'] = current_colors
    
    return fig_json

if __name__ == '__main__':
    print("Starting Dash server...")
    app.run(debug=True, use_reloader=False)