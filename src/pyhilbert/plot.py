import torch
import numpy as np
import sympy as sy
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Optional, List, Union, Dict, Callable
from .abstracts import Plottable
from .spatials import Lattice, cartes, Offset

#lattice related plots
@Plottable.register_plot_method('structure')
def plot_structure(obj, lattice: Lattice, offsets: Optional[List[Offset]] = None, 
                   subs: Optional[Dict] = None, spin_data: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                   plot_type: str = 'edge-and-node', show: bool = True, **kwargs) -> go.Figure:
    """Plots the structure of the lattice"""
    pass

#wavefunction related plots
@Plottable.register_plot_method('wavefunction')
def plot_wavefunction(obj, *args,**kwargs) -> go.Figure:
    """Plots the wavefunction"""
    pass

#matrix related plots
@Plottable.register_plot_method('re-im')
def plot_heatmap(obj, matrix: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                 title: str = "Matrix Visualization", show: bool = True, **kwargs) -> go.Figure:
    """Plots Real and Imaginary parts as separate heatmaps"""
    pass

@Plottable.register_plot_method('eigenspectrum')
def plot_eigenspectrum(obj, eigenvalues: Optional[Union[np.ndarray, torch.Tensor, List]] = None, 
                       title: str = "Eigenvalue Spectrum", show: bool = True, **kwargs) -> go.Figure:
    """Scatter plot of sorted eigenvalues"""
    pass

#band structure related plots
@Plottable.register_plot_method('band_structure')
def plot_band_structure(obj, *args,**kwargs) -> go.Figure:
    """Plots the band structure"""
    pass

#band surface related plots
@Plottable.register_plot_method('band_surface')
def plot_band_surface(obj, hamiltonian: Callable, 
                      k_range_x: tuple = (-np.pi, np.pi), 
                      k_range_y: tuple = (-np.pi, np.pi),
                      resolution: int = 50,
                      show: bool = True, **kwargs) -> go.Figure:
    """Plots the 2D band structure as 3D surfaces"""
    pass

@Plottable.register_plot_method('wannier3D')
def plot_wannier3D(obj,*args,**kwargs) -> go.Figure:
    """Plots the Wannier functions as 3D surfaces"""
    pass

@Plottable.register_plot_method('wannier2D')
def plot_wannier2D(obj,*args,**kwargs) -> go.Figure:
    """Plots the Wannier functions as 2D surfaces"""
    pass

