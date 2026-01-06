import torch
import numpy as np
import sympy as sy
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Optional, List, Union, Dict, Callable
from .abstracts import Plottable
from .spatials import Lattice, cartes, Offset

@Plottable.register_plot_method('structure', backend='plotly')
def plot_structure(obj, *args, **kwargs) -> go.Figure:
    """Plots the structure of the lattice"""
    pass