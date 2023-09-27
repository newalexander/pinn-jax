# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.


"""this module has functions for implementing features of the 2D steady-state channel flow problem."""

import numpy as np

from pinn_jax.bcs import ConstantDirichletBC, FunctionDirichletBC

X_MIN, X_MAX = 0., 1.
Y_MIN, Y_MAX = 0., 20.

"""
dumb illustration of room and BCs

             p=1
          |       |
          |       |
  u=0,v=0 |       |u=0,v=0
          |       |      
          ---------
           u=0,v=v_0(x)        
"""

"""
NS:

    (d_x u) + (d_y v) + eps p = 0``
    u (d_x u) + v (d_y u) + (1/rho) * (d_x p) - nu * ((d_xx u) + (d_yy u) = 0
    u (d_x v) + v (d_y v) + (1/rho) * (d_y p) - nu * ((d_xx v) + (d_yy v) = 0
"""


def boundary_left(x, on_boundary, x_min=X_MIN, y_max=Y_MAX):
    return on_boundary and np.isclose(x[0], x_min) and not np.isclose(x[1], y_max)


def boundary_top(x, on_boundary, y_max=Y_MAX):
    return on_boundary and np.isclose(x[1], y_max)


def boundary_right(x, on_boundary, x_max=X_MAX):
    return on_boundary and np.isclose(x[0], x_max) and not np.isclose(x[1], 1.0)


def boundary_bottom(x, on_boundary, y_min=Y_MIN):
    return on_boundary and np.isclose(x[1], y_min)


def get_bcs(geom, bc_weights=None, default_weight=1., default_mu=1., default_c=-8., x_max=X_MAX):
    """boundary conditions based on poiseuille flow for 2d channel flow. `default_mu` is the default fluid viscosity,
    and `default_c` is chosen such that, in the middle of the stream, the x-velocity is 1."""
    # TODO: add units to this
    if bc_weights is None:
        bc_weights = {}

    def v_bottom(x: np.ndarray) -> np.ndarray:
        """bottom BC for y-velocity: -c * (1/2 mu) * x * (X_MAX - x)"""
        return -default_c * 1./(2.*default_mu) * x[:, 0].reshape([-1, 1])*(x_max - x[:, 0].reshape([-1, 1]))
    bcs = {
        # u BCs
        'u_bottom': ConstantDirichletBC(geom, 0.0, boundary_bottom, component=0),
        'u_left': ConstantDirichletBC(geom, 0.0, boundary_left, component=0),
        'u_right': ConstantDirichletBC(geom, 0.0, boundary_right, component=0),
        # v BCs
        'v_bottom': FunctionDirichletBC(geom, v_bottom, boundary_bottom, component=1),
        'v_left': ConstantDirichletBC(geom, 0.0, boundary_left, component=1),
        'v_right': ConstantDirichletBC(geom, 0.0, boundary_right, component=1),
        # p BC
        'p_top': ConstantDirichletBC(geom, 1.0, boundary_top, component=2)
    }
    component_weights = {}
    for bc_name in bcs:
        component_weights[bc_name] = bc_weights.get(bc_name, default_weight)
    return bcs, component_weights
