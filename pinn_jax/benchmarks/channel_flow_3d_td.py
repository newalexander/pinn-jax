# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.


import numpy as np

from jax import numpy as jnp

from pinn_jax.bcs import ConstantDirichletBC, FunctionDirichletBC, IC

X_MIN, X_MAX = 0., 8.*jnp.pi
Y_MIN, Y_MAX = 0., 2.
Z_MIN, Z_MAX = 0., 3.*jnp.pi


"""
see here for a diagram of what "x, y, z" and "top, bottom, left, right, front, back" are for the 3d channel flow problem
https://jhuapl.box.com/s/ujh14w80gwjoerh14eztta23yhi35ip6

in text form:
- left: 
  - `y == y_max` and `(x, y, z)` not near any `x` or `z` boundaries
- right:
  - `y == y_min` and `(x, y, y)` not near any `x` or `z` boundaries
- top: 
  - `z == z_max` and `(x, y, z)` not near any `x` or `y` boundaries
- bottom:
  - `z == z_min` and `(x, y, z)` not near any `x` or `y` boundaries
- front:
  - `x == x_max` and `(x, y, z)` not near any `y` or `z` boundaries
- back:
  - `x == x_min` and `(x, y, z)` not near any `y` or `z` boundaries
"""
# TODO: this should be saved as a png and added in a README as well

"""
NS:

    ``(d_x u) + (d_y v) + (d_z w) + eps p = 0``

    ``(d_t u) + u (d_x u) + v (d_y u) + w (d_z u) + (1/rho) * (d_x p) - nu * ((d_xx u) + (d_yy u) + (d_zz u)) = 0``

    ``(d_t v) + u (d_x v) + v (d_y v) + w (d_z v) + (1/rho) * (d_y p) - nu * ((d_xx v) + (d_yy v) + (d_zz v)) = 0``

    ``(d_t w) + u (d_x v) + v (d_y v) + w (d_z w) + (1/rho) * (d_z p) - nu * ((d_xx w) + (d_yy w) + (d_zz w)) = 0``
"""


def boundary_left(x, on_boundary, x_min=X_MIN, x_max=X_MAX, y_max=Y_MAX, z_min=Z_MIN, z_max=Z_MAX):
    return on_boundary and np.isclose(x[1], y_max) and not np.isclose(x[0], x_min) and not np.isclose(x[0], x_max) \
            and not np.isclose(x[2], z_min) and not np.isclose(x[2], z_max)


def boundary_top(x, on_boundary, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, z_max=Z_MAX):
    return on_boundary and np.isclose(x[2], z_max) and not np.isclose(x[0], x_min) and not np.isclose(x[0], x_max) \
            and not np.isclose(x[1], y_min) and not np.isclose(x[1], y_max)


def boundary_right(x, on_boundary, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, z_min=Z_MIN, z_max=Z_MAX):
    return on_boundary and np.isclose(x[1], y_min) and not np.isclose(x[0], x_min) and not np.isclose(x[0], x_max) \
                    and not np.isclose(x[2], z_min) and not np.isclose(x[2], z_max)


def boundary_bottom(x, on_boundary, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, z_min=Z_MIN):
    return on_boundary and np.isclose(x[2], z_min) and not np.isclose(x[0], x_min) and not np.isclose(x[0], x_max) \
            and not np.isclose(x[1], y_min) and not np.isclose(x[1], y_max)


def boundary_front(x, on_boundary, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, z_min=Z_MIN, z_max=Z_MAX):
    return on_boundary and np.isclose(x[0], x_max) and not np.isclose(x[1], y_min) and not np.isclose(x[1], y_max) \
            and not np.isclose(x[2], z_min) and not np.isclose(x[2], z_max)


def boundary_back(x, on_boundary, x_min=X_MIN, y_min=Y_MIN, y_max=Y_MAX, z_min=Z_MIN, z_max=Z_MAX):
    return on_boundary and np.isclose(x[0], x_min) and not np.isclose(x[1], y_min) and not np.isclose(x[1], y_max) \
            and not np.isclose(x[2], z_min) and not np.isclose(x[2], z_max)


def initial_u_generator(pressure_gradient, viscosity, channel_height):
    def initial_u(x):
        return (pressure_gradient*(1./(2. * viscosity)) * x[:, 1] * (channel_height - x[:, 1])).reshape([-1, 1])
    return initial_u


def initial_v(x):
    return jnp.zeros(x[:, 0].shape).reshape([-1, 1])


def initial_w(x):
    return jnp.zeros(x[:, 0].shape).reshape([-1, 1])


def initial_p_generator(pressure_gradient, channel_length):
    def initial_p(x):
        return (1. - pressure_gradient*channel_length + pressure_gradient * x[:, 0]).reshape([-1, 1])
    return initial_p


def on_time_boundary(_, on_initial):
    return on_initial


def get_bcs(geom, bc_weights=None, default_weight=1.0, mu=1.0, pressure_gradient=1.0, x_max=X_MAX, y_max=Y_MAX):
    if bc_weights is None:
        bc_weights = {}

    initial_u = initial_u_generator(pressure_gradient, mu, y_max)
    initial_p = initial_p_generator(pressure_gradient, x_max)
    bcs = {
        # u BCs
        'u_bottom': ConstantDirichletBC(geom, 1.0E-6, boundary_bottom, component=0),
        'u_left': ConstantDirichletBC(geom, 1.0E-6, boundary_left, component=0),
        'u_right': ConstantDirichletBC(geom, 1.0E-6, boundary_right, component=0),
        'u_back': FunctionDirichletBC(geom, initial_u, boundary_back, component=0),
        'u_top': ConstantDirichletBC(geom, 1.0E-6, boundary_top, component=0),
        'u_front':  FunctionDirichletBC(geom, initial_u, boundary_back, component=0),
        # v BCs
        'v_bottom': ConstantDirichletBC(geom, 1.0E-6, boundary_bottom, component=1),
        'v_left': ConstantDirichletBC(geom, 1.0E-6, boundary_left, component=1),
        'v_right': ConstantDirichletBC(geom, 1.0E-6, boundary_right, component=1),
        'v_back': ConstantDirichletBC(geom, 1.0E-6, boundary_back, component=1),
        'v_top': ConstantDirichletBC(geom, 1.0E-6, boundary_top, component=1),
        # w BCs
        'w_bottom': ConstantDirichletBC(geom, 1.0E-6, boundary_bottom, component=2),
        'w_left': ConstantDirichletBC(geom, 1.0E-6, boundary_left, component=2),
        'w_right': ConstantDirichletBC(geom, 1.0E-6, boundary_right, component=2),
        'w_back': ConstantDirichletBC(geom, 1.0E-6, boundary_back, component=2),
        'w_top': ConstantDirichletBC(geom, 1.0E-6, boundary_top, component=2),
        # w ICs
        'ic_u': IC(geom, initial_u, on_time_boundary, component=0),
        'ic_v': IC(geom, initial_v, on_time_boundary, component=1),
        'ic_w': IC(geom, initial_w, on_time_boundary, component=2),
        'ic_p': IC(geom, initial_p, on_time_boundary, component=3),

        # p BC
        'p_front': ConstantDirichletBC(geom, 1.0, boundary_front, component=3)
    }

    component_weights = {}
    for bc_name in bcs:
        component_weights[bc_name] = bc_weights.get(bc_name, default_weight)
    return bcs, component_weights
