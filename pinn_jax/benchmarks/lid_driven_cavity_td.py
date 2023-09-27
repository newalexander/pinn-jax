# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import numpy as np

from jax import numpy as jnp
from jax.nn import sigmoid
from flax.core.frozen_dict import FrozenDict

from pinn_jax.equations.navier_stokes import get_uv_fn
from pinn_jax.bcs import ConstantDirichletBC, ConstantIC, FunctionDirichletBC, PointSetBC
from pinn_jax.geometry import Rectangle, GeometryXTime, TimeDomain

from typing import Callable

"""
dumb illustration of domain and BCs

    u(x, t)=sigma(a t - b)
           v=0
        ---------
        |       |
  u,v=0 |       |u,v=0
        |       |
        ---------
          u,v=0

default time domain of (0, 10)
"""

DEFAULT_PSC_x_loc, DEFAULT_PSC_y_loc, DEFAULT_PSC_p, DEFAULT_PSC_n_t = 0.01, 0.01, 0., 128
DEFAULT_u_VALUE, DEFAULT_v_VALUE, DEFAULT_p_VALUE = 0., 0., 0.
DEFAULT_a_VALUE, DEFAULT_b_VALUE = 10., 10.  # u(x, t) = sigma(a t - b) on top of boundary


def get_u_top_func(a: float = DEFAULT_a_VALUE, b: float = DEFAULT_b_VALUE) -> Callable:
    def u_top_func(xyt: jnp.ndarray) -> jnp.ndarray:
        return sigmoid(a * xyt[:, [2]] - b)
    return u_top_func


def boundary_left(xyt, on_boundary):
    return on_boundary and np.isclose(xyt[0], 0.0) and not np.isclose(xyt[1], 1.0)


def boundary_top(xyt, on_boundary):
    return on_boundary and np.isclose(xyt[1], 1.0)


def boundary_right(xyt, on_boundary):
    return on_boundary and np.isclose(xyt[0], 1.0) and not np.isclose(xyt[1], 1.0)


def boundary_bottom(xyt, on_boundary):
    return on_boundary and np.isclose(xyt[1], 0.0)


def get_bcs(geom: GeometryXTime, u_top_func: Callable, bc_weights: dict = None, default_weight=1.0):
    """defined for a function (x, y, t) -> (u, v, p)"""
    if bc_weights is None:
        bc_weights = {}
    bcs = {
        # u BCs
        'u_left': ConstantDirichletBC(geom, 0.0, boundary_left, component=0),
        'u_top': FunctionDirichletBC(geom, u_top_func, boundary_top, component=0),
        'u_right': ConstantDirichletBC(geom, 0.0, boundary_right, component=0),
        'u_bottom': ConstantDirichletBC(geom, 0.0, boundary_bottom, component=0),
        # v BCs
        'v_left': ConstantDirichletBC(geom, 0.0, boundary_left, component=1),
        'v_top': ConstantDirichletBC(geom, 0.0, boundary_top, component=1),
        'v_right': ConstantDirichletBC(geom, 0.0, boundary_right, component=1),
        'v_bottom': ConstantDirichletBC(geom, 0.0, boundary_bottom, component=1),
    }
    component_weights = {}
    for bc_name in bcs:
        component_weights[bc_name] = bc_weights.get(bc_name, default_weight)
    return bcs, component_weights


def get_uvp_fn(u_hat: Callable) -> Callable:
    """return a function that performs the mapping (x, y, t) -> (psi, p) -> (u, v, p)"""
    uv_fn = get_uv_fn(u_hat)

    # reference indices for array slicing
    _, _p = 0, 1
    _u, _v = 0, 1

    def uvp_fn(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        fields = u_hat(params, points)  # n_batch x 2 = (psi, p)
        uv = uv_fn(params, points)  # n_batch x 2 = (u, v)

        return jnp.hstack([uv[:, [_u]], uv[:, [_v]], fields[:, [_p]]])

    return uvp_fn


def get_constant_ics(config, geom, ic_weights=None, default_weight=1.0):
    """defined for a function (x, y, t) -> (u, v, p)"""
    if ic_weights is None:
        ic_weights = {}
    ics = {'u_0': ConstantIC(geom, config.get('u_0_value', DEFAULT_u_VALUE), component=0),
           'v_0': ConstantIC(geom, config.get('v_0_value', DEFAULT_v_VALUE), component=1),
           'p_0': ConstantIC(geom, config.get('p_0_value', DEFAULT_p_VALUE), component=2)}
    component_weights = {key: ic_weights.get(key, default_weight) for key in ics.keys()}
    return ics, component_weights


def get_pscs(config, psc_weights=None, default_weight=1.):
    """pin the pressure to a fixed value at a location (x, y) for all time t.
    defined for a function (x, y, t) -> (psi, p)
    """
    if psc_weights is None:
        psc_weights = {}

    x_loc, y_loc = config.get('psc_x_loc', DEFAULT_PSC_x_loc), config.get('psc_y_loc', DEFAULT_PSC_y_loc)
    psc_value, n_t = config.get('psc_value', DEFAULT_PSC_p), config.get('psc_n_t', DEFAULT_PSC_n_t)
    points = np.stack([[x_loc, y_loc, t] for t in np.linspace(config['t_min'], config['t_max'], num=n_t)])

    pscs = {'p_psc': PointSetBC(points, psc_value, component=1)}
    component_weights = {'p_psc': psc_weights.get('p_psc', default_weight)}
    return pscs, component_weights, {'p_psc': points}


def get_geom_ss():
    return Rectangle((0.0, 0.0), (1.0, 1.0))


def get_geom_td(t_min, t_max):
    geometry = get_geom_ss()
    time_domain = TimeDomain(t_min, t_max)
    return GeometryXTime(geometry=geometry, timedomain=time_domain)


def get_bcs_and_weights_and_names(geom, default_weight=1.0):
    # Assuming uniform weights
    bcs, component_weights = get_bcs(geom)
    residual_names = ['continuity', 'x_momentum', 'y_momentum']
    component_weights['continuity'] = default_weight
    component_weights['x_momentum'] = default_weight
    component_weights['y_momentum'] = default_weight

    return bcs, component_weights, residual_names
