# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp
from flax.core.frozen_dict import FrozenDict

from pinn_jax.derivatives import get_batch_jacobian, get_batch_hessian, get_batch_jerk

from typing import Callable


def get_ns_2d_ss_pde(u_hat: Callable, config: dict) -> Callable:
    """
    steady-state incompressible flow navier-stokes equations with two spatial variables
    - adapted from: https://github.com/lululxvi/deepxde/issues/80
    - see also: https://docs.juliahub.com/AdFem/xxjw7/0.1.1/SteadyStateNavierStokes/

    basic form of time-independent navier-stokes equation. adapted from: https://github.com/lululxvi/deepxde/issues/80

    uses input variables

    - ``x, y``

    uses state variables:

    - ``u``: ``x``-component of velocity
    - ``v``: ``y``-component of velocity
    - ``p``: pressure

    the network defines a map

    - ``x, y -> u, v, p``

    solves the equations:

        ``(d_x u) + (d_y v) + eps p = 0``

        ``u (d_x u) + v (d_y u) + (1/rho) * (d_x p) - nu * ((d_xx u) + (d_yy u) = 0``

        ``u (d_x v) + v (d_y v) + (1/rho) * (d_y p) - nu * ((d_xx v) + (d_yy v) = 0``

    where ``d`` is the partial derivative operator, and ``eps``, ``rho``, and ``nu`` are parameters supplied in the
    ``config`` argument
    """
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    assert 'eps' in config.keys() and 'rho' in config.keys() and 'nu' in config.keys(), \
        '`config` should have values for `eps`, `rho`, and `nu`!!!'

    def navier_stokes(params: FrozenDict, points: jnp.ndarray) -> dict:
        # calculate everything
        fields = u_hat(params, points)  # n_batch x n_output
        jacobian = batch_jacobian(params, points)  # n_batch x n_output x n_input
        hessian = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input
        # the issue here is that we're calculating a lot of derivatives we don't end up using

        # extract out state variables and derivatives
        u, v, p = fields[:, 0], fields[:, 1], fields[:, 2]

        u_x, u_y = jacobian[:, 0, 0], jacobian[:, 0, 1]
        v_x, v_y = jacobian[:, 1, 0], jacobian[:, 1, 1]
        p_x, p_y = jacobian[:, 2, 0], jacobian[:, 2, 1]

        u_xx, u_yy = hessian[:, 0, 0, 0], hessian[:, 0, 1, 1]
        v_xx, v_yy = hessian[:, 1, 0, 0], hessian[:, 1, 1, 1]

        continuity = u_x + v_y + config['eps'] * p
        x_momentum = u * u_x + v * u_y + 1. / config['rho'] * p_x - config['nu'] * (u_xx + u_yy)
        y_momentum = u * v_x + v * v_y + 1. / config['rho'] * p_y - config['nu'] * (v_xx + v_yy)

        return {'continuity': continuity, 'x_momentum': x_momentum, 'y_momentum': y_momentum}

    return navier_stokes


def get_ns_vv_2d_td_pde(u_hat: Callable, config: dict) -> Callable:
    """
    two-dimensional incompressible non-dimensionalized time-dependent navier-stokes equation, in velocity-vorticity
    formulation
    - adapted from: https://arxiv.org/abs/2203.07404 (section 5.3)
    - see also: https://doi.org/10.1016/0168-9274(95)00133-6 (eq. 9 in section 5)
    - see also: https://doi.org/10.1016/S0169-5983(99)00020-9

    uses input variables:

    - ``x, y, t``

    uses state variables:
    - ``u``: ``x``-component of velocity
    - ``v``: ``y``-component of velocity

    the network defines a map

    - ``x, y, t -> u, v``

    and solves the equations:

        ``(partial_t w) + u (d_x u) + v (d_y v) = (1/Re) ((d_xx w) + (d_yy w))``

        ``(d_x u) + (d_y v) = 0``

    where ``d`` is the partial derivative operator ``w`` is vorticity (curl of ``(u, v)``), and ``Re > 0`` is the
    reynolds number

    see here for how the equation in terms of u, v, and w is broken down into only involving w:
    https://jhuapl.box.com/s/cxvdw1ecpyz0hj2lkpcot8y5pxke76kx

    note: using the `PeriodicMLPxyt` will introduce an extra singleton dimension when evaluating batched derivatives. to
    mitigate this, we `squeeze` everything. this is generally okay because `n_output, n_input > 1`, but this will
    produce bad behavior when `n_batch` is 1.
    """
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)
    batch_jerk = get_batch_jerk(u_hat)

    assert 'Re' in config.keys(), "`config` must specify the reynolds number `Re`!!!"

    # index references for accessing array subsets
    _u, _v = 0, 1
    _x, _y, _t = 0, 1, 2

    def navier_stokes(params: FrozenDict, points: jnp.ndarray) -> dict:
        # calculate everything
        fields = u_hat(params, points)  # n_batch x n_output
        jacobian = batch_jacobian(params, points).squeeze()  # n_batch x n_output x n_input
        hessian = batch_hessian(params, points).squeeze()  # n_batch x n_output x n_input x n_input
        jerk = batch_jerk(params, points).squeeze()  # n_batch x n_output x n_input x n_input x n_input

        # extract out state variables and derivatives
        u, v = fields[:, _u], fields[:, _v]
        u_x, v_y = jacobian[:, _u, _x], jacobian[:, _v, _y]

        w_x = hessian[:, _v, _x, _x] - hessian[:, _u, _x, _y]  # w_x = v_xx - u_xy
        w_y = hessian[:, _v, _x, _y] - hessian[:, _u, _y, _y]  # w_y = v_xy - u_yy
        w_t = hessian[:, _v, _x, _t] - hessian[:, _u, _y, _t]  # w_t = v_xt - u_yt

        w_xx = jerk[:, _v, _x, _x, _x] - jerk[:, _u, _x, _y, _y]  # v_xxx - u_xyy
        w_yy = jerk[:, _v, _x, _y, _y] - jerk[:, _u, _y, _y, _y]  # v_xyy - u_yyy

        momentum = w_t + (u * w_x + v * w_y) - 1. / config['Re'] * (w_xx + w_yy)
        continuity = u_x + v_y

        return {'continuity': continuity, 'momentum': momentum}

    return navier_stokes


def get_ns_sf_2d_td_pde(u_hat: Callable, config: dict) -> Callable:
    """
    two-dimensional time-dependent navier-stokes, in the streamfunction formulation
    - based on: https://arxiv.org/abs/2001.04536

    uses input variables

    - ``x, y, t``

    uses state variables:

    - ``u``: x-component of velocity
    - ``v``: y-component of velocity
    - ``p``: pressure

    the network defines a map

    - ``x, y, t -> psi, p``

    where ``psi`` is the streamfunction, and the velocity is given by its curl

    solves the system:

        `` d_yt psi + (d_y psi) (d_xy psi) - (d_x psi) (d_yy psi) + d_x p - (1 / Re) (d_xxy psi + d_yyy psi) = 0``

        ``-d_xt psi - (d_y psi) (d_xx psi) + (d_x psi) (d_xy psi) + d_y p + (1 / Re) (d_xxx psi + d_xyy psi) = 0``

    where ``d`` is the partial derivative operator, and ``Re`` is the reynolds number

    see here for more detailed derivation: https://jhuapl.app.box.com/file/1066944819546
    """

    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)
    batch_jerk = get_batch_jerk(u_hat)

    assert 'Re' in config.keys(), "`config` must specify the reynolds number `Re`!!!"

    # index references for (x, y, t) -> (psi, p)
    x, y, t = 0, 1, 2
    psi, p = 0, 1
    Re = config['Re']

    def navier_stokes(params: FrozenDict, points: jnp.ndarray) -> dict:
        # calculate basic terms
        jacobian = batch_jacobian(params, points)  # n_batch x n_output x n_input
        hessian = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input
        jerk = batch_jerk(params, points)  # n_batch x n_output x n_input x n_input x n_input

        # extract out derivatives
        psi_x, psi_y = jacobian[:, psi, x], jacobian[:, psi, y]
        p_x, p_y = jacobian[:, p, x], jacobian[:, p, y]

        psi_xx, psi_xy, psi_yy = hessian[:, psi, x, x], hessian[:, psi, x, y], hessian[:, psi, y, y]
        psi_xt, psi_yt = hessian[:, psi, x, t], hessian[:, psi, y, t]

        psi_xxx, psi_yyy = jerk[:, psi, x, x, x], jerk[:, psi, y, y, y]
        psi_xxy, psi_xyy = jerk[:, psi, x, x, y], jerk[:, psi, x, y, y]

        # calculate residuals
        x_momentum = psi_yt + psi_y * psi_xy - psi_x * psi_yy + p_x - 1. / Re * (psi_xxy + psi_yyy)
        y_momentum = - psi_xt - psi_y * psi_xx + psi_x * psi_xy + p_y + 1. / Re * (psi_xxx + psi_xyy)

        return {'x_momentum': x_momentum, 'y_momentum': y_momentum}

    return navier_stokes


def get_ns_sf_2d_ss_pde(u_hat: Callable, config: dict) -> Callable:
    """
    two-dimensional steady-state navier-stokes, in the streamfunction formulation
    - based on: https://arxiv.org/abs/2001.04536

    uses input variables

    - ``x, y``

    uses state variables:

    - ``u``: x-component of velocity
    - ``v``: y-component of velocity
    - ``p``: pressure

    the network defines a map

    - ``x, y -> psi, p``

    where ``psi`` is the streamfunction, and the velocity is given by its curl

    solves the system:

        ``  (d_y psi) (d_xy psi) - (d_x psi) (d_yy psi) + d_x p - (1 / Re) (d_xxy psi + d_yyy psi) = 0``

        `` -(d_y psi) (d_xx psi) + (d_x psi) (d_xy psi) + d_y p + (1 / Re) (d_xxx psi + d_xyy psi) = 0``

    where ``d`` is the partial derivative operator, and ``Re`` is the reynolds number

    see here for more detailed derivation: https://jhuapl.app.box.com/file/1066944819546
    """

    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)
    batch_jerk = get_batch_jerk(u_hat)

    assert 'Re' in config.keys(), "`config` must specify the reynolds number `Re`!!!"

    # index references for (x, y) -> (psi, p)
    x, y = 0, 1
    psi, p = 0, 1
    Re = config['Re']

    def navier_stokes(params: FrozenDict, points: jnp.ndarray) -> dict:
        # calculate basic terms
        jacobian = batch_jacobian(params, points)  # n_batch x n_output x n_input
        hessian = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input
        jerk = batch_jerk(params, points)  # n_batch x n_output x n_input x n_input x n_input

        # extract out derivatives
        psi_x, psi_y = jacobian[:, psi, x], jacobian[:, psi, y]
        p_x, p_y = jacobian[:, p, x], jacobian[:, p, y]

        psi_xx, psi_xy, psi_yy = hessian[:, psi, x, x], hessian[:, psi, x, y], hessian[:, psi, y, y]

        psi_xxx, psi_yyy = jerk[:, psi, x, x, x], jerk[:, psi, y, y, y]
        psi_xxy, psi_xyy = jerk[:, psi, x, x, y], jerk[:, psi, x, y, y]

        # calculate residuals
        x_momentum = psi_y * psi_xy - psi_x * psi_yy + p_x - 1. / Re * (psi_xxy + psi_yyy)
        y_momentum = - psi_y * psi_xx + psi_x * psi_xy + p_y + 1. / Re * (psi_xxx + psi_xyy)

        return {'x_momentum': x_momentum, 'y_momentum': y_momentum}

    return navier_stokes


def get_uv_fn(u_hat: Callable) -> Callable:
    """construct function to output the velocity vector from the streamfunction"""
    batch_jacobian = get_batch_jacobian(u_hat)

    # index references for (x, y, t) -> (psi, p)
    x, y, _ = 0, 1, 2
    psi, _ = 0, 1

    def uv_fn(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        """(u, v) = (partial_y psi, -partial_x psi)"""
        jacobian = batch_jacobian(params, points)  # n_batch x n_output x n_input
        u, v = jacobian[:, psi, y], -jacobian[:, psi, x]

        return jnp.hstack([u[:, None], v[:, None]])

    return uv_fn


def get_ns_3d_td_pde(u_hat: Callable, config: dict) -> Callable:
    """
    time-dependent incompressible flow navier-stokes equations with three spatial variables
    - adapted from: https://github.com/lululxvi/deepxde/issues/80
    - see also: https://docs.juliahub.com/AdFem/xxjw7/0.1.1/SteadyStateNavierStokes/

    uses input variables

    - ``x, y, z, t``

    uses state variables:

    - ``u``: ``x``-component of velocity
    - ``v``: ``y``-component of velocity
    - ``w``: ``z``-component of velocity
    - ``p``: pressure

    the network defines a map

    - ``x, y, z, t -> u, v, w, p``

    solves the equations:

        ``(d_x u) + (d_y v) + (d_z w) + eps p = 0``

        ``(d_t u) + u (d_x u) + v (d_y u) + w (d_z u) + (1/rho) * (d_x p) - nu * ((d_xx u) + (d_yy u) + (d_zz u)) = 0``

        ``(d_t v) + u (d_x v) + v (d_y v) + w (d_z v) + (1/rho) * (d_y p) - nu * ((d_xx v) + (d_yy v) + (d_zz v)) = 0``

        ``(d_t w) + u (d_x v) + v (d_y v) + w (d_z w) + (1/rho) * (d_z p) - nu * ((d_xx w) + (d_yy w) + (d_zz w)) = 0``

    where ``d`` is the partial derivative operator, and ``eps``, ``rho``, and ``nu`` are parameters supplied in the
    ``config`` argument
    """
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    assert 'eps' in config.keys() and 'rho' in config.keys() and 'nu' in config.keys() and \
           'pressure_gradient' in config.keys(), \
        '`config` should have values for `eps`, `rho`, `nu`, and `pressure_gradient`!!!'

    def navier_stokes(params: FrozenDict, points: jnp.ndarray) -> dict:
        # calculate everything
        fields = u_hat(params, points)  # n_batch x n_output
        jacobian = batch_jacobian(params, points)  # n_batch x n_output x n_input
        hessian = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input
        # the issue here is that we're calculating a lot of derivatives we don't end up using

        # extract out state variables and derivatives
        u, v, w, p = fields[:, 0], fields[:, 1], fields[:, 2], fields[:, 3]

        u_x, u_y, u_z, u_t = jacobian[:, 0, 0], jacobian[:, 0, 1], jacobian[:, 0, 2], jacobian[:, 0, 3]
        v_x, v_y, v_z, v_t = jacobian[:, 1, 0], jacobian[:, 1, 1], jacobian[:, 1, 2], jacobian[:, 1, 3]
        w_x, w_y, w_z, w_t = jacobian[:, 2, 0], jacobian[:, 2, 1], jacobian[:, 2, 2], jacobian[:, 2, 3]
        p_x, p_y, p_z = jacobian[:, 3, 0], jacobian[:, 3, 1], jacobian[:, 3, 2]

        u_xx, u_yy, u_zz = hessian[:, 0, 0, 0], hessian[:, 0, 1, 1], hessian[:, 0, 2, 2]
        v_xx, v_yy, v_zz = hessian[:, 1, 0, 0], hessian[:, 1, 1, 1], hessian[:, 1, 2, 2]
        w_xx, w_yy, w_zz = hessian[:, 2, 0, 0], hessian[:, 2, 1, 1], hessian[:, 2, 2, 2]

        continuity = u_x + v_y + + w_z + config['eps'] * p
        x_momentum = u_t + u * u_x + v * u_y + w * u_z + 1. / config['rho'] * p_x - config['nu'] * (u_xx + u_yy + u_zz)
        y_momentum = v_t + u * v_x + v * v_y + w * v_z + 1. / config['rho'] * p_y - config['nu'] * (v_xx + v_yy + v_zz)
        z_momentum = w_t + u * w_x + v * w_y + w * w_z + 1. / config['rho'] * p_z - config['nu'] * (w_xx + w_yy + w_zz)

        x_pressure_differential = p_x - config['pressure_gradient']

        return {'continuity': continuity, 'x_momentum': x_momentum, 'y_momentum': y_momentum, 'z_momentum': z_momentum,
                'x_pressure_differential': x_pressure_differential}

    return navier_stokes
