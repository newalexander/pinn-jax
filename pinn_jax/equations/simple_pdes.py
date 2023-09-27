# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp
from flax.core.frozen_dict import FrozenDict

from pinn_jax.derivatives import get_batch_jacobian, get_batch_hessian, get_batch_snap

from typing import Callable, Tuple


def get_burgers(u_hat: Callable, nu: float) -> Callable:
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    def burgers_eqn(params: FrozenDict, points: jnp.ndarray) -> dict:
        u = u_hat(params, points).squeeze()
        hessian = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input
        jacobian = batch_jacobian(params, points)  # n_batch x n_output x n_input

        du_xx = hessian[:, 0, 0, 0]
        du_t = jacobian[:, 0, 1]
        du_x = jacobian[:, 0, 0]

        residual = du_t + u * du_x - nu * du_xx
        return {'burgers_eqn': residual}

    return burgers_eqn


def get_wave_eqn(u_hat: Callable, c: float, v: float) -> Callable:
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    def wave_eqn(params: FrozenDict, points: jnp.ndarray) -> dict:
        hessians = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input
        jacobians = batch_jacobian(params, points)  # n_batch x n_output x n_input
        residual = hessians[:, 0, 0, 0] - 1. / c**2 * hessians[:, 0, 1, 1] - v * jacobians[:, 0, 1]
        return {'wave': (residual**2).mean()}

    return wave_eqn


def get_allen_cahn(u_hat: Callable, a: float, b: float, c: float) -> Callable:
    """
    u_hat is a map (x, t) -> u

    equation: D(x, t; u) = partial_t u - a u_xx + b u^3 - c u = 0,   (x, t) in (-1, 1) x (0, 1)

    note: using the `PeriodicMLP` will introduce an extra singleton dimension when evaluating batched derivatives. to
    mitigate this, we `squeeze` everything. this is generally okay because `n_input > 1`, but this will
    produce bad behavior when `n_batch` is 1.
    """
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    def allen_cahn(params: FrozenDict, points: jnp.ndarray) -> dict:
        u = u_hat(params, points).squeeze()
        jacobian = batch_jacobian(params, points).squeeze()  # n_batch x n_input
        hessian = batch_hessian(params, points).squeeze()  # n_batch x n_input x n_input

        u_t = jacobian[:, 1]
        u_xx = hessian[:, 0, 0]

        return {'allen_cahn': u_t - a * u_xx + b * u**3 - c * u}

    return allen_cahn


def get_kuramoto_sivashinsky(u_hat: Callable, alpha: float, beta: float, gamma: float) -> Callable:
    """u_hat is a map (x, t) -> u"""
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)
    batch_snap = get_batch_snap(u_hat)

    def kuramoto_sivashinsky(params: FrozenDict, points: jnp.ndarray) -> dict:
        u = u_hat(params, points).squeeze()
        j = batch_jacobian(params, points).squeeze()  # n_batch x n_input
        h = batch_hessian(params, points).squeeze()   # n_batch x n_input x n_input
        s = batch_snap(params, points).squeeze()      # n_batch x n_input x n_input x n_input x n_inputs

        u_x, u_t = j[:, 0], j[:, 1]
        u_xx = h[:, 0, 0]
        u_xxxx = s[:, 0, 0, 0, 0]

        return {'kuramoto_sivashinksy': u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx}

    return kuramoto_sivashinsky


def get_convection_diffusion(u_hat: Callable, alpha: float) -> Tuple[Callable, Callable]:
    """
    u_hat is a map x -> u

    u_x + alpha u_xx = 0,  x in (0, 1)

    with the boundary conditions u(0) = 0.5 and u(1) = -0.5, has the analytic solution

    exp(-x / alpha) / (1 - exp(-1 / alpha)) - 0.5
    """
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    def convection_diffusion(params: FrozenDict, points: jnp.ndarray) -> dict:
        j = batch_jacobian(params, points).squeeze()  # n_batch
        h = batch_hessian(params, points).squeeze()  # n_batch

        return {'convection_diffusion': j + alpha * h}

    def u_fn(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-points[:, 0] / alpha) / jnp.exp(-1. / alpha)

    return convection_diffusion, u_fn


def get_helmholtz(u_hat: Callable, k: float, a_1: float, a_2: float) -> Tuple[Callable, Callable]:
    """
    u_hat is a map (x, y) -> u

    Delta u + k^2 u - q = 0,  (x, y) in [-1, 1] x [-1, 1]

    q = - (a_1 pi)^2 sin(a_1 pi x) sin(a_2 pi y) - (a_2 pi)^2 sin(a_1 pi x) sin(a_2 pi y) + k^2 sin(a_1 pi x) sin(a_2 pi y)

    has the analytic solution u = sin(a_1 pi x) sin(a_2 pi y)
    """
    def q_fn(points: jnp.ndarray) -> jnp.ndarray:
        q = jnp.sin(a_1 * jnp.pi * points[:, 0]) * jnp.sin(a_2 * jnp.pi * points[:, 1])
        return (k**2 - a_1**2 * jnp.pi**2 - a_2**2 * jnp.pi**2) * q

    batch_hessian = get_batch_hessian(u_hat)

    def helmholtz(params: FrozenDict, points: jnp.ndarray) -> dict:
        q = q_fn(points)  # n_batch
        u = u_hat(params, points).squeeze()  # n_batch
        h = batch_hessian(params, points).squeeze()  # n_batch x n_input x n_input

        return {'helmholtz': h[:, 0, 0] + h[:, 1, 1] + k**2 * u - q}  # , 'u_xx': h[:,0,0], 'u_yy': h[:,1,1], 'q': q}

    def u_fn(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(a_1 * jnp.pi * points[:, 0]) * jnp.sin(a_2 * jnp.pi * points[:, 1])

    return helmholtz, u_fn


def get_reaction_diffusion(u_hat: Callable, nu: float, rho: float) -> Callable:
    """
    u_hat is a map (x, t) -> u

    u_t - nu u_xx - rho u (1 - u) = 0

    when `u` is periodic in `x`, it has a solution that can be calculated exactly for each time step

    note: using the `PeriodicMLP` will introduce an extra singleton dimension when evaluating batched derivatives. to
    mitigate this, we `squeeze` everything. this will produce bad behavior `n_batch` is 1.
    """
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    def reaction_diffusion(params: FrozenDict, points: jnp.ndarray) -> dict:
        u = u_hat(params, points).squeeze()  # n_batch
        jacobian = batch_jacobian(params, points).squeeze()  # n_batch x n_input
        hessian = batch_hessian(params, points).squeeze()  # n_batch x n_input x n_input

        u_t = jacobian[:, 1]
        u_xx = hessian[:, 0, 0]

        return {'reaction_diffusion': u_t - nu * u_xx - rho * u * (1. - u)}

    return reaction_diffusion


def get_brusselator(uv_hat: Callable, d_0: float, d_1: float, a: float, b: float) -> Callable:
    """brusselator reaction-diffusion system presented in: https://doi.org/10.1016/j.jcp.2023.112008
    u_t = d_0 (u_xx + u_yy) + a - (1 + b) u + v u^2
    v_t = d_1 (v_xx + v_yy) + b u - v u^2

    note: using the `PeriodicMLP` will introduce an extra singleton dimension when evaluating batched derivatives. to
    mitigate this, we `squeeze` everything. this will produce bad behavior `n_batch` is 1.
    """
    batch_jacobian = get_batch_jacobian(uv_hat)
    batch_hessian = get_batch_hessian(uv_hat)

    (_x, _y, _t), (_u, _v) = (0, 1, 2), (0, 1)

    def brusselator(params: FrozenDict, points: jnp.ndarray) -> dict:
        uv = uv_hat(params, points).squeeze()
        jacobian = batch_jacobian(params, points).squeeze()  # n_batch x n_output x n_input
        hessian = batch_hessian(params, points).squeeze()  # n_batch x n_output x n_input x n_input

        u, v = uv[:, _u], uv[:, _v]
        u_t, v_t = jacobian[:, _u, _t], jacobian[:, _v, _x]
        u_xx, u_yy = hessian[:, _u, _x, _x], hessian[:, _u, _y, _y]
        v_xx, v_yy = hessian[:, _v, _x, _x], hessian[:, _v, _y, _y]

        return {'u': -u_t + d_0 * (u_xx + u_yy) + a - (1. + b) * u + v * u**2,
                'v': -v_t + d_1 * (v_xx + v_yy) + b * u - v * u**2}

    return brusselator
