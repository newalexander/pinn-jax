# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp
from flax.core.frozen_dict import FrozenDict

from pinn_jax.derivatives import get_batch_jacobian, get_batch_hessian

from typing import Callable


def get_sinusoid_ode(u_hat: Callable, matrix: jnp.ndarray) -> Callable:
    batch_jacobian = get_batch_jacobian(u_hat)

    def sinusoid_ode(params: FrozenDict, points: jnp.ndarray):
        u = u_hat(params, points)
        u_t = batch_jacobian(params, points).squeeze()
        return u_t - u @ matrix

    return sinusoid_ode


def get_heat_ode(u_hat: Callable, matrix: jnp.ndarray, offset: jnp.ndarray):
    batch_jacobian = get_batch_jacobian(u_hat)

    def heat_ode(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        """
        - `points` is `(n_time, 1)`
        - `matrix` is `(n_space, n_space)`
        - `offset` is `(n_space, )`

        - `u` is `(n_time, n_space)`
        - `u_t` is `(n_time, n_space)`

        - `u[n_t, n_x]` is the predicted value
        """
        u = u_hat(params, points)
        u_t = batch_jacobian(params, points).squeeze()

        return u_t - (u @ matrix).squeeze() - offset

    return heat_ode


def get_oscillator_ode(u_hat: Callable, matrix: jnp.ndarray) -> Callable:
    """`matrix` is `n_oscillators x n_oscillators`, with the `i`th diagonal representing the frequency of the `i`th
    oscillator, and the `[i,j]`th entry representing the coupling of the `i`th oscillator onto the `j`th one

    in the limiting case when `matrix` is diagonal, reduces down to `partial_tt u = - k u` for each component
    """

    batch_hessian = get_batch_hessian(u_hat)

    def oscillator_ode(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        """
        - `points` is `(n_time, 1)`
        - `matrix` is `(n_space, n_space)`
        - `offset` is `(n_space, )`

        - `u` is `(n_time, n_space)`
        - `u_tt` is `(n_time, n_space)`

        - `u[n_t, n_x]` is the predicted value
        """
        u = u_hat(params, points)
        u_tt = batch_hessian(params, points).squeeze()

        return u_tt + (u @ matrix).squeeze()

    return oscillator_ode