# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import numpy as np

from jax import numpy as jnp
from scipy.integrate import solve_ivp

from typing import Union, Callable, Tuple


def get_ode_parameters(ic_fn: Callable, n_space: int, t: np.ndarray, t_eval: Union[np.ndarray, None],
                       x_min=0.0, x_max=1.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Union[None, jnp.ndarray]]:
    xs_np = np.linspace(x_min, x_max, num=n_space)
    xs = jnp.array(xs_np)

    A_np = (-2 * np.eye(n_space) + np.eye(n_space, k=1) + np.eye(n_space, k=-1)) * (n_space + 1) ** 2
    A = jnp.array(A_np)

    phi_np = np.zeros((n_space,))
    phi_np[0] = 1.0
    phi_np[-1] = 1.0
    phi_np = phi_np * (n_space - 1) ** 2
    phi = jnp.array(phi_np)

    if t_eval is not None:
        u = jnp.array(solve_ivp(lambda _, _u: A_np @ _u + phi_np, (t[0], t[-1]), ic_fn(xs_np), t_eval=t_eval).y.T)
    else:
        u = None

    return A, phi, xs, u
