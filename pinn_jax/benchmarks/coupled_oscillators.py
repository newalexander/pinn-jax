# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.


import numpy as np
import pandas as pd

from jax import numpy as jnp
from scipy.integrate import solve_ivp

from typing import Union, Tuple


def get_ode_parameters(u0: np.ndarray, du0: np.ndarray, omega: np.ndarray, t: np.ndarray, epsilon: float, chi: float,
                       t_eval: Union[np.ndarray, None], random_state=0) -> Tuple[jnp.ndarray, Union[None, jnp.ndarray]]:
    np.random.seed(random_state)
    n_oscillators = u0.shape[0]
    coupling_matrix = omega * np.eye(n_oscillators, n_oscillators)
    for i in range(n_oscillators):
        for j in range(n_oscillators):
            if i == j:
                continue
            coupling_matrix[i, j] = np.random.choice([0.0, chi], p=[1.0-epsilon, epsilon])

    if t_eval is not None:
        # standard way to convert 2nd order ODE to 2x as many first orders:
        # https://math.stackexchange.com/questions/3732243/how-to-solve-vector-second-order-differential-equation
        joint_array = np.zeros((2*n_oscillators, 2*n_oscillators))
        joint_array[:n_oscillators, n_oscillators:] = -coupling_matrix
        joint_array[n_oscillators:, :n_oscillators] = np.eye(n_oscillators, n_oscillators)
        _u0 = np.concatenate([du0, u0])
        u = jnp.array(solve_ivp(lambda _, _u: (joint_array @ _u),
                                (t[0], t[-1]), _u0, t_eval=t_eval).y.T)
    else:
        u = None

    return coupling_matrix, u


def example():
    n_oscillators = 20
    coupling_value = 2.0 / n_oscillators  # this is maybe too strong and will make things blow up for high coupling_prob
    coupling_prob = 0.5
    diagonal_value = np.array([1.0])
    t_min, t_max = 0.0, 2 * np.pi  # full period
    t = np.linspace(t_min, t_max, num=50)
    u0 = np.random.uniform(-np.pi / 8, np.pi / 8, size=(n_oscillators,))
    du0 = np.random.uniform(-np.pi / 4, np.pi / 4, size=(n_oscillators,))
    matrix, u = get_ode_parameters(u0, du0, diagonal_value, t, coupling_prob, coupling_value, t)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')

    pd.DataFrame(np.array(u)[:, n_oscillators:], index=t).plot(legend=False)
    plt.xlabel('t')
    plt.ylabel('u')
    plt.waitforbuttonpress()
    print(np.array(u).shape)
