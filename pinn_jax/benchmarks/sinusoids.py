# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from jax import numpy as jnp

from scipy.integrate import solve_ivp

from typing import Union, Tuple


def get_ode_parameters(u0: np.ndarray, omega: np.ndarray, t: Union[np.ndarray, None],
                       t_eval: Union[np.ndarray, None]) -> Tuple[jnp.ndarray, Union[None, jnp.ndarray]]:
    matrix = np.zeros((2, 2))
    matrix[0, 1] = -omega[0]
    matrix[1, 0] = omega[1]

    if t_eval is not None:
        u = jnp.array(solve_ivp(lambda _, _u: -matrix @ _u, (t[0], t[-1]), u0.ravel(), t_eval=t_eval).y.T)
    else:
        u = None

    return jnp.array(matrix), u


def test():
    def du(_, _u):
        return np.array([_u[1], -_u[0]])
    t_max = 4*np.pi
    t_eval = np.linspace(0, t_max, num=100)
    u0 = np.array([0.0, np.pi/2])

    import seaborn as sns
    sns.set_style('darkgrid')

    u = solve_ivp(du, (0.0, t_max), u0, t_eval=t_eval).y
    pd.DataFrame(u.T, index=t_eval).plot(legend=None)
    plt.ylabel('u')
    plt.xlabel('t')
    plt.waitforbuttonpress()


if __name__ == '__main__':
    test()
