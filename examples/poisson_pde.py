"""
we're solving poisson's equation on a rectangular domain:

    D(x, y; u) = partial_xx u + partial_yy u + sin(pi x) sin(pi y),   (x, y) in Omega = [0, 1] x [0, 1]
    L(y; u) = u(0, y) = 0
    R(y; u) = u(1, y) = 0
    B(x; u) = u(x, 0) = 0
    T(x; u) = u(x, 1) = 0

let u: R^2 -> R be a neural network. we minimize the loss function

    sum_{(x, y) in Omega} | D(x, y; u) |^2 +
        sum_{x in [0, 1]} (| B(x; u) |^2 + | T(x; u) |^2 ) +
        sum_{y in [0, 1]} (| L(y; u) |^2 + | R(y; u) |^2 )

the learned solution can be compared to the known analytic solution:

    u(x, y) = sin(pi x) sin(pi y) / (2 pi^2)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jax import numpy as jnp
from flax.core.frozen_dict import FrozenDict

from pinn_jax.bcs import ConstantDirichletBC
from pinn_jax.derivatives import get_batch_hessian
from pinn_jax.geometry import Rectangle
from pinn_jax.problems.pdes import ComponentWeightedPDE
from pinn_jax.models import MLP

from typing import Callable, Dict


def get_poisson(u_hat: Callable) -> Callable:
    batch_hessian = get_batch_hessian(u_hat)

    def poisson(params: FrozenDict, points: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        hessians = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input

        residual = (hessians[:, 0, 0, 0] + hessians[:, 0, 1, 1] +
                    jnp.sin(jnp.pi * points[:, 0]) * jnp.sin(jnp.pi * points[:, 1]))

        return {'poisson': jnp.mean(residual ** 2)}

    return poisson


def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0) and not np.isclose(x[1], 1.0)


def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1.0)


def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1.0) and not np.isclose(x[1], 1.0)


def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)


def get_bcs(geom, bc_weights=None, default_weight=1.0):
    if bc_weights is None:
        bc_weights = {}
    bcs = {
        'u_left': ConstantDirichletBC(geom, 0.0, boundary_left, component=0),
        'u_top': ConstantDirichletBC(geom, 0.0, boundary_top, component=0),
        'u_right': ConstantDirichletBC(geom, 0.0, boundary_right, component=0),
        'u_bottom': ConstantDirichletBC(geom, 0.0, boundary_bottom, component=0),
    }
    component_weights = {}
    for bc_name in bcs:
        component_weights[bc_name] = bc_weights.get(bc_name, default_weight)
    return bcs, component_weights


def get_mesh(apply, params, num=32):
    def u_analytic(_x):
        return jnp.sin(jnp.pi * _x[:, 0]) * jnp.sin(jnp.pi * _x[:, 1]) / 2. / jnp.pi / jnp.pi
    mesh = []
    for x in np.linspace(0, 1, num=num):
        for y in np.linspace(0, 1, num=num):
            mesh.append([x, y])
    mesh = jnp.array(np.stack(mesh, axis=0))

    return mesh, apply(params, mesh), u_analytic(mesh)


def main():
    config = {'n_interior': 1_024, 'n_boundary': 1_024, 'random_state': 0, 'n_input': 2, 'n_epochs': 10_001,
              'log_every': 1_000, 'lr': 1e-3, 'save_every': 1_000, 'model_path': '', 'dtype': 'float32',
              'sampling_strategy': 'fixed', 'optimizer': 'adam'}
    n_features = (16, 16, 1)

    def model():
        return MLP(n_features)

    pde = get_poisson(model().apply)

    np.random.seed(1234)

    residual_names = ['poisson']
    geom = Rectangle((0.0, 0.0), (1.0, 1.0))

    # Assuming uniform weights
    bcs, component_weights = get_bcs(geom)
    component_weights['poisson'] = 1.0

    problem = ComponentWeightedPDE(
        pde=pde,
        residual_names=residual_names,
        geom=geom,
        bcs=bcs,
        ics={},
        ic_fds={},
        pscs={},
        anchor_points={},
        model=model,
        component_weights=component_weights,
        config=config,
        metric_functions={},
        additional_keys=[],
        additional_info=[]
    )

    problem.fit()

    fig, ax = plt.subplots(ncols=5, figsize=(25, 5))
    pd.DataFrame(problem.loss_records).set_index('epoch').plot(ax=ax[0])
    ax[0].set_yscale('log')
    pd.DataFrame(problem.error_records).set_index('epoch').plot(ax=ax[1])
    ax[1].set_yscale('log')

    num = 128
    _, u_hat, u_analytic = get_mesh(problem.state.apply_fn, problem.state.params, num)
    ctf = ax[2].contourf(np.linspace(0, 1, num=num), np.linspace(0, 1, num=num),
                         u_hat.reshape(num, num, order='F'))
    ax[2].set_title('u_hat')
    ctf = ax[3].contourf(np.linspace(0, 1, num=num), np.linspace(0, 1, num=num),
                         u_analytic.reshape(num, num, order='F'))
    ax[3].set_title('u_analytic')
    ctf = ax[4].contourf(np.linspace(0, 1, num=num), np.linspace(0, 1, num=num),
                         (u_hat - u_analytic.reshape(-1, 1)).reshape(num, num, order='F'))
    ax[4].set_title('error')
    plt.colorbar(ctf)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
