"""
we're solving the 1d burgers equation

    D(x, t; u) = partial_t u + u partial_x u - nu partial_xx u = 0,   (x, t) in (-1, 1) x (0, 1)
    u(t, -1) = u(t, 1) = 0
    u(0, x) = -sin pi x

with nu = 0.01 / pi
"""

import numpy as np

from jax import numpy as jnp, jit
from flax.core.frozen_dict import FrozenDict
from scipy import stats

from pinn_jax.derivatives import get_batch_jacobian, get_batch_hessian
from pinn_jax.bcs import ConstantDirichletBC, FunctionIC
from pinn_jax.geometry import GeometryXTime, TimeDomain, Interval
from pinn_jax.problems.pdes import ComponentWeightedPDE, ConstraintSatisfactionPDE
from pinn_jax.models import MLP, get_activation
from pinn_jax.utils import get_logs_id

from typing import Callable, Dict, Tuple
from argparse import ArgumentParser


def get_burgers(u_hat: Callable, nu: float) -> Tuple[Callable, Callable]:
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    def residuals(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        u = u_hat(params, points).squeeze()
        hessian = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input
        jacobian = batch_jacobian(params, points)  # n_batch x n_output x n_input

        du_xx = hessian[:, 0, 0, 0]
        du_t = jacobian[:, 0, 1]
        du_x = jacobian[:, 0, 0]

        residual = du_t + u * du_x - nu * du_xx
        return residual

    def burgers_eqn(params: FrozenDict, points: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return {'burgers': residuals(params, points)}

    return burgers_eqn, residuals


# source: https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Burgers.py
def gen_testdata() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """returns a ground-truth solution to burger's equation evaluated on a uniform discretization of [-1, 1] x [0, 1]"""
    data = np.load("burgers_solution.npz")
    t, x, u = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T  # (25,600 x 2)
    u = u.flatten()[:, None]  # (25,600 x 1)
    return jnp.array(X), jnp.array(u)


def get_geom(x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0) -> GeometryXTime:
    geom_x = Interval(x_min, x_max)
    geom_t = TimeDomain(t_min, t_max)
    return GeometryXTime(geometry=geom_x, timedomain=geom_t)


def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[0], -1.0)


def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1.0)


def get_bcs(geom, bc_weights=None, default_weight=1.0):
    if bc_weights is None:
        bc_weights = {}
    bcs = {
        'u_bottom': ConstantDirichletBC(geom, 0.0, boundary_bottom, component=0),
        'u_top': ConstantDirichletBC(geom, 0.0, boundary_top, component=0),
    }
    component_weights = {}
    for bc_name in bcs.keys():
        component_weights[bc_name] = bc_weights.get(bc_name, default_weight)
    return bcs, component_weights


def get_ics(geom, ic_weights=None, default_weight=1.0):
    if ic_weights is None:
        ic_weights = {}
    ics = {'u_0': FunctionIC(geom, lambda x: -jnp.sin(np.pi * x[:, [0]]))}
    component_weights = {'u_0': ic_weights.get('u_0', default_weight)}
    return ics, component_weights


def main():
    nu = 0.01 / np.pi

    # ground-truth data to evaluate against
    xt_grid, u_soln = gen_testdata()
    u_soln_norm = jnp.linalg.norm(u_soln)

    # construct neural network
    n_features = (config['n_hidden'], config['n_hidden'], config['n_hidden'], 1)
    activation = get_activation(config['activation'], config['activation_scaling'])

    def model():
        return MLP(n_features, activation=activation)

    pde, residual_fn = get_burgers(model().apply, nu)

    np.random.seed(1234)

    residual_names = ['burgers']
    geom = get_geom()

    # assuming uniform weights
    bcs, bc_component_weights = get_bcs(geom, default_weight=1.)
    ics, ic_component_weights = get_ics(geom, default_weight=1.)

    component_weights = {'burgers': 1.0}
    for c_w in [bc_component_weights, ic_component_weights]:
        component_weights.update(c_w)

    @jit
    def rel_error(params):
        _u_hat = model().apply(params, xt_grid)
        return jnp.linalg.norm(_u_hat - u_soln) / u_soln_norm

    @jit
    def jit_residual(params):
        return residual_fn(params, xt_grid)

    def kurtosis(params):
        residuals = residual_fn(params, xt_grid)
        return stats.kurtosis(residuals)

    metric_functions = {'rel_error': rel_error, 'kurtosis': kurtosis}
    if config['dist_fn'] == 'abs':
        dist_fn = jnp.abs
    elif config['dist_fn'] is None:
        dist_fn = None
    else:
        raise ValueError('invalid distance function specified!!!')

    if config['problem_type'] == 'component_weighted_pde':
        problem = ComponentWeightedPDE(
            pde=pde,
            residual_names=residual_names,
            geom=geom,
            bcs=bcs,
            ics=ics,
            pscs={},
            anchor_points={},
            model=model,
            component_weights=component_weights,
            config=config,
            metric_functions=metric_functions,
            additional_keys=[],
            additional_info=[]
        )
    elif config['problem_type'] == 'constraint_satisfaction':
        problem = ConstraintSatisfactionPDE(
            pde=pde,
            residual_names=residual_names,
            geom=geom,
            bcs=bcs,
            ics=ics,
            pscs={},
            anchor_points={},
            model=model,
            component_weights=component_weights,
            config=config,
            metric_functions=metric_functions,
            additional_keys=[],
            additional_info=[],
            dist_fn=dist_fn
        )
    else:
        raise ValueError('invalid problem type specified!!!')
    problem.fit()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_interior', type=int, default=2_540)
    parser.add_argument('--n_boundary', type=int, default=80)
    parser.add_argument('--n_initial', type=int, default=160)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--key_r', type=int, default=0)
    parser.add_argument('--n_input', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=120_001)
    parser.add_argument('--log_every', type=int, default=100)#10_000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=100)#40_000)
    parser.add_argument('--model_path', type=str, default='/fast/newas1/data/firstline/burgers/2022_10_03/')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--sampling_strategy', type=str, default='fixed')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--activation_scaling', type=float, default=1.0)
    parser.add_argument('--problem_type', type=str,
                        choices=['component_weighted_pde', 'augmented_lagrangian', 'augmented_total_lagrangian'],
                        default='augmented_total_lagrangian')

    config = vars(parser.parse_args())

    config['constraints_tolerance'] = 1e-6
    config['max_penalty'] = 1e2
    config['n_inner_steps'] = 10
    config['dist_fn'] = 'abs'

    config['model_path'] += get_logs_id()

    main()
