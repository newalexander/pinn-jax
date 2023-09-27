"""
we're solving the 1d heat equation:

w u_t = u_xx,  (x, t) in (0, L) x (0, T)
u(0, t) = l
u(L, t) = r
u(x, 0) = f(x)

specific example here: https://personal.math.ubc.ca/~peirce/HeatProblems.pdf

4 u_t = u_xx,  (x, t) in (0, 2) x (0, T)
u(0, t) = 0
u(2, t) = 0
u(x, 0) = 2 sin (pi/2 x) - sin (pi x) + 4 sin (2 pi x)

this admits the solution:

u(x, t) = 2 sin (pi/2 x) exp (-pi^2/16 t) - sin (pi x) exp (-pi^2/4 t) + 4 sin (2 pi x) exp (-pi^2 t)
"""


import numpy as np

from jax import numpy as jnp, jit, random
from flax.core.frozen_dict import FrozenDict

from pinn_jax.derivatives import get_batch_jacobian, get_batch_hessian
from pinn_jax.bcs import ConstantDirichletBC, FunctionIC
from pinn_jax.geometry import GeometryXTime, TimeDomain, Interval
from pinn_jax.problems.pdes import ComponentWeightedPDE
from pinn_jax.models import MLP, get_activation
from pinn_jax.utils import get_logs_id

from typing import Callable, Dict, Tuple
from argparse import ArgumentParser

X_MIN, X_MAX = 0., 2.
T_MIN, T_MAX = 0., 1.


def get_heat_eqn(u_hat: Callable, w: float) -> Tuple[Callable, Callable]:
    batch_jacobian = get_batch_jacobian(u_hat)
    batch_hessian = get_batch_hessian(u_hat)

    def residuals(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        hessian = batch_hessian(params, points)  # n_batch x n_output x n_input x n_input
        jacobian = batch_jacobian(params, points)  # n_batch x n_output x n_input

        du_xx = hessian[:, 0, 0, 0]
        du_t = jacobian[:, 0, 1]

        residual = du_xx - w * du_t
        return residual

    def heat_eqn(params: FrozenDict, points: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return {'heat_eqn': residuals(params, points)}

    return heat_eqn, residuals


def solution(xt: jnp.ndarray):
    return (2. * jnp.sin(jnp.pi / 2. * xt[:, 0]) * jnp.exp(-jnp.pi**2 / 16. * xt[:, 1])
            - jnp.sin(jnp.pi * xt[:, 0]) * jnp.exp(-jnp.pi**2 / 4. * xt[:, 1])
            + 4. * jnp.sin(2. * np.pi * xt[:, 0]) * jnp.exp(-jnp.pi**2 * xt[:, 1]))


def u_0(xt: jnp.ndarray):
    return 2. * jnp.sin(jnp.pi / 2. * xt[:, [0]]) - jnp.sin(jnp.pi * xt[:, [0]]) + 4. * jnp.sin(2. * jnp.pi * xt[:, [0]])


def get_geom(x_min=X_MIN, x_max=X_MAX, t_min=T_MIN, t_max=T_MAX) -> GeometryXTime:
    geom_x = Interval(x_min, x_max)
    geom_t = TimeDomain(t_min, t_max)
    return GeometryXTime(geometry=geom_x, timedomain=geom_t)


def boundary_left(xt, on_boundary):
    return on_boundary and np.isclose(xt[0], X_MIN)


def boundary_right(xt, on_boundary):
    return on_boundary and np.isclose(xt[0], X_MAX)


def get_bcs(geom, bc_weights=None, default_weight=1.0):
    if bc_weights is None:
        bc_weights = {}
    bcs = {
        'u_left': ConstantDirichletBC(geom, 0.0, boundary_left, component=0),
        'u_right': ConstantDirichletBC(geom, 0.0, boundary_right, component=0),
    }
    component_weights = {}
    for bc_name in bcs.keys():
        component_weights[bc_name] = bc_weights.get(bc_name, default_weight)
    return bcs, component_weights


def get_ics(geom, ic_weights=None, default_weight=1.0):
    if ic_weights is None:
        ic_weights = {}
    ics = {'u_0': FunctionIC(geom, u_0)}
    component_weights = {'u_0': ic_weights.get('u_0', default_weight)}
    return ics, component_weights


def main():
    n_eval = 1_024
    key_x, key_t = random.split(random.PRNGKey(0))
    xt = jnp.vstack([
        random.uniform(key_x, shape=(n_eval, ), minval=X_MIN, maxval=X_MAX),
        random.uniform(key_t, shape=(n_eval, ), minval=T_MIN, maxval=T_MAX)
    ]).T
    u_soln = u_0(xt)
    u_soln_norm = jnp.linalg.norm(u_soln)

    print(xt.shape, u_soln.shape)

    # construct neural network
    n_features = (config['n_hidden'], config['n_hidden'], config['n_hidden'], 1)
    activation = get_activation(config['activation'], config['activation_scaling'])

    def model():
        return MLP(n_features, activation=activation)

    w = 4.
    pde, _ = get_heat_eqn(model().apply, w)

    residual_names = ['heat_eqn']
    geom = get_geom()

    # assuming uniform weights
    bcs, bc_component_weights = get_bcs(geom, default_weight=1.)
    ics, ic_component_weights = get_ics(geom, default_weight=1.)

    component_weights = {'heat_eqn': 1.0}
    for c_w in [bc_component_weights, ic_component_weights]:
        component_weights.update(c_w)

    @jit
    def rel_error(params):
        u_hat = model().apply(params, xt)
        return jnp.linalg.norm(u_hat - u_soln) / u_soln_norm

    metric_functions = {'rel_error': rel_error}

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
    else:
        raise ValueError('invalid problem type specified!!!')
    problem.fit()


if __name__ == '__main__':
    np.random.seed(1234)

    parser = ArgumentParser()
    parser.add_argument('--n_interior', type=int, default=1_024)
    parser.add_argument('--n_boundary', type=int, default=64)
    parser.add_argument('--n_initial', type=int, default=256)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--key_r', type=int, default=0)
    parser.add_argument('--n_input', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=120_001)
    parser.add_argument('--log_every', type=int, default=5)#10_000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=5)#40_000)
    parser.add_argument('--model_path', type=str, default='/Users/newas1/data/firstline/heat_equation/')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--sampling_strategy', type=str, default='stochastic')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--activation_scaling', type=float, default=1.0)
    parser.add_argument('--problem_type', type=str, choices=['component_weighted_pde'])

    config = vars(parser.parse_args())
    config['model_path'] += get_logs_id()

    if config['dist_fn'] == 'abs':
        dist_fn = jnp.abs

    main()
