import optax

from jax import numpy as jnp, tree_map, random
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict

from pinn_jax import trees
from pinn_jax.random_trees import random_rademacher

from jaxtyping import Float, Bool, Array
from typing import Callable


# optax optimizers all use the same interface while training
OPTAX_OPTIMIZERS = ['adabelief', 'adahessian', 'adam', 'adamw', 'fromage', 'lamb', 'noisy_sgd', 'optimistic_sgd',
                    'sgd', 'yogi']

DEFAULT_OPTIMIZER_ARGS = {
    'adabelief': {'b1': 0.9, 'b2': 0.999},
    'adahessian': {},
    'adam': {'b1': 0.9, 'b2': 0.999},
    'adamw': {'b1': 0.9, 'b2': 0.999, 'weight_decay': 0.001},
    'diagsg': {'c_0': 1e-5, 'gamma': 1. / 6., 'alpha': 1e-2, 'b1': 0.9, 'b2': 0.999},
    'fromage': {'min_norm': 1e-6},
    'lamb': {'b1': 0.9, 'b2': 0.999, 'weight_decay': 0.0},
    'noisy_sgd': {'eta': 0.01, 'gamma': 0.55},
    # 'optimistic_sgd': {'alpha': 1.0, 'beta': 1.0},  # not in stable optax yet
    'sgd': {'momentum': 0.9, 'nesterov': True},
    'yogi': {'b1': 0.9, 'b2': 0.999},
}


def get_init_state(key: jnp.ndarray, key_x: jnp.ndarray, n_input: int, model: Callable, lr: float,
                   optimizer: str, dtype: str, optimizer_options: dict = None) -> TrainState:

    if optimizer_options is None:
        optimizer_options = DEFAULT_OPTIMIZER_ARGS.get(optimizer, None)

    if optimizer == 'adabelief':  # https://optax.readthedocs.io/en/latest/api.html#adabelief
        tx = optax.adabelief(lr, b1=optimizer_options['b1'], b2=optimizer_options['b2'])
    elif optimizer == 'adahessian':
        tx = adahessian(lr)
    elif optimizer == 'adam':  # https://optax.readthedocs.io/en/latest/api.html#adam
        tx = optax.adam(lr, b1=optimizer_options['b1'], b2=optimizer_options['b2'])
    elif optimizer == 'adamw':  # https://optax.readthedocs.io/en/latest/api.html#adamw
        tx = optax.adamw(lr, b1=optimizer_options['b1'], b2=optimizer_options['b2'],
                         weight_decay=optimizer_options['weight_decay'])
    elif optimizer == 'diagsg':  # hardcoded diagsg params for the moment
        tx = optax.adam(lr, b1=optimizer_options['b1'], b2=optimizer_options['b2'])
    elif optimizer == 'fromage':  # https://optax.readthedocs.io/en/latest/api.html#optax.fromage
        tx = optax.fromage(lr, min_norm=optimizer_options['min_norm'])
    elif optimizer == 'lamb':  # https://optax.readthedocs.io/en/latest/api.html#lamb
        tx = optax.lamb(lr, b1=optimizer_options['b1'], b2=optimizer_options['b2'],
                        weight_decay=optimizer_options['weight_decay'])
    elif optimizer == 'noisy_sgd':  # https://optax.readthedocs.io/en/latest/api.html#noisy-sgd
        tx = optax.noisy_sgd(lr, eta=optimizer_options['eta'], gamma=optimizer_options['gamma'])
    # elif optimizer == 'optimistic_sgd':  # https://optax.readthedocs.io/en/latest/api.html#optimistic-gd
    #     tx = None  # not in stable optax yet
    elif optimizer == 'sgd':  # https://optax.readthedocs.io/en/latest/api.html#sgd
        tx = optax.sgd(lr, momentum=optimizer_options['momentum'], nesterov=optimizer_options['nesterov'])
    elif optimizer == 'yogi':  # https://optax.readthedocs.io/en/latest/api.html#yogi
        tx = optax.yogi(lr, b1=optimizer_options['b1'], b2=optimizer_options['b2'])
    else:
        raise ValueError('unsupported optimizer specified!!!')

    x_dummy = random.normal(key_x, (2, n_input))
    assert x_dummy.dtype == dtype, f"dtype is {dtype} and desired precision not being used!!! " \
                                   f"set jax.config.config.update('jax_enable_x64', True) to use 64-bit precision"
    return TrainState.create(
        apply_fn=model().apply,
        params=model().init(key, x_dummy),
        tx=tx,
    )


def get_estimate_diag(grad_fn: Callable, c_0: float, gamma: float, alpha_0: float) -> Callable:
    def estimate_diag(state: TrainState, params: FrozenDict, _diag_old: FrozenDict,
                      interior_points: Float[Array, "n_interior n_inputs"],
                      boundary_points: Float[Array, "n_boundary n_inputs"],
                      boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                      initial_points: Float[Array, "n_initial n_inputs"], _key_r
                      ) -> FrozenDict:
        """hessian diagonal estimation for diagonal hessian stochastic gradient (diagSG) method based on simultaneous
        perturbation.
        for details see:
        - https://ieeexplore.ieee.org/document/9400243
        - https://jhuapl.box.com/s/43bpk1914hdyd7npv2b8ld9siznze52f
        """
        # calculate perturbed gradients
        step_scale = jnp.array(c_0 / (1. + state.step) ** gamma)  # perturbation step size `c_t`
        _key_r, subkey = random.split(_key_r, num=2)
        rademacher = trees.scale(random_rademacher(subkey, params), step_scale)
        _, grads_m = grad_fn(trees.add(params, rademacher, beta=-1.),  # g(w_t - c_t Delta_t)
                             interior_points, boundary_points, boundary_idxs, initial_points)
        _, grads_p = grad_fn(trees.add(params, rademacher),  # g(w_t + c_t Delta_t)
                             interior_points, boundary_points, boundary_idxs, initial_points)
        # construct estimated hessian diagonal
        diag_raw = trees.add(grads_p, grads_m, beta=-1.)
        diag_raw = trees.divide(diag_raw, rademacher, beta=2. * step_scale)
        _diag_old = trees.add(_diag_old, diag_raw,
                              alpha=(state.step + 1.) / (state.step + 2.),
                              beta=1. / (state.step + 2.))
        diag = tree_map(lambda _p: jnp.abs(_p) + alpha_0, _diag_old)

        return diag

    return estimate_diag
