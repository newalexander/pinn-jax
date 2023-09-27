# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import numpy as np
import logging

from jax import numpy as jnp, random, jit, value_and_grad
from jax.tree_util import tree_map
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from optax import adam, apply_updates

from pinn_jax.problems.abstract_problem import AbstractProblem
from pinn_jax.derivatives import get_batch_jacobian

from tqdm import tqdm
from abc import ABC
from typing import Callable, Dict, Union, List, Tuple


class OrdinaryDiffEq(AbstractProblem, ABC):
    """abstract class meant to represent any arbitrary-dimensional ODE"""
    def __init__(self,
                 ode: Callable,
                 time_min: float,
                 time_max: float,
                 ic_vals: jnp.ndarray,
                 ic_fd_vals: Union[jnp.ndarray, None],
                 eval_times: Union[jnp.ndarray, None],
                 model: Callable,
                 config: Dict,
                 metric_functions: Dict[str, Callable],
                 additional_keys: List[str],
                 additional_info: List[str]):

        super().__init__(model, config, metric_functions, additional_keys, additional_info)
        # update shared records
        self.error_records['residual'] = []
        self.error_records['ic_error'] = []
        self.error_records['ic_fd_error'] = []

        self.interior_points = jnp.array(np.linspace(time_min, time_max, num=self.config['n_interior'])[:, None])
        self.initial_points = time_min * jnp.ones((1, 1))

        _u_hat = model().apply(self.state.params, self.initial_points)  # used to validate sizes of things
        self.ode = ode
        self.ic_vals = ic_vals
        self.ic_fd_vals = ic_fd_vals
        assert ic_vals.shape == _u_hat.shape, "invalid shape of `ic_vals` specified!!!"

        if eval_times is not None:
            self.eval_times = eval_times
        else:
            self.eval_times = self.initial_points
        assert len(eval_times.shape) == 2 and eval_times.shape[1] == 1, "`eval_times` should a n x 1 array!!!"

        # make metric functions have interface expected by the abstract problem
        # TODO: this doesn't work if `self.metric_functions` has more than one key because of how dict comprehensions
        #  handle variable binding
        #  https://stackoverflow.com/questions/36805071/dictionary-comprehension-with-lambda-functions-gives-wrong-results
        #  we can also just use for loops
        self.metric_functions = {key: lambda _params: value(_params, self.eval_times)
                                 for key, value in self.metric_functions.items()}

    def get_residual_loss_func(self):
        """
        :return: (n_interior, ) vector of residual errors summed over all state variables
        """
        def get_residual_loss(params: FrozenDict) -> jnp.ndarray:
            residuals = (self.ode(params, self.interior_points) ** 2).sum(axis=1)
            return residuals

        return get_residual_loss

    def get_ic_loss_func(self):
        """
        :return: scalar of initial condition errors, summed over all state variables
        """
        def get_ic_loss(params: FrozenDict) -> jnp.ndarray:
            u = self.state.apply_fn(params, self.initial_points)
            ic_errors = ((u - self.ic_vals) ** 2).sum()
            return ic_errors

        return get_ic_loss

    def get_ic_fd_loss_func(self):
        """
        :return: scalar of initial condition errors for first derivative of solution, summed over all state variables
        """
        batch_jacobian = get_batch_jacobian(self.state.apply_fn)

        def get_ic_fd_loss(params: FrozenDict) -> jnp.ndarray:
            u_t = batch_jacobian(params, self.initial_points).squeeze(-1)
            ic_fd_errors = ((u_t - self.ic_fd_vals) ** 2).sum()
            return ic_fd_errors

        def get_zero(_):
            return np.array([0.0])

        if self.ic_fd_vals is not None:
            return get_ic_fd_loss
        else:
            return get_zero

    def get_loss_components(self):
        residual_loss_func = self.get_residual_loss_func()
        ic_loss_func = self.get_ic_loss_func()
        ic_fd_loss_func = self.get_ic_fd_loss_func()

        @jit
        def predict():
            return residual_loss_func(self.state.params).mean(), ic_loss_func(self.state.params), ic_fd_loss_func(self.state.params)

        residual_loss, ic_loss, ic_fd_loss = predict()
        return {'residual': float(residual_loss), 'ic_error': float(ic_loss), 'ic_fd_error': float(ic_fd_loss)}


class ComponentWeightedODE(OrdinaryDiffEq):
    """
    solve the ODE
        ``N(u)(t) = 0,   t in [0, T]``

        ``I(u)(t) = 0,   t = 0``

    by minimizing the loss
        ``L(w) = lambda_r L_r(w) + lambda_i L_i(w)``

    where
        ``L_r(w) = sum_n ||N(w)(t_n)||^2``

        ``L_i(w) = ||I(w)(0)||^2``

    for pre-specified loss component weights ``lambda_r`` and ``lambda_i``
    """
    def __init__(self,
                 ode: Callable,
                 time_min: float,
                 time_max: float,
                 ic_vals: jnp.ndarray,
                 ic_fd_vals: Union[jnp.ndarray, None],
                 eval_times: Union[jnp.ndarray, None],
                 component_weights: Dict[str, float],
                 model: Callable,
                 config: Dict,
                 metric_functions: Dict[str, Callable]):
        super().__init__(ode, time_min, time_max, ic_vals, ic_fd_vals, eval_times, model, config, metric_functions, [], [])
        assert 'residual' in component_weights.keys(), 'residual component weight should be specified!!!'
        assert 'ic' in component_weights.keys(), 'ic component weight should be specified!!!'
        assert 'ic_fd' in component_weights.keys(), 'ic_fd component weight should be specified!!!'
        self.component_weights = component_weights

    def get_train_step(self):
        residual_loss_func = self.get_residual_loss_func()
        ic_loss_func = self.get_ic_loss_func()
        ic_fd_loss_func = self.get_ic_fd_loss_func()

        def loss_fn(params: FrozenDict) -> jnp.ndarray:
            residual = self.component_weights['residual'] * residual_loss_func(params).mean()
            ic = self.component_weights['ic'] * ic_loss_func(params).mean()
            ic_fd = self.component_weights['ic_fd'] * ic_fd_loss_func(params).mean()
            return residual + ic + ic_fd

        grad_fn = value_and_grad(loss_fn)

        @jit
        def train_step_adam(state: TrainState, _) -> Tuple[TrainState, jnp.ndarray, FrozenDict, None]:
            # calculate gradient updates
            loss, grads = grad_fn(state.params)
            # update network state
            state = state.apply_gradients(grads=grads)
            return state, loss, grads, None

        if self.config['optimizer'] == 'adam':
            return train_step_adam

    def _train(self):
        state, train_step = self.state, self.get_train_step()
        if self.config['optimizer'] == 'adam':
            key_r = None
        else:
            raise ValueError('unsupported optimizer specified!!!')

        for epoch in tqdm(range(self.config['n_epochs'])):
            state, loss, _, key_r = train_step(state, key_r)

            self.state = state
            self.log_loss_records(epoch, loss)

            if epoch % self.config['log_every'] == 0:
                self.log_error_records(epoch)
                self.log_metric_records(epoch)
            if epoch % self.config['save_every'] == 0:
                self.save_params(epoch)


class AdaptiveWeightedODE(OrdinaryDiffEq):
    """
    solve the ODE

        ``N(u)(t) = 0,   t  [0, T]``

        ``I(u)(t) = 0,   t = 0``


    by minimizing the loss

        ``L(w, lambda_r, lambda_i) = lambda_r L_r(w) + lambda_i L_i(w)``

        ``s.t. lambda_r >= 0, lambda_i >= 0``

    where

        ``L_r(w) = sum_n m(lambda_{r,n}) ||N(w)(t_n)||^2``

        ``L_i(w) = lambda_i m||I(w)(0)||^2``

    for some non-negative, differentiable, strictly increasing function ``m``


    optimization uses a gradient descent/ascent procedure

        ``w <- w - eta_w partial_w L(w, lambda_r, lambda_b)``

        ``lambda_r <- lambda_r + eta_r partial_{lambda_r} L(w, lambda_r, lambda_b)``

        ``lambda_i <- lambda_i + eta_i partial_{lambda_i} L(w, lambda_r, lambda_b)``

    implementation of the "self-adaptive collocation weights via soft attention" mechanism proposed in
    in mcclenny and braga-neto (2020) [1].

    [1] https://arxiv.org/abs/2009.04544).
    """
    def __init__(self,
                 ode: Callable,
                 time_min: float,
                 time_max: float,
                 ic_vals: jnp.ndarray,
                 ic_fd_vals: Union[jnp.ndarray, None],
                 eval_times: Union[jnp.ndarray, None],
                 model: Callable,
                 config: Dict,
                 metric_functions: Dict[str, Callable],
                 masking_func: Union[None, Callable]):
        super().__init__(ode, time_min, time_max, ic_vals, ic_fd_vals, eval_times, model, config, metric_functions,
                         ['lr_weights', 'random_state_weights'], [])
        if masking_func is None:
            def identity(x): return x

            self.masking_func = identity
        else:
            self.masking_func = masking_func

        key_r, key_i = random.split(random.PRNGKey(config['random_state_weights']))
        self.weights = {
            'residual': random.uniform(key_r, (self.config['n_interior'],)),
            'ic': random.uniform(key_i, ()),
            'ic_fd': random.uniform(key_i, ())
        }
        self.weights_optimizer = adam(self.config['lr_weights'])

    def get_train_step(self):
        ic_loss_func = self.get_ic_loss_func()
        ic_fd_loss_func = self.get_ic_fd_loss_func()
        residual_loss_func = self.get_residual_loss_func()

        def loss_fn(params: FrozenDict, weights: dict) -> jnp.ndarray:
            residual = (weights['residual'] * residual_loss_func(params)).mean()
            ic = (weights['ic'] * ic_loss_func(params)).mean()
            ic_fd = (weights['ic_fd']*ic_fd_loss_func(params)).mean()

            return residual + ic + ic_fd

        grad_fn = value_and_grad(loss_fn, argnums=(0, 1))

        @jit
        def train_step(state: TrainState, weights: dict, weights_state):
            # calculate gradient updates
            loss, (grads_w, grads_lambda) = grad_fn(state.params, weights)
            # negate gradients of adaptive weights since performing gradient ascent
            grads_lambda = tree_map(lambda x: -x, grads_lambda)
            # update network state
            state = state.apply_gradients(grads=grads_w)
            # update adaptive weights
            weights_updates, weights_state = self.weights_optimizer.update(grads_lambda, weights_state, weights)
            weights = apply_updates(weights, weights_updates)
            return state, loss, weights, weights_state, grads_w, grads_lambda

        return train_step

    def _train(self):
        # unpack things for use in training loop
        state, weights, train_step = self.state, self.weights, self.get_train_step()
        weights_state = self.weights_optimizer.init(weights)

        for epoch in tqdm(range(self.config['n_epochs'])):
            state, loss, weights, weights_state, _, _ = train_step(state, weights, weights_state)

            self.state = state
            self.weights = weights
            self.log_loss_records(epoch, loss)

            if epoch % self.config['log_every'] == 0:
                self.log_error_records(epoch)
                self.log_metric_records(epoch)
            if epoch % self.config['save_every'] == 0:
                self.save_params(epoch)


class CausalWeightedODE(OrdinaryDiffEq):
    """
    adapted from: https://arxiv.org/abs/2203.07404
    compared to this, we:
    - focus on the case of ODEs rather than PDEs
    - move all derivatives into the operator N in eq. 13, rather than having a separate first-order time derivative
    we might also specify some ambiguities wrt the handling of the initial condition

    we have the full loss function

        ``L(w, z) = 1 / (N+1) sum_{n=0}^N z_i L(t_n, w)``

    where the individual terms are given by

        ``L(t_0, w) = mu_ic ||I(w)||^2``

        ``L(t_n, w) = ||N(w)(t_n)||^2, n = 1, ..., N``

    then training looks like this:

        for ``epsilon`` in ``steepness_weights``
            for ``step`` in ``1``, ..., ``n_steps``
                ``z = zeros(N+1), z_0 = 1.0``  # temporal weights

                for ``n`` in ``1``, ..., ``N``
                    ``z_n = exp(-epsilon sum_{n'=0}^{n-1} L(t_n', w)``
                # treat z as constant wrt theta and do a gradient step

                ``w <- w - eta grad_w L(w, z)``

                if ``min z > delta``
                    break
    """
    def __init__(self,
                 ode: Callable,
                 time_min: float,
                 time_max: float,
                 ic_vals: jnp.ndarray,
                 ic_fd_vals: Union[jnp.ndarray, None],
                 eval_times: Union[jnp.ndarray, None],
                 component_weights: Dict[str, float],
                 model: Callable,
                 config: Dict,
                 metric_functions: Dict[str, Callable]):
        super().__init__(ode, time_min, time_max, ic_vals, ic_fd_vals, eval_times, model, config, metric_functions,
                         ['steepness_weights', 'convergence_criterion'], ['steepness_weight'])
        assert 'residual' in component_weights.keys(), 'residual component weight should be specified!!!'
        assert 'ic' in component_weights.keys(), 'ic component weight should be specified!!!'
        assert 'ic_fd' in component_weights.keys(), 'ic_fd component weight should be specified!!!'
        self.component_weights = component_weights

    def get_train_step(self):
        ic_loss_func = self.get_ic_loss_func()  # initial condition error
        ic_fd_loss_func = self.get_ic_fd_loss_func()  # initial condition first derivative error
        residual_error_func = self.get_residual_loss_func()  # timepoint-specific residual errors

        # use matrix multiplication with `mat` to implement cumulative sums without a for-loop
        mat = jnp.tril(jnp.ones((self.config['n_interior'] + 1, self.config['n_interior'] + 1)), k=-1)

        def loss_vec_fn(params: FrozenDict) -> jnp.ndarray:
            """calculate time-specific errors `R_n` (including the initial condition error `R_0 = I`"""
            ic_error = jnp.array([self.component_weights['ic'] * ic_loss_func(params).sum() +
                                  self.component_weights['ic_fd'] * ic_fd_loss_func(params).sum()])
            residual_error = residual_error_func(params)
            return jnp.concatenate([ic_error, residual_error], axis=0)

        def loss_fn(params: FrozenDict, steepness_weight: float) -> jnp.ndarray:
            """the full weighted loss that takes the average over all the time points and returns a scalar"""
            loss_vec = loss_vec_fn(params)
            temporal_weights = jnp.exp(-steepness_weight * (mat @ loss_vec))
            return (loss_vec * temporal_weights).mean()

        grad_fn = value_and_grad(loss_fn, argnums=0)

        @jit
        def train_step(state: TrainState, steepness_weight: float) -> Tuple[TrainState, float, FrozenDict]:
            # calculate gradient updates
            loss, grads = grad_fn(state.params, steepness_weight)
            # update network state
            state = state.apply_gradients(grads=grads)
            return state, loss, grads

        return train_step

    def _train(self):
        # unpack things for use in training loop
        state, train_step = self.state, self.get_train_step()

        for steepness_idx, steepness_weight in enumerate(self.config['steepness_weights']):
            logging.info(f'steepness_weight: {steepness_weight}\t{steepness_idx+1} / {len(self.config["steepness_weights"])}')
            for epoch in tqdm(range(self.config['n_epochs'])):
                state, loss, _ = train_step(state, steepness_weight)

                self.state = state
                kwargs = {'steepness_weight': steepness_weight}
                self.log_loss_records(epoch, loss, **kwargs)

                if epoch % self.config['log_every'] == 0:
                    self.log_error_records(epoch, **kwargs)
                    self.log_metric_records(epoch, **kwargs)
                if epoch % self.config['save_every'] == 0:
                    self.save_params(epoch)
