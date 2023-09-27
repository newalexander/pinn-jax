# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import json
import os

import numpy as np
import pandas as pd
import logging

from scipy.spatial import distance_matrix

from jax import numpy as jnp, jit, value_and_grad, random, tree_map
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from pinn_jax.geometry import Geometry, GeometryXTime
from pinn_jax.bcs import BC, IC, PointSetBC
from pinn_jax.optimization import get_estimate_diag
from pinn_jax import trees

from pinn_jax.problems.abstract_problem import AbstractProblem

from tqdm import tqdm
from abc import ABC
from typing import Callable, Dict, Union, List, Tuple
from jaxtyping import Float, Bool, Array  # `Array` is an alias for `jax.numpy.ndarray`

TRAINING_STRATEGIES = ['stochastic',  # sample a new batch of points every steps (default every step)
                       'fixed',  # sample a single set of points and never change it
                       'evo',  # "evolutionarily" build up a set of collocation points
                               # for details see (https://openreview.net/forum?id=Jzliv-bxZla)
                       # 'prespecified'  # TODO: add support here
                       ]
C_0, GAMMA, ALPHA_0 = 1e-5, 1. / 6., 1e-2  # hardcoded diagsg parameters (see `train_step_diagsg` methods)


class PartialDiffEq(AbstractProblem, ABC):
    """abstract class meant to represent a general PDE for a state variable u(x) in R^n:

        N_i(u)(x) = 0,      x in Omega subseteq R^m
        u_i(x) = b_ij(x),   x in Gamma_j

    where `N_i` is the `i`th residual and `b_ij` is imposed on `u_i` for `j`th boundary `Gamma_j`
    """
    def __init__(self,
                 pde: Callable,
                 residual_names: list,
                 geom: Union[Geometry, GeometryXTime],
                 bcs: Dict[str, BC],
                 ics: Dict[str, IC],
                 pscs: Dict[str, PointSetBC],
                 anchor_points: Dict[str, Float[Array, "n_points n_inputs"]],
                 model: Callable,
                 component_weights: Dict[str, float],
                 config: Dict,
                 metric_functions: Dict[str, Callable],
                 additional_keys: List[str],
                 additional_info: List[str],
                 bc_fn: Callable,
                 ic_fn: Callable,
                 psc_fn: Callable
                 ):
        additional_keys = ['sampling_strategy', 'n_boundary'] + additional_keys
        if isinstance(geom, GeometryXTime):
            additional_keys += ['n_initial']
        super().__init__(model, config, metric_functions, additional_keys, additional_info)

        # update shared records
        for name in set().union(residual_names, bcs.keys(), ics.keys(), pscs.keys()):
            self.error_records[name] = []

        self.pde = pde
        self.residual_names = residual_names
        self.geom = geom
        self.bcs = bcs
        self.ics = ics

        # "condition function" used to evaluate BCs, ICs, PSCs
        self.bc_fn = bc_fn
        self.ic_fn = ic_fn
        self.psc_fn = psc_fn

        self.pscs = pscs
        self.anchor_points = anchor_points
        assert pscs.keys() == anchor_points.keys(), "point set conditions must be aligned with anchor points!!!"

        self.component_weights = component_weights

        self.boundary_points, self.boundary_idxs, self.interior_points, self.initial_points = None, None, None, None
        self.prepare_sampling_strategy()
        self.prepare_lbfgs()

        self._validate_keys()

    """methods for sampling points from the domain"""

    def sample_interior(self, n=None) -> Float[Array, "n_points n_inputs"]:
        # default value based on `config`
        if n is None:
            n = self.config['n_interior']
        return jnp.array(self.geom.random_points(n))

    def sample_boundary(self) -> Tuple[Union[Float[Array, "n_points n_inputs"], None],
                                       Union[Bool[Array, "n_points n_bcs"], None]]:
        if self.config['n_boundary'] > 0:
            boundary_points = self.geom.random_boundary_points(self.config['n_boundary'])

            # find out which boundaries each boundary point is on
            boundary_idxs = np.zeros((len(boundary_points), len(self.bcs)), dtype=bool)
            for (bc_idx, bc) in enumerate(self.bcs.values()):
                boundary_idxs[:, bc_idx] = bc.filter_idx(boundary_points)

            # everything still needs to be a jax array
            boundary_points = jnp.array(boundary_points)
            boundary_idxs = jnp.array(boundary_idxs)
            return boundary_points, boundary_idxs
        else:
            return None, None

    def sample_initial(self) -> Float[Array, "n_points n_inputs"]:
        if isinstance(self.geom, GeometryXTime) and (self.config['n_initial'] > 0):
            return jnp.array(self.geom.random_initial_points(self.config['n_initial']))
        else:
            return jnp.array([])

    def get_points(self, epoch, interval, residual_norm_fn=None, lambda_=None
                   ) -> Tuple[Float[Array, "n_interior n_inputs"], Float[Array, "n_boundary n_inputs"],
                              Bool[Array, "n_boundary n_bcs"], Float[Array, "n_initial n_inputs"]]:

        if self.config['sampling_strategy'] == 'stochastic':
            # are we sampling a new mesh?
            if (epoch % interval) == 0:
                interior_points = self.sample_interior()
                boundary_points, boundary_idxs = self.sample_boundary()
                initial_points = self.sample_initial()
            # otherwise use current mesh
            else:
                interior_points = self.interior_points
                boundary_points, boundary_idxs = self.boundary_points, self.boundary_idxs
                initial_points = self.initial_points

        elif self.config['sampling_strategy'] == 'fixed':
            interior_points = self.interior_points
            boundary_points, boundary_idxs = self.boundary_points, self.boundary_idxs
            initial_points = self.initial_points

        elif self.config['sampling_strategy'] == 'evo':
            # we hold boundary and initial points constant. TODO: is this the right thing to do?
            boundary_points, boundary_idxs = self.boundary_points, self.boundary_idxs
            initial_points = self.initial_points

            # are we adding things to our mesh?
            if (epoch % interval) == 0 and epoch > 0:
                if lambda_ is None:
                    interior_residuals = residual_norm_fn(self.state.params, self.interior_points)  # summed over residuals
                else:
                    interior_residuals = residual_norm_fn(self.state.params, self.interior_points, lambda_)
                residual_thresh = interior_residuals.mean()
                n_kept = jnp.sum(interior_residuals > residual_thresh)
                n_new = self.interior_points.shape[0] - n_kept
                # keep points with bad residual
                interior_points = jnp.vstack([
                    self.interior_points[interior_residuals > residual_thresh, :],
                    jnp.array(self.sample_interior(n=n_new))])  # new points

            # otherwise use current mesh
            else:
                interior_points = self.interior_points

        else:  # default option that will throw a failure
            interior_points = None
            boundary_points, boundary_idxs = None, None
            initial_points = None

        return interior_points, boundary_points, boundary_idxs, initial_points

    # TODO: support for other sampling strategies (see
    #  https://neuralpde.sciml.ai/stable/manual/training_strategies/#Training-Strategies)
    #  broad challenge is that things like latin hypercubes don't easily generalize to arbitrary irregular domains
    # TODO: support for ingesting a prespecified mesh
    def prepare_sampling_strategy(self):
        assert self.config['sampling_strategy'] in TRAINING_STRATEGIES, "invalid training strategies specified!!"
        if self.config['sampling_strategy'] == 'fixed':
            self.boundary_points, self.boundary_idxs = self.sample_boundary()
            self.interior_points = self.sample_interior()
            self.initial_points = self.sample_initial()
            self.config['update_mesh_every'] = None  # not used for this strategy

        if self.config['sampling_strategy'] == 'stochastic':
            if 'update_mesh_every' not in self.config.keys():
                logging.info('warning: `stochastic` sampling strategy chosen without specifying `update_mesh_every`, '
                             'setting to 1')
                self.config['update_mesh_every'] = 1
            # generate initial sample
            self.boundary_points, self.boundary_idxs = self.sample_boundary()
            self.interior_points = self.sample_interior()
            self.initial_points = self.sample_initial()

        if self.config['sampling_strategy'] == 'evo':
            if 'update_mesh_every' not in self.config.keys():
                logging.info('warning: `evo` sampling strategy chosen without specifying `update_mesh_every`, '
                             'setting to 1')
                self.config['update_mesh_every'] = 1
            # generate initial sample
            self.boundary_points, self.boundary_idxs = self.sample_boundary()
            self.interior_points = self.sample_interior()
            self.initial_points = self.sample_initial()

    def prepare_lbfgs(self):
        if self.config.get('n_lbfgs_steps', 0) > 0:
            if 'update_mesh_every_lbfgs' not in self.config.keys():
                logging.info('warning: lbfgs is being used, but `update_mesh_every_lbfgs` was not specified and is being set to 1')
                self.config['update_mesh_every_lbfgs'] = 1

    """methods for getting loss functions"""

    def get_residual_loss_func(self) -> Callable:
        def get_residual_loss(params: FrozenDict, interior_points: Float[Array, "n_points n_inputs"]) -> Float[Array, ""]:
            residual_loss = jnp.zeros(())
            residuals = self.pde(params, interior_points)
            for key in residuals.keys():
                residual_loss += self.component_weights[key] * (residuals[key] ** 2).mean()
            return residual_loss

        return get_residual_loss

    def get_boundary_loss_func(self) -> Callable:
        bc_fn = self.state.apply_fn if self.bc_fn is None else self.bc_fn

        def get_boundary_loss(params: FrozenDict, boundary_points: Float[Array, "n_boundary n_inputs"],
                              boundary_idxs: Bool[Array, "n_boundary n_bcs"]) -> Float[Array, ""]:
            bc_loss = jnp.zeros(())
            bc_fields = bc_fn(params, boundary_points)

            for bc_idx_idx, (bc_name, bc) in enumerate(self.bcs.items()):
                bc_idx = boundary_idxs[:, bc_idx_idx]
                # we have to filter down to only the points on a specific boundary because jax's JIT is annoying
                bc_loss = (bc.error(boundary_points, bc_fields).squeeze() * bc_idx) ** 2

                bc_loss = bc_loss.sum() / bc_idx.sum() * self.component_weights[bc_name]
                bc_loss += bc_loss
            return bc_loss

        def get_zero_boundary_loss(params: FrozenDict, boundary_points: Float[Array, "n_boundary n_inputs"],
                                   boundary_idxs: Bool[Array, "n_boundary n_bcs"]) -> Float[Array, ""]:
            return jnp.zeros(())

        if self.config['n_boundary'] > 0:
            return get_boundary_loss
        else:
            return get_zero_boundary_loss

    def get_initial_condition_loss_func(self) -> Callable:
        ic_fn = self.state.apply_fn if self.ic_fn is None else self.ic_fn

        def get_initial_condition_loss(params: FrozenDict, initial_points: Float[Array, "n_initial n_inputs"]) -> Float[Array, ""]:
            ic_loss = jnp.zeros(())
            ic_fields = ic_fn(params, initial_points)
            for ic_name, ic in self.ics.items():
                ic_loss += self.component_weights[ic_name] * (ic.error(initial_points, ic_fields).squeeze() ** 2).mean()
            return ic_loss

        def get_zero(params: FrozenDict, initial_points: jnp.ndarray):
            return jnp.zeros(())

        if isinstance(self.geom, GeometryXTime) and (self.config['n_initial'] > 0):
            return get_initial_condition_loss
        else:
            return get_zero

    def get_point_set_loss_func(self):
        psc_fn = self.state.apply_fn if self.psc_fn is None else self.psc_fn

        def get_point_set_loss(params: FrozenDict) -> Float[Array, ""]:
            point_set_loss = jnp.zeros(())
            for psc_id in self.pscs.keys():
                point_set_fields = psc_fn(params, self.anchor_points[psc_id])
                point_set_error = self.pscs[psc_id].error(None, point_set_fields)
                point_set_loss += self.component_weights[psc_id] * (point_set_error.squeeze() ** 2).mean()

            return point_set_loss

        return get_point_set_loss

    """other methods"""

    def get_loss_components(self) -> Dict[str, np.ndarray]:
        interior_points = jnp.array(self.geom.random_points(self.config['n_interior']))
        if self.config['n_boundary']:
            boundary_points = self.geom.random_boundary_points(self.config['n_boundary'])
        else:
            boundary_points = None
        if isinstance(self.geom, GeometryXTime) and (self.config['n_initial'] > 0):
            initial_points = jnp.array(self.geom.random_initial_points(self.config['n_initial']))
        else:
            initial_points = None

        @jit
        def get_residuals(points: Float[Array, "n_points n_inputs"]) -> Dict[str, jnp.ndarray]:
            return self.pde(self.state.params, points)

        # some problems impose conditions on functions of state variables
        bc_fn = jit(self.state.apply_fn) if self.bc_fn is None else jit(self.bc_fn)
        ic_fn = jit(self.state.apply_fn) if self.ic_fn is None else jit(self.ic_fn)
        psc_fn = jit(self.state.apply_fn) if self.psc_fn is None else jit(self.psc_fn)

        loss_components = get_residuals(interior_points)

        for psc_id in self.pscs.keys():
            point_set_fields = psc_fn(self.state.params, self.anchor_points[psc_id])
            loss_components[psc_id] = self.pscs[psc_id].error(None, point_set_fields)

        for bc_name, bc in self.bcs.items():
            bc_points = bc.filter(boundary_points)

            if bc_points.shape[0] > 0:
                bc_points = jnp.array(bc_points)
                bc_fields = bc_fn(self.state.params, bc_points)
                loss_components[bc_name] = bc.error(bc_points, bc_fields)
            else:
                loss_components[bc_name] = np.nan

        for ic_id in self.ics.keys():
            ic_fields = ic_fn(self.state.params, initial_points)
            loss_components[ic_id] = self.ics[ic_id].error(initial_points, ic_fields)

        loss_components = {key: float((np.array(value).squeeze()**2).mean()) for key, value in loss_components.items()}
        return loss_components

    def _validate_keys(self):
        assert self.pscs.keys() == self.anchor_points.keys(), "you need an array of anchor points for every " \
                                                              "point-set condition!!!"
        keys = set().union(self.residual_names, self.bcs.keys(), self.pscs.keys(), self.ics.keys())
        assert self.component_weights.keys() == keys, "you need a `component_weight` for every PDE, IC, BC, and PSC!!!"

    def get_resolution_info(self, include_time=True) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """returns summary statistics about
        (1) the distribution of distances between interior points in the space/time domain
        (2) summary statistics about the points themselves

        this can be used to infer properties of the "resolution" defined by our set of points

        distances treat time as another coordinate and so require that all dimensions be non-dimensionalized to enable
        meaningful comparison. (unless `include_time=False`, in which case the time coordinate is stripped out
        """
        interior_points = np.array(self.sample_interior())
        # strip out the time dimension if we want to (assuming that time is the last coordinate!!!)
        if not include_time and 'geometry' in dir(self.geom):
            interior_points = interior_points[:, :-1]

        dist_mat = distance_matrix(interior_points, interior_points)
        distances = dist_mat[np.triu_indices_from(dist_mat, k=1)]
        return {'distances': pd.Series(distances).describe(), 'points': pd.DataFrame(interior_points).describe()}


class ComponentWeightedPDE(PartialDiffEq):
    def __init__(self,
                 pde: Callable,
                 residual_names: list,
                 geom: Union[Geometry, GeometryXTime],
                 bcs: Dict[str, BC],
                 ics: Dict[str, IC],
                 pscs: Dict[str, PointSetBC],
                 anchor_points: Dict[str, jnp.ndarray],
                 model: Callable,
                 component_weights: Dict[str, float],
                 config: Dict,
                 metric_functions: Dict[str, Callable],
                 additional_keys: List[str],
                 additional_info: List[str],
                 bc_fn: Callable = None,
                 ic_fn: Callable = None,
                 psc_fn: Callable = None):
        """
        solves the PDE system
                N_i(u)(x) = 0,      x in Omega subseteq R^m
                u_i(x) = b_ij(x),   x in Gamma_j

            where `N_i` is the `i`th residual and `b_ij` is imposed on `u_i` for `j`th boundary `Gamma_j`

        using a soft-weighting approach:

                w_hat = arg min_w L(w)
                L(w) = sum_i lambda_i ||N_i(w)||^2 + sum_{i,j} lambda_{i,j} ||B_{i,j}(w)||^2

            where `lambda_i` and `lambda_{i,j}` are fixed nonnegative weights

        """
        super().__init__(pde, residual_names, geom, bcs, ics, pscs, anchor_points, model, component_weights,
                         config, metric_functions, additional_keys, additional_info, bc_fn, ic_fn, psc_fn)

    def _train(self):
        state, train_step = self.state, self.get_train_step()

        @jit
        def residual_norms(params: FrozenDict, points: Float[Array, "n_points n_inputs"]) -> jnp.ndarray:
            pde = self.pde(params, points)
            return sum(self.component_weights[term] * (_pde ** 2) for term, _pde in pde.items())

        if self.config['optimizer_interface'] == 'optax':
            key_r, diag_old = None, None
        elif self.config['optimizer'] == 'adahessian':
            key_r, diag_old = random.PRNGKey(self.config['key_r']), None
        elif self.config['optimizer'] == 'diagsg':
            key_r, diag_old = random.PRNGKey(self.config['key_r']), tree_map(jnp.ones_like, state.params)
        else:
            raise ValueError('unsupported optimizer specified!!!')

        for epoch in tqdm(range(self.config['n_epochs'])):
            interior_points, boundary_points, boundary_idxs, initial_points = \
                self.get_points(epoch, self.config['update_mesh_every'], residual_norms)

            state, loss, _, diag_old, key_r = train_step(state, state.params, diag_old,
                                                         interior_points,
                                                         boundary_points, boundary_idxs,
                                                         initial_points,
                                                         key_r)
            self.state = state
            self.log_loss_records(epoch, loss)

            if epoch % self.config['log_every'] == 0:
                self.log_error_records(epoch)
                self.log_metric_records(epoch)
            if epoch % self.config['save_every'] == 0:
                self.save_params(epoch)
                self.save_records()

    def get_train_step(self):
        bc_loss_func = self.get_boundary_loss_func()
        ic_loss_func = self.get_initial_condition_loss_func()
        pde_loss_func = self.get_residual_loss_func()
        psc_loss_func = self.get_point_set_loss_func()

        def loss_fn(params: FrozenDict, interior_points: Float[Array, "n_interior n_inputs"],
                    boundary_points: Float[Array, "n_boundary n_inputs"], boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                    initial_points: Float[Array, "n_initial n_inputs"]):
            return (bc_loss_func(params, boundary_points, boundary_idxs)
                    + ic_loss_func(params, initial_points)
                    + pde_loss_func(params, interior_points)
                    + psc_loss_func(params))

        grad_fn = value_and_grad(loss_fn, argnums=0)
        hutchinson = get_hutchinson(loss_fn, mode='pde')
        estimate_diag = get_estimate_diag(grad_fn, C_0, GAMMA, ALPHA_0)

        @jit
        def train_step_optax(state: TrainState, params: FrozenDict, _diag_old: FrozenDict,
                             interior_points: Float[Array, "n_interior n_inputs"],
                             boundary_points: Float[Array, "n_boundary n_inputs"],
                             boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                             initial_points: Float[Array, "n_initial n_inputs"], _
                             ) -> Tuple[TrainState, Float[Array, ""], FrozenDict, None, None]:
            """training step for any optimizer with a default optax interface"""
            # calculate gradient updates
            loss, grads = grad_fn(params, interior_points, boundary_points, boundary_idxs, initial_points)
            # update network state
            state = state.apply_gradients(grads=grads)
            return state, loss, grads, None, None

        @jit
        def train_step_diagsg(state: TrainState, params: FrozenDict, _diag_old: FrozenDict,
                              interior_points: Float[Array, "n_interior n_inputs"],
                              boundary_points: Float[Array, "n_boundary n_inputs"],
                              boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                              initial_points: Float[Array, "n_initial n_inputs"], _key_r
                              ) -> Tuple[TrainState, Float[Array, ""], FrozenDict, FrozenDict, jnp.ndarray]:
            """training step for the diagonal hessian stochastic gradient (diagSG) method based on simultaneous
            perturbation.
            for details see:
            - https://ieeexplore.ieee.org/document/9400243
            - https://jhuapl.box.com/s/43bpk1914hdyd7npv2b8ld9siznze52f
            """
            # calculate gradient updates
            loss, grads = grad_fn(params, interior_points, boundary_points, boundary_idxs, initial_points)
            # estimate hessian diagonal
            diag = estimate_diag(state, params, _diag_old,
                                 interior_points, boundary_points, boundary_idxs, initial_points, _key_r)
            grads = trees.divide(grads, diag)
            # take gradient step
            state = state.apply_gradients(grads=grads)
            return state, loss, grads, _diag_old, _key_r

        @jit
        def train_step_adahessian(state: TrainState, params: FrozenDict,
                                  interior_points: Float[Array, "n_interior n_inputs"],
                                  boundary_points: Float[Array, "n_boundary n_inputs"],
                                  boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                                  initial_points: Float[Array, "n_initial n_inputs"], _key_r
                                  ) -> Tuple[TrainState, Float[Array, ""], FrozenDict, None, jnp.ndarray]:
            # get rademacher sample to approximate the diagonal of the hessian
            # TODO: we can sample more than one rademacher variable for a better approximation if desired
            # TODO: can we make an optax interface for adahessian and remove the need for this function?
            _key_r, subkey = random.split(_key_r)
            hessian_diag = hutchinson(subkey, params, interior_points, boundary_points, boundary_idxs, initial_points)
            # calculate gradient updates
            loss, grads = grad_fn(params, interior_points, boundary_points, boundary_idxs, initial_points)
            # update network state
            state = state.apply_gradients(grads=(grads, hessian_diag))  # this is a hack for our custom-made optimizer
            return state, loss, grads, None, _key_r

        if self.config['optimizer_interface'] == 'optax':
            return train_step_optax
        elif self.config['optimizer_interface'] == 'adahessian':
            return train_step_adahessian
        elif self.config['optimizer_interface'] == 'diagsg':
            return train_step_diagsg
        else:
            raise ValueError('invalid optimizer specified!!!')


class GatedComponentWeightedPDE(PartialDiffEq):
    def __init__(self,
                 pde: Callable,
                 residual_names: list,
                 geom: GeometryXTime,
                 bcs: Dict[str, BC],
                 ics: Dict[str, IC],
                 pscs: Dict[str, PointSetBC],
                 anchor_points: Dict[str, jnp.ndarray],
                 model: Callable,
                 component_weights: Dict[str, float],
                 config: Dict,
                 metric_functions: Dict[str, Callable],
                 additional_keys: List[str],
                 additional_info: List[str],
                 bc_fn: Callable = None,
                 ic_fn: Callable = None,
                 psc_fn: Callable = None):
        """
        solves the PDE system
                N_i(u)(x) = 0,      x in Omega subseteq R^m
                u_i(x) = b_ij(x),   x in Gamma_j

            where `N_i` is the `i`th residual and `b_ij` is imposed on `u_i` for `j`th boundary `Gamma_j`

        using a soft-weighting approach:

                w_hat = arg min_w L(w)
                L(w) = sum_i lambda_i ||N_i(w)||^2 + sum_{i,j} lambda_{i,j} ||B_{i,j}(w)||^2

            where `lambda_i` and `lambda_{i,j}` are fixed nonnegative weights

        default hyperparameters:
        - gating_scale = 5
        - gating_lr = 1e-3
        - gating_initial_offset = -0.5
        - gating_update_every = 100
        - gating_update_scale = 20
        """
        additional_keys += ['gating_scale', 'gating_lr', 'gating_initial_offset', 'gating_update_every',
                            'gating_update_scale']
        additional_info += ['gating_offset']
        super().__init__(pde, residual_names, geom, bcs, ics, pscs, anchor_points, model, component_weights,
                         config, metric_functions, additional_keys, additional_info, bc_fn, ic_fn, psc_fn)

    def gating_function(self, points: Float[Array, "n_points n_inputs"], offset: jnp.ndarray) -> Float[Array, "n_points"]:
        t_scaled = points[:, -1] / (self.geom.timedomain.t1 - self.geom.timedomain.t0)
        return 0.5 * (1. - jnp.tanh(self.config['gating_scale'] * (t_scaled - offset)))

    def get_residual_loss_func(self) -> Callable:
        def get_residual_loss(params: FrozenDict, interior_points: Float[Array, "n_points n_inputs"], offset: jnp.ndarray,
                              ) -> Float[Array, ""]:
            residual_loss = jnp.zeros(())
            residuals = self.pde(params, interior_points)
            for key in residuals.keys():
                residual_loss += (self.component_weights[key] *
                                  ((residuals[key] ** 2) * self.gating_function(interior_points, offset)).mean())
            return residual_loss

        return get_residual_loss

    def _train(self):
        state, train_step = self.state, self.get_train_step()
        offset = jnp.array(self.config['gating_initial_offset'])

        @jit
        def residual_norms(params: FrozenDict, points: Float[Array, "n_points n_inputs"], _offset: jnp.ndarray) -> jnp.ndarray:
            pde = self.pde(params, points)
            return sum(self.component_weights[term] * (_pde ** 2) * self.gating_function(points, _offset)
                       for term, _pde in pde.items())

        if self.config['optimizer_interface'] == 'optax':
            key_r, diag_old = None, None
        else:
            raise ValueError('unsupported optimizer specified!!!')

        for epoch in tqdm(range(self.config['n_epochs'])):
            interior_points, boundary_points, boundary_idxs, initial_points = \
                self.get_points(epoch, self.config['update_mesh_every'], residual_norms)

            state, loss, _, diag_old, key_r = train_step(state, state.params, offset, diag_old,
                                                         interior_points,
                                                         boundary_points, boundary_idxs,
                                                         initial_points,
                                                         key_r)
            # update gating function offset
            if (epoch % self.config['gating_update_every']) == 0:
                update = jnp.exp(-self.config['gating_update_scale'] * residual_norms(state.params, interior_points, offset))
                offset = offset + self.config['gating_lr'] * update
            self.state = state
            self.log_loss_records(epoch, loss, **{'gating_offset': float(offset[0])})

            if epoch % self.config['log_every'] == 0:
                self.log_error_records(epoch)
                self.log_metric_records(epoch)
            if epoch % self.config['save_every'] == 0:
                self.save_params(epoch)
                self.save_records()

    def get_train_step(self):
        bc_loss_func = self.get_boundary_loss_func()
        ic_loss_func = self.get_initial_condition_loss_func()
        pde_loss_func = self.get_residual_loss_func()
        psc_loss_func = self.get_point_set_loss_func()

        def loss_fn(params: FrozenDict, interior_points: Float[Array, "n_interior n_inputs"], offset: jnp.ndarray,
                    boundary_points: Float[Array, "n_boundary n_inputs"], boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                    initial_points: Float[Array, "n_initial n_inputs"]):
            return (bc_loss_func(params, boundary_points, boundary_idxs)
                    + ic_loss_func(params, initial_points)
                    + pde_loss_func(params, interior_points, offset)
                    + psc_loss_func(params))

        grad_fn = value_and_grad(loss_fn, argnums=0)

        @jit
        def train_step_optax(state: TrainState, params: FrozenDict, offset: jnp.ndarray, _diag_old: FrozenDict,
                             interior_points: Float[Array, "n_interior n_inputs"],
                             boundary_points: Float[Array, "n_boundary n_inputs"],
                             boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                             initial_points: Float[Array, "n_initial n_inputs"], _
                             ) -> Tuple[TrainState, Float[Array, ""], FrozenDict, None, None]:
            """training step for any optimizer with a default optax interface"""
            # calculate gradient updates
            loss, grads = grad_fn(params, interior_points, offset, boundary_points, boundary_idxs, initial_points)
            # update network state
            state = state.apply_gradients(grads=grads)
            return state, loss, grads, None, None

        if self.config['optimizer_interface'] == 'optax':
            return train_step_optax
        else:
            raise ValueError('invalid optimizer specified!!!')


class ConstraintSatisfactionPDE(PartialDiffEq):
    """ 'total' augmented-lagrangian method for solving PINNs with exact BCs.

    defines the loss:

        L(w, lambda) = sum_i lambda_i sum_{x in D} |N_i(u_w)(x)|^2
                       + sum_{i, j} lambda_ij sum_{x in Gamma_j} phi(u_{w,i}(x) - b_ij(x))
                       + mu / 2 sum_{i,j} sum_{x in Gamma_j} |phi(u_{w,i}(x) - b_ij(x))|^2

    where:

    - `u_w` is the NN with weights `w`, `N_i` is the residual operator for the `i`th component `u_{w,i}`, `b_ij` is the
      BC function for the `i`th component `u_{w,i}` and the `j`th boundary `Gamma_j`
    - `lambda_ij` is the lagrange multiplier for a constraint on the `j`th boundary `Gamma_j` for the `i`th component
      `u_{w,i}`
    - `lambda_i` is the lagrange multiplier for the `i`th residual
    - `mu` is a fixed penalty terms

    algorithm overview:

    for t = 0, 1, ...
        w^{t+1} <- min_w L(w, lambda^t)  # approximately solve the minimization problem with current multipliers

        # is the penalty larger than the constraint violation tolerances?
        for each i, if pi_i(w^{t+1}) >= max(0.25 eta_i, epsilon)
            eta^{t+1}_i <- pi_i(w^{t+1})  # record current violation
            mu^{t+1}_i <- min(2 mu^t_i, u_max)  # grow the penalty term
            lambda^{t+1}_i <- lambda^t_i + mu^{t+1}_i pi_i(w^{t+1})  # grow the constraint's multiplier

        # is the penalty larger than the constraint violation tolerances?
        for each (i,j), if pi_ij(w^{t+1}) >= max(0.25 eta_ij, epsilon)
            eta^{t+1}_ij <- pi_ij(w^{t+1})  # record current violation
            mu^{t+1}_ij <- min(2 mu^t_ij, u_max)  # grow the penalty term
            lambda^{t+1}_ij <- lambda^t_ij + mu^{t+1}_ij pi_ij(w^{t+1})  # grow the constraint's multiplier

    here:
    - pi_i(w) = sum_{x in Omega} phi(N_i(u_w)(x))
    - pi_ij(w) = sum_{x in Gamma_j} phi(u_{w,i}(x) - b_ij(x))
    - tau_i(w) = sum_{x in Omega} |phi(N_i(u_w)(x))|^2
    - tau_ij(w) = sum_{x in Gamma_j} |phi(u_{w,i}(x) - b_ij(x))|^2

    the method is similar to https://arxiv.org/abs/2102.04626 and https://doi.org/10.1016/j.jcp.2022.111301, but we
    assign a lagrange multiplier to each residual (not to specific points)

    See here for more details: https://jhuapl.app.box.com/file/1127979381836
    """
    def __init__(self,
                 pde: Callable,
                 residual_names: list,
                 geom: Union[Geometry, GeometryXTime],
                 bcs: Dict[str, BC],
                 ics: Dict[str, IC],
                 pscs: Dict[str, PointSetBC],
                 anchor_points: Dict[str, jnp.ndarray],
                 model: Callable,
                 component_weights: Dict[str, float],
                 config: Dict,
                 metric_functions: Dict[str, Callable],
                 additional_keys: List[str],
                 additional_info: List[str],
                 bc_fn: Callable = None,
                 ic_fn: Callable = None,
                 psc_fn: Callable = None,
                 dist_fn: Callable = None
                 ):
        additional_keys += ['constraints_tolerance', 'max_penalty', 'n_inner_steps']
        additional_info += list(set().union(residual_names, bcs.keys(), ics.keys(), pscs.keys()))
        super().__init__(pde, residual_names, geom, bcs, ics, pscs, anchor_points, model, component_weights,
                         config, metric_functions, additional_keys, additional_info, bc_fn, ic_fn, psc_fn)

        # initialize lagrange multipliers at 0, penalty weights at 1, and constraint violations at 0.
        self.lambda_, self.mu, self.eta = {}, {}, {}
        for key in self.component_weights.keys():
            self.lambda_[key], self.mu[key], self.eta[key] = 0., 1., 0.
        del self.component_weights  # replaced by `lambda_` in this formulation
        self.eps = float(self.config['constraints_tolerance'])  # tolerance for violating constraints
        self.mu_max = float(self.config['max_penalty'])  # max value allowed for mu
        # distance function `phi` for assessing constraint violation
        if dist_fn is None:
            self.dist_fn = lambda x: x
        else:
            self.dist_fn = dist_fn

    def get_residual_loss_funcs(self, dist_fn=None) -> Dict[str, Callable]:
        if dist_fn is None:
            dist_fn = self.dist_fn

        def get_residual_loss_func(residual_name: str) -> Callable:
            """return mean_x phi(N_i(u_w)(x)) for a residual i"""
            def get_residual_loss(params: FrozenDict, interior_points: Float[Array, "n_points n_inputs"]) -> Float[Array, ""]:
                residual = self.pde(params, interior_points)[residual_name].squeeze()
                return dist_fn(residual).mean()

            return get_residual_loss

        return {residual_name: get_residual_loss_func(residual_name) for residual_name in self.residual_names}

    def get_boundary_loss_funcs(self, dist_fn=None) -> Dict[str, Callable]:
        bc_fn = self.state.apply_fn if self.bc_fn is None else self.bc_fn
        bc_names = list(self.bcs.keys())
        if dist_fn is None:
            dist_fn = self.dist_fn

        def get_boundary_loss_func(bc_name: str) -> Callable:
            """returns `mean_x phi(u_{w,i}(x) - b_ij(x))` for a given boundary `j` and state variable `i`"""
            def get_boundary_loss(params: FrozenDict, boundary_points: Float[Array, "n_boundary n_inputs"],
                                  boundary_idxs: Bool[Array, "n_boundary n_bcs"]) -> Float[Array, ""]:
                bc = self.bcs[bc_name]
                bc_fields = bc_fn(params, boundary_points)  # bc_fn values on boundary
                bc_idx_idx = bc_names.index(bc_name)  # locate which BC we care about
                bc_idx = boundary_idxs[:, bc_idx_idx]  # identify which of `boundary_points` are on the boundary
                bc_loss = dist_fn(bc.error(boundary_points, bc_fields).squeeze()) * bc_idx
                return bc_loss.sum() / bc_idx.sum()

            return get_boundary_loss

        def get_zero_boundary_loss(params: FrozenDict, boundary_points: Float[Array, "n_boundary n_inputs"],
                                   boundary_idxs: Bool[Array, "n_boundary n_bcs"]) -> Float[Array, ""]:
            return jnp.zeros(())

        if self.config['n_boundary'] > 0:
            return {bc_name: get_boundary_loss_func(bc_name) for bc_name in self.bcs.keys()}
        else:
            return {bc_name: get_zero_boundary_loss for bc_name in self.bcs.keys()}

    def get_initial_condition_loss_funcs(self, dist_fn=None) -> Dict[str, Callable]:
        ic_fn = self.state.apply_fn if self.ic_fn is None else self.ic_fn
        if dist_fn is None:
            dist_fn = self.dist_fn

        def get_initial_condition_loss_func(ic_name: str) -> Callable:
            """returns `mean_x phi(u_{w,i}(x) - b_ij(x))` for a given temporal boundary `j` and state variable `i`"""
            def get_initial_condition_loss(params: FrozenDict,
                                           initial_points: Float[Array, "n_initial n_inputs"]) -> Float[Array, ""]:
                ic = self.ics[ic_name]
                ic_fields = ic_fn(params, initial_points)  # ic_fn values on temporal boundary
                ic_loss = dist_fn(ic.error(initial_points, ic_fields).squeeze()).mean()
                return ic_loss

            return get_initial_condition_loss

        def get_zero(params: FrozenDict, initial_points: jnp.ndarray):
            return jnp.zeros(())

        if isinstance(self.geom, GeometryXTime) and (self.config['n_initial'] > 0):
            return {ic_name: get_initial_condition_loss_func(ic_name) for ic_name in self.ics.keys()}
        else:
            return {ic_name: get_zero for ic_name in self.ics.keys()}

    def get_point_set_loss_funcs(self, dist_fn=None):
        psc_fn = self.state.apply_fn if self.psc_fn is None else self.psc_fn
        if dist_fn is None:
            dist_fn = self.dist_fn

        def get_point_set_loss_func(psc_id: str) -> Callable:
            """returns mean_{x in P_k} phi(u_{w,i} - b_k(x)) for a set `P_k` and a state variable `i`"""
            def get_point_set_loss(params: FrozenDict) -> Float[Array, ""]:
                psc = self.pscs[psc_id]
                psc_fields = psc_fn(params, self.anchor_points[psc_id])
                psc_loss = dist_fn(psc.error(None, psc_fields).squeeze()).mean()
                return psc_loss

            return get_point_set_loss

        return {psc_id: get_point_set_loss_func(psc_id) for psc_id in self.pscs.keys()}

    def get_squared_penalty_funcs(self) -> Dict[str, Callable]:
        """return `sum_{x in Omega} |phi(N_i(u_w)(x))|^2` for each `i` and
        `sum_{x in Gamma_j} |phi(u_{w,i}(x) - b_ij(x))|^2` for each `i,j`
        """
        def dist_fn(x: jnp.ndarray) -> jnp.ndarray: return self.dist_fn(x)**2

        residual_loss_funcs = self.get_residual_loss_funcs(dist_fn)
        bc_loss_funcs = self.get_boundary_loss_funcs(dist_fn)
        ic_loss_funcs = self.get_initial_condition_loss_funcs(dist_fn)
        psc_loss_funcs = self.get_point_set_loss_funcs(dist_fn)

        return {**residual_loss_funcs, **bc_loss_funcs, **ic_loss_funcs, **psc_loss_funcs}

    def get_train_step(self):
        residual_loss_funcs = self.get_residual_loss_funcs()
        bc_loss_funcs = self.get_boundary_loss_funcs()
        ic_loss_funcs = self.get_initial_condition_loss_funcs()
        psc_loss_funcs = self.get_point_set_loss_funcs()
        squared_penalty_funcs = self.get_squared_penalty_funcs()  # note that these are based around |phi(...)|^2

        def loss_fn(params: FrozenDict, interior_points: Float[Array, "n_interior n_inputs"],
                    boundary_points: Float[Array, "n_boundary n_inputs"],
                    boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                    initial_points: Float[Array, "n_initial n_inputs"],
                    mu: Dict[str, float], lambda_: Dict[str, float]) -> Float[Array, ""]:
            """
            calculates
            L(w, lambda) = sum_i lambda_i sum_{x in Omega} |N_i(u_w)(x)|^2 # TODO
                       + sum_{i, j} lambda_ij sum_{x in Gamma_j} phi(u_{w,i}(x) - b_ij(x))
                       + sum_i mu_i / 2 lambda_i sum_{x in Omega}
                       + sum_{i,j} mu_ij / 2 sum_{x in Gamma_j} |phi(u_{w,i}(x) - b_ij(x))|^2
            """
            loss = jnp.zeros(())

            for r_name in residual_loss_funcs.keys():
                loss += lambda_[r_name] * residual_loss_funcs[r_name](params, interior_points)
                loss += mu[r_name] * squared_penalty_funcs[r_name](params, interior_points)
            for bc_name in bc_loss_funcs.keys():
                loss += lambda_[bc_name] * bc_loss_funcs[bc_name](params, boundary_points, boundary_idxs)
                loss += mu[bc_name] * squared_penalty_funcs[bc_name](params, boundary_points, boundary_idxs)
            for ic_name in ic_loss_funcs.keys():
                loss += lambda_[ic_name] * ic_loss_funcs[ic_name](params, initial_points)
                loss += mu[ic_name] * squared_penalty_funcs[ic_name](params, initial_points)
            for psc_name in psc_loss_funcs.keys():
                loss += lambda_[psc_name] * psc_loss_funcs[psc_name](params)
                loss += mu[psc_name] * squared_penalty_funcs[psc_name](params)

            return loss

        grad_fn = value_and_grad(loss_fn, argnums=0)

        @jit
        def train_step_optax(state: TrainState, params: FrozenDict, _diag_old: FrozenDict,
                             interior_points: Float[Array, "n_interior n_inputs"],
                             boundary_points: Float[Array, "n_boundary n_inputs"],
                             boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                             initial_points: Float[Array, "n_initial n_inputs"],
                             mu: Dict[str, float], lambda_: Dict[str, float], _
                             ) -> Tuple[TrainState, Float[Array, ""], FrozenDict, None, None]:
            """training step for any optimizer with a default optax interface"""
            # calculate gradient updates
            loss, grads = grad_fn(params, interior_points, boundary_points, boundary_idxs, initial_points, mu, lambda_)
            # update network state
            state = state.apply_gradients(grads=grads)
            return state, loss, grads, None, None

        if self.config['optimizer_interface'] == 'optax':
            return train_step_optax, residual_loss_funcs, bc_loss_funcs, ic_loss_funcs, psc_loss_funcs, squared_penalty_funcs
        elif self.config['optimizer_interface'] == 'adahessian':
            raise NotImplementedError('adahessian not implemented yet!!!')
        elif self.config['optimizer_interface'] == 'diagsg':
            raise NotImplementedError('diagsg not implemented yet!!!')
        else:
            raise ValueError('invalid optimizer specified!!!')

    def save_multipliers_and_weights(self, epoch):
        file_name = 'multipliers.json' if epoch is None else f"multipliers_{epoch}.json"
        with open(os.path.join(self.paths['checkpoint'], file_name), 'w') as f:
            json.dump(self.lambda_, f)
        file_name = 'weights.json' if epoch is None else f"weights_{epoch}.json"
        with open(os.path.join(self.paths['checkpoint'], file_name), 'w') as f:
            json.dump(self.mu, f)

    def _train(self):
        state = self.state
        mu, lambda_ = self.mu, self.lambda_
        train_step, residual_loss_funcs, bc_loss_funcs, ic_loss_funcs, psc_loss_funcs, squared_penalty_funcs = self.get_train_step()

        @jit
        def residual_norms(params: FrozenDict, points: Float[Array, "n_points n_inputs"],
                           _lambda_: Dict[str, float]) -> jnp.ndarray:
            pde = self.pde(params, points)
            return sum(_lambda_[term] * (_pde ** 2) for term, _pde in pde.items())

        @jit
        def penalty_fns(params, _interior_points, _boundary_points, _boundary_idxs, _initial_points):
            _residual_vals = {residual_name: residual_loss_func(params, _interior_points)
                              for residual_name, residual_loss_func in residual_loss_funcs.items()}
            _bc_vals = {bc_name: bc_loss_func(params, boundary_points, boundary_idxs) for bc_name, bc_loss_func
                        in bc_loss_funcs.items()}
            _ic_vals = {ic_name: ic_loss_func(params, initial_points) for ic_name, ic_loss_func
                        in ic_loss_funcs.items()}
            _psc_vals = {psc_name: psc_loss_func(params, boundary_points, boundary_idxs) for psc_name, psc_loss_func
                         in psc_loss_funcs.items()}

            return {**_residual_vals, **_bc_vals, **_ic_vals, **_psc_vals}

        if self.config['optimizer_interface'] == 'optax':
            key_r, diag_old = None, None
        elif self.config['optimizer'] == 'adahessian':
            raise ValueError('unsupported optimizer specified!!!')
        elif self.config['optimizer'] == 'diagsg':
            key_r, diag_old = random.PRNGKey(self.config['key_r']), tree_map(jnp.ones_like, state.params)
        else:
            raise ValueError('unsupported optimizer specified!!!')

        for epoch in tqdm(range(self.config['n_epochs'])):
            interior_points, boundary_points, boundary_idxs, initial_points = \
                self.get_points(epoch, self.config['update_mesh_every'], residual_norms, lambda_)
            loss = None
            # inner optimization loop
            for _ in range(self.config['n_inner_steps']):
                state, loss, _, diag_old, key_r = train_step(state, state.params, diag_old,
                                                             interior_points,
                                                             boundary_points, boundary_idxs,
                                                             initial_points,
                                                             mu, lambda_,
                                                             key_r)

            # calculate current penalties for each constraint
            penalties = penalty_fns(state.params, interior_points, boundary_points, boundary_idxs, initial_points)

            # see if lagrange multipliers and penalty weights need to be updated
            for penalty_name, penalty_value in penalties.items():
                # is the current penalty sufficiently bad?
                if penalty_value >= max(0.25 * self.eta[penalty_name], self.eps):
                    self.eta[penalty_name] = penalty_value  # record current penalty
                    mu[penalty_name] = min(2. * mu[penalty_name], self.mu_max)  # update penalty weight
                    lambda_[penalty_name] += float(mu[penalty_name] * penalty_value)  # update multiplier

            self.state, self.mu, self.lambda_ = state, mu, lambda_
            self.log_loss_records(epoch, loss, **penalties)

            if epoch % self.config['log_every'] == 0:
                self.log_error_records(epoch)
                self.log_metric_records(epoch)
            if epoch % self.config['save_every'] == 0:
                self.save_params(epoch)
                self.save_records()
                self.save_multipliers_and_weights(epoch)
