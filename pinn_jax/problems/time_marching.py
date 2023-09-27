# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import numpy as np
import os
import logging

from jax import numpy as jnp
from flax.training.train_state import TrainState

from pinn_jax.problems.pdes import PartialDiffEq
from pinn_jax.problems.odes import OrdinaryDiffEq
from pinn_jax.utils import filter_dict
from pinn_jax.derivatives import get_batch_jacobian
from pinn_jax.bcs import FunctionIC

from typing import Callable
from abc import ABC, abstractmethod
from copy import deepcopy


class AbstractTimeMarchingPDE(ABC):
    def __init__(
        self, problem_constructor: Callable[..., PartialDiffEq], time_points: np.ndarray, **kwargs
    ):
        self.base_model_path = kwargs["config"]["model_path"]
        self.base_geom = kwargs["geom"]
        self.time_points = time_points

        self.problems = []

        for time_idx in range(len(self.time_points) - 1):
            t_min, t_max = self.time_points[time_idx], self.time_points[time_idx + 1]

            curr_geom = deepcopy(self.base_geom)
            curr_geom.timedomain.t0 = t_min
            curr_geom.timedomain.t1 = t_max

            kwargs["config"]["model_path"] = os.path.join(
                self.base_model_path, f"t_{time_idx}"
            )
            kwargs["geom"] = curr_geom
            # TODO: update
            kwargs['ics']['u_0'].geom = deepcopy(curr_geom)
            # kwargs['bcs']['u_top'].geom = deepcopy(curr_geom)
            # kwargs['bcs']['u_bottom'].geom = deepcopy(curr_geom)

            # TODO:
            # Update metric functions for current problem (time domain interval)


            filtered_kwargs = filter_dict(kwargs, problem_constructor)

            # self.problems.append(problem_constructor(**filtered_kwargs))
            self.problems.append(deepcopy(problem_constructor(**filtered_kwargs)))

    @abstractmethod
    def fit(self):
        pass

    def _get_ic_fcn(self, state):
        def ic_fcn(points):
            return state.apply_fn(state.params, points)

        return ic_fcn

    def _update_problem_data(self, problem_idx):
        """prepare for solving the next time interval by setting the next problem's PINN's weights to the trained ones
        and generating the initial conditions"""
        if problem_idx < len(self.problems) - 1:
            self.problems[problem_idx + 1].state = TrainState.create(
                apply_fn=self.problems[problem_idx].state.apply_fn,
                params=self.problems[problem_idx].state.params,
                tx=self.problems[problem_idx].state.tx,
            )

            # Update the ics
            for ic_name in self.problems[problem_idx + 1].ics:
                curr_ic = self.problems[problem_idx].ics[ic_name]
                # Update next_ic geom from next problem
                next_ic = FunctionIC(
                    geom=deepcopy(self.problems[problem_idx+1].ics[ic_name].geom),
                    func=self._get_ic_fcn(self.problems[problem_idx].state),
                    component=curr_ic.component
                )

                # next_ic = FunctionIC(
                #     geom=curr_ic.geom,
                #     func=self._get_ic_fcn(self.problems[problem_idx].state),
                #     component=curr_ic.component,
                # )
                self.problems[problem_idx + 1].ics[ic_name] = next_ic
            # Note: Not updating the id_fds, as that functionality may be depreciated.


class AbstractTimeMarchingODE(ABC):
    def __init__(self,
                 problem_constructor: Callable[..., OrdinaryDiffEq],
                 time_points: np.ndarray,
                 **kwargs):
        self.base_model_path = kwargs['config']['model_path']
        self.time_points = time_points

        # create a `Problem` for each time interval `[t_n, t_{n+1}]` in `time_points`
        self.problems = []
        for time_idx in range(len(time_points)-1):
            # each interval has its own set of evaluation time
            t_min, t_max = time_points[time_idx], time_points[time_idx + 1]
            train_times = np.linspace(t_min, t_max, num=kwargs['config']['n_interior'])
            eval_times = (train_times[1:] + train_times[:-1]) / 2
            eval_times = jnp.array(eval_times[:, None])

            # each subproblem gets its own sub-directory
            kwargs['config']['model_path'] = os.path.join(self.base_model_path, f't_{time_idx}')
            # filter kwargs based on what the current constructor needs
            filtered_kwargs = filter_dict(kwargs, problem_constructor)
            filtered_kwargs['time_min'], filtered_kwargs['time_max'] = t_min, t_max
            filtered_kwargs['eval_times'] = eval_times

            self.problems.append(problem_constructor(**filtered_kwargs))

    @abstractmethod
    def fit(self):
        pass

    def _update_problem_data(self, problem_idx):
        """prepare for solving the next time interval by setting the next problem's PINN's weights to the trained ones
        and generating the initial conditions"""
        if problem_idx < len(self.problems) - 1:
            t_max = jnp.ones((1, 1)) * self.problems[problem_idx].interior_points[-1].squeeze()
            self.problems[problem_idx+1].state = TrainState.create(apply_fn=self.problems[problem_idx].state.apply_fn,
                                                                   params=self.problems[problem_idx].state.params,
                                                                   tx=self.problems[problem_idx].state.tx)
            self.problems[problem_idx+1].ic_vals = self.problems[problem_idx].state.apply_fn(
                self.problems[problem_idx].state.params, t_max)
            if self.problems[problem_idx+1].ic_fd_vals is not None:
                batch_jacobian = get_batch_jacobian(self.problems[problem_idx].model.apply)
                self.problems[problem_idx+1].ic_fd_vals = batch_jacobian(
                    self.problems[problem_idx].state.params, t_max)


class FineTuningTimeMarchingPDE(AbstractTimeMarchingPDE):
    """solve a PDE defined over a long time interval `[0, t_N]` by breaking it into iterative subproblems defined
    over intervals `[0, t_1], [t_1, t_2], ..., [t_{N-1}, t_N]`

    after solving a problem on `[t_{n-1}, t_n]`, the trained network predicts the state at `t_n`, and this is used as
    the initial condition for solving on `[t_n, t_{n+1}]`.
    """

    def __init__(
        self, problem_constructor: Callable[..., PartialDiffEq], time_points: np.ndarray, **kwargs
    ):
        super().__init__(problem_constructor, time_points, **kwargs)

    def fit(self):
        for problem_idx, _ in enumerate(self.problems):
            logging.info(f"problem_idx = {problem_idx}")
            self.problems[problem_idx].fit()
            self._update_problem_data(problem_idx)


class FineTuningTimeMarchingODE(AbstractTimeMarchingODE):
    """solve an ODE defined over a long time interval `[0, t_N]` by breaking it into iterative subproblems defined
    over intervals `[0, t_1], [t_1, t_2], ..., [t_{N-1}, t_N]`

    after solving a problem on `[t_{n-1}, t_n]`, the trained network predicts the state at `t_n`, and this is used as
    the initial condition for solving on `[t_n, t_{n+1}]`.

    `problem_constructor` should define an `OrdinaryDiffEq` problem of some sort
    """
    def __init__(self,
                 problem_constructor: Callable[..., OrdinaryDiffEq],
                 time_points: np.ndarray,
                 **kwargs):
        super().__init__(problem_constructor, time_points, **kwargs)

    def fit(self):
        for problem_idx, _ in enumerate(self.problems):
            logging.info(f'problem_idx = {problem_idx}')
            self.problems[problem_idx].fit()
            self._update_problem_data(problem_idx)
