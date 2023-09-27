# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp, lax, tree_map

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jaxtyping import Float, Array  # `Array` is an alias for `jax.numpy.ndarray`

from datetime import datetime
from typing import Callable
from inspect import signature


def get_logs_id() -> str:
    now = datetime.now()
    year, month, day, hour, minute, second = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H"), now.strftime('%M'), now.strftime('%S')
    return f'{year}-{month}-{day}-{hour}-{minute}-{second}'


def filter_dict(dict_to_filter: dict, callable_with_kwargs: Callable) -> dict:
    """filter the entries of `dict_to_filter` so that it only contains keyword-arguments of `callable_with_kwargs`"""
    sig = signature(callable_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter.get(filter_key) for filter_key in filter_keys}
    return filtered_dict


def get_nn_func(u_0: Float[Array, "n_xyt 1"], xyt: Float[Array, "n_xyt n_inputs"]) -> Callable:
    """nearest-neighbor interpolator for enforcing an initial condition from a uniform grid"""
    assert u_0.shape == (xyt.shape[0], 1), "target array must be 2d matrix of same size as input array!!!"
    sq_xyt_norm = (xyt ** 2).sum(axis=1)

    def ic_func(xyt_sample: Float[Array, "n_sample n_inputs"]) -> Float[Array, "n_sample n_xyt"]:
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 (x, y)
        dists = (xyt_sample ** 2).sum(axis=1)[:, None] + sq_xyt_norm[None, :] - 2. * lax.dot(xyt_sample, xyt.T)
        neighbor_idxs = dists.argmin(axis=1)
        return u_0[neighbor_idxs, :]  # return an `n x 1` array

    return ic_func


def get_intermediate_fn(u: nn.Module):
    def intermediate_fn(p: FrozenDict, x: jnp.ndarray):
        _intermediate_values = tree_map(jnp.asarray, u.apply(p, x, capture_intermediates=True, mutable=['intermediates']
                                                            )[1])['intermediates']
        intermediate_values = {}
        for key in _intermediate_values.keys():
            if key == '__call__':
                intermediate_values['output'] = _intermediate_values[key][0]
            else:
                try:
                    intermediate_values[key] = _intermediate_values[key]['__call__'][0].val
                except AttributeError:
                    intermediate_values[key] = _intermediate_values[key]['__call__'][0]
        return intermediate_values
    return intermediate_fn
