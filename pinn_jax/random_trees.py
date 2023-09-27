# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp
from jax.tree_util import tree_map
from jax.random import split, normal, choice

from flax.core import FrozenDict

from typing import Callable

RADEMACHER_VALUES = jnp.array([-1.0, 1.0])


def rademacher(key, shape) -> jnp.ndarray:
    return choice(key, RADEMACHER_VALUES, shape)


def create_map_fn(key, sampler) -> Callable:
    key, subkey = split(key)

    def map_fn(p) -> jnp.ndarray:
        nonlocal key, subkey
        params = sampler(subkey, p.shape)
        key, subkey = split(key)
        return params
    return map_fn


def random_normal(key, ps: FrozenDict) -> FrozenDict:
    """generates a `FrozenDict` of random gaussian numbers of the same size as the entries of `ps`"""
    return tree_map(create_map_fn(key, normal), ps)


def random_rademacher(key, ps: FrozenDict) -> FrozenDict:
    """generates a `FrozenDict` of random rademacher variables of the same size as the entries of `ps`"""
    return tree_map(create_map_fn(key, rademacher), ps)
