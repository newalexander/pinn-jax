# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp
from jax.tree_util import tree_map, tree_flatten
from flax.core import FrozenDict
from numpy import prod

from typing import List

ONE = jnp.ones(())
ZERO = jnp.zeros(())


def count(xs: FrozenDict) -> int:
    """count the total number of parameters in a FrozenDict"""
    sizes = tree_map(lambda x: prod(x.shape), xs)
    leaves, _ = tree_flatten(sizes)
    return sum(leaf for leaf in leaves)


def inner_product(xs: FrozenDict, ys: FrozenDict) -> jnp.ndarray:
    """inner product of two sets of variables"""
    tree = tree_map(lambda x, y: (x * y).sum(), xs, ys)
    leaves, _ = tree_flatten(tree)
    return sum(leaf for leaf in leaves)


def normalize(vs: FrozenDict) -> FrozenDict:
    """normalize a set of vectors wrt their frobenius norm"""
    s = inner_product(vs, vs)**0.5
    return tree_map(lambda x: x / (s+1e-6), vs)


def fill(xs: FrozenDict, value=ONE) -> FrozenDict:
    """returns a `FrozenDict` of constant values, the same shapes as `xs`"""
    return tree_map(lambda x: x*ZERO + value, xs)


def add(xs: FrozenDict, ys: FrozenDict, alpha=ONE, beta=ONE) -> FrozenDict:
    """returns (alpha x + beta y) for every (x, y) pair in (xs, ys)"""
    return tree_map(lambda x, y: alpha * x + beta * y, xs, ys)


def multiply(xs: FrozenDict, ys: FrozenDict, alpha=ONE, beta=ONE) -> FrozenDict:
    """returns (alpha x * beta y) for every (x, y) pair in (xs, ys)"""
    return tree_map(lambda x, y: (alpha * x) * (beta * y), xs, ys)


def divide(xs: FrozenDict, ys: FrozenDict, alpha=ONE, beta=ONE) -> FrozenDict:
    """returns (alpha x / beta y) for every (x, y) pair in (xs, ys)"""
    return tree_map(lambda x, y: (alpha * x) / (beta * y), xs, ys)


def scale(params: FrozenDict, alpha: jnp.ndarray) -> FrozenDict:
    """param <- alpha * param for every param in params"""
    return tree_map(lambda x: alpha * x, params)


def orthonormalize(w: FrozenDict, v_list: List[FrozenDict]) -> FrozenDict:
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = add(w, v, beta=-inner_product(w, v))
    return normalize(w)
