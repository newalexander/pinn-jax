# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from pinn_jax.derivatives import hvp
from pinn_jax import trees

from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp, grad, vmap, jit


def get_curvature_1(f, num=50):
    """estimates curvature by computing `int_0^1 g(ps1)^T H(ps1 + t(ps2 - ps1)) g(ps1) dt`"""
    ts = jnp.linspace(0.0, 1.0, num=num)

    @jit
    def curvature(ps1: FrozenDict, ps2: FrozenDict) -> jnp.ndarray:
        gs1 = grad(f)(ps1)
        difference = trees.add(ps1, ps2, beta=-trees.ONE)  # ps1 - ps2

        def integrand(t):
            convex_combo = trees.add(ps1, difference, beta=t)  # ps1 + t(ps2 - ps1)
            _, hvp_value = hvp(f, convex_combo, gs1)
            return trees.inner_product(gs1, hvp_value)

        hvps = vmap(integrand)(ts)
        return jnp.trapz(hvps, ts)

    return curvature


def get_curvature_2(f, num=50):
    """computes `int_0^1 C(ps1 + t(ps2 - ps1)) dt`, where `C(ps) = g(ps)^T H(ps) g(ps)` is the curvature at `ps`"""
    ts = jnp.linspace(0.0, 1.0, num=num)

    @jit
    def curvature(ps1: FrozenDict, ps2: FrozenDict) -> jnp.ndarray:
        difference = trees.add(ps1, ps2, beta=-trees.ONE)  # ps1 - ps2

        def integrand(t):
            convex_combo = trees.add(ps1, difference, beta=t)  # ps1 + t(ps2 - ps1)
            gs_value = grad(f)(convex_combo)
            _, hvp_value = hvp(f, convex_combo, gs_value)
            return trees.inner_product(gs_value, hvp_value)

        hvps = vmap(integrand)(ts)
        return jnp.trapz(hvps, ts)

    return curvature
