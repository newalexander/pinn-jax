# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp, vmap, jacfwd, jacrev, jvp, grad
from jax.flatten_util import ravel_pytree

from flax.core import FrozenDict

from typing import Callable, Tuple
from jaxtyping import Float, Bool, Array


def get_batch_jacobian(func: Callable, argnums=1, in_axes=(None, 0)) -> Callable:
    """
    calculate the batch-wise jacobian of a function with respect to its input argument (not its weights)

    - ``func`` implements a neural network ``f: R^n_input -> R^n_output``
      - by default, ``func`` has signature ``f(params, points) = prediction``
    - ``points`` should be of dimension ``n_batch x n_input``
    - returns tensor ``J`` of size ``n_batch x n_output x n_input``,
            where ``J[b, o, i] = partial_i f_o(x[b])``"""
    jacobian = jacfwd(func, argnums=argnums)

    def batch_jacobian(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        return vmap(jacobian, in_axes=in_axes)(params, points)

    return batch_jacobian


def get_batch_hessian(func: Callable, argnums=1, in_axes=(None, 0)) -> Callable:
    """calculate the batch-wise hessian of a function with respect to its input argument (not its weights)

    - ``func`` implements a neural network ``f: R^n_input -> R^n_output``
          - has signature ``f(params, points) = prediction``
    - ``points`` should be of dimension ``n_batch x n_input``
    - returns tensor ``H`` of size ``n_batch x n_output x n_input x n_input``,
                where ``H[b, o, i1, i2] = partial_i1 partial_i2 f_o(x[b])``
    - this uses forward-over-forward automatic differentiation, which has a decent chance of being most efficient
       when ``n_input \approx n_output`` and both are fairly small
    """
    hessian = jacfwd(jacfwd(func, argnums=argnums), argnums=argnums)

    def batch_hessian(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        return vmap(hessian, in_axes=in_axes)(params, points)

    return batch_hessian


def get_batch_jerk(func: Callable, argnums=1, in_axes=(None, 0)) -> Callable:
    """calculate the batch-wise "jerk" (array of third-order derivatives) of a function with respect to its input
    argument (not its weights)

    - ``func`` implements a neural network ``f: R^n_input -> R^n_output``
        - has signature ``f(params, points) = prediction``
    - ``points`` should be of dimension ``n_batch x n_input``
    - returns tensor ``J`` of size ``n_batch x n_output x n_input x n_input x n_input``,
        where ``H[b, o, i1, i2, i3] = partial_i1 partial_i2 partial_i3 f_o(x[b])``
    - this uses forward-over-forward-over-forward automatic differentiation, which has a decent chance of being most
        efficient when ``n_input \approx n_output`` and both are fairly small
    """
    jerk = jacfwd(jacfwd(jacfwd(func, argnums=argnums), argnums=argnums), argnums=argnums)

    def batch_jerk(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        return vmap(jerk, in_axes=in_axes)(params, points)

    return batch_jerk


def get_batch_snap(func: Callable, argnums=1, in_axes=(None, 0)) -> Callable:
    """calculate the batch-wise "snap" (array of fourth-order derivatives) of a function with respect to its input
        argument (not its weights)

    - ``func`` implements a neural network ``f: R^n_input -> R^n_output``
        - has signature ``f(params, points) = prediction``
    - ``points`` should be of dimension ``n_batch x n_input``
    - returns tensor ``S`` of size ``n_batch x n_output x n_input x n_input x n_input x n_input``,
        where ``H[b, o, i1, i2, i3, i4] = partial_i1 partial_i2 partial_i3 partial_i4 f_o(x[b])``
    - this uses forward-over-forward-over-forward automatic differentiation, which has a decent chance of being most
        efficient when ``n_input \approx n_output`` and both are fairly small
    """
    snap = jacfwd(jacfwd(jacfwd(jacfwd(func, argnums=argnums), argnums=argnums), argnums=argnums), argnums=argnums)

    def batch_snap(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        return vmap(snap, in_axes=in_axes)(params, points)

    return batch_snap


def get_hvp(func: Callable, argnums=0, mode='ode') -> Callable:
    """
    given a function ``f: R^n -> R``, a vector ``p in R^n`` at which we linearize ``f`` (the `primal`), and
    vector ``t in R^n`` (the `tangent`), we calculate the output of the linear map

        (p, t) -> (H(f)(p)) t

    where ``H(f)(p)`` is the ``n x n`` hessian matrix of ``f`` evaluated at ``p``. this function never calculates
    the full hessian, which makes it more scalable for high-dimensional problems.

    if ``f`` involves a neural network, and ``primal`` is the network's parameters, then ``primal`` and `tangent` should
    be ``FrozenDict``s with the same set of keys and value sizes.

    returns a tuple of ``FrozenDict``s ``(grad(f)(primal), hvp(f, primal)(tangent))``
    """
    # TODO: can we clean this up so we don't have a bunch of different function interfaces?
    f_grad = grad(func, argnums=argnums)

    def hvp_ode(primal: FrozenDict, tangent: FrozenDict, points: jnp.ndarray) -> Tuple:
        """defined for a function ``func(params, points)``"""
        return jvp(lambda _primal: f_grad(_primal, points), (primal,), (tangent,))

    def hvp_fixed(primal: FrozenDict, tangent: FrozenDict) -> Tuple:
        """defined for a function ``func(params)``"""
        return jvp(f_grad(primal), (primal,), (tangent,))

    def hvp_pde(primal: FrozenDict, tangent: FrozenDict,
                interior_points: Float[Array, "n_interior n_inputs"],
                boundary_points: Float[Array, "n_boundary n_inputs"], boundary_idxs: Bool[Array, "n_boundary n_bcs"],
                initial_points: Float[Array, "n_initial n_inputs"]):
        """defined for a function ``func(params, interior_points, boundary_points, boundary_idxs, initial_points)``"""
        return jvp(lambda _primal: f_grad(_primal, interior_points, boundary_points, boundary_idxs, initial_points),
                   (primal, ), (tangent,))

    if mode == 'ode':
        return hvp_ode
    elif mode == 'fixed':
        return hvp_fixed
    elif mode == 'pde':
        return hvp_pde
    else:
        raise ValueError("invalid function mode specified!!!")


def get_flat_hessian(func: Callable, argnums=0, mode='ode') -> Callable:
    """calculate the hessian of ``f`` evaluated at ``primal``. returned as a full matrix, which requires flattening
        ``primal`` into a 1d vector."""
    def hessian(_func: Callable) -> Callable:
        return jacfwd(jacrev(_func, argnums=argnums), argnums=argnums)

    def hessian_ode(primal: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        flat_primal, unravel = ravel_pytree(primal)

        def _f(_flat_primal, _points):
            return func(unravel(_flat_primal), points)

        return hessian(_f)(flat_primal, points)

    if mode == 'ode':
        return hessian_ode
    else:
        raise ValueError('invalid function mode specified!!!')


def flat_hessian(func: Callable, primal: FrozenDict) -> jnp.ndarray:
    """calculate the hessian of ``f`` evaluated at ``primal``. returned as a full matrix, which requires flattening
    ``primal`` into a 1d vector."""
    flat_primal, unravel = ravel_pytree(primal)

    def _f(_flat_primal): return func(unravel(_flat_primal))

    def hessian(_func): return jacfwd(jacrev(_func))

    return hessian(_f)(flat_primal)


def test():
    from jax import jit
    from jax.random import split, PRNGKey, normal
    from pinn_jax.models import MLP

    n_batch, n_input, n_output, n_hidden = 17, 5, 3, 31

    key_x, key_p = split(PRNGKey(1234))

    def mlp():
        return MLP((n_hidden, n_hidden, n_output))

    def loss(_p, _x):
        return (mlp().apply(_p, _x) ** 2).mean()

    x = normal(key_x, (n_batch, n_input))
    p = mlp().init(key_p, x)

    mlp().apply(p, x)

    hessian = jit(get_flat_hessian(loss, mode='ode'))
    print(hessian(p, x).shape)


if __name__ == '__main__':
    test()
