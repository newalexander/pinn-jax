from jax import jit, numpy as jnp
from jax.random import split, PRNGKey, normal
from jax.flatten_util import ravel_pytree
from flax.core import FrozenDict

from pinn_jax.derivatives import hvp
from pinn_jax.random_trees import random_normal, random_rademacher
from pinn_jax import trees

from typing import Callable, Tuple, List


def get_eigeniteration(f: Callable) -> Callable:
    """returns a function that performs one iteration of power iteration"""
    @jit
    def eigeniteration(params: FrozenDict, eigenvector, eigenvectors) -> Tuple[FrozenDict, jnp.ndarray]:
        eigenvector = trees.orthonormalize(eigenvector, eigenvectors)
        _, Hv = hvp(f, params, eigenvector)
        eigenvalue = trees.inner_product(eigenvector, Hv)
        Hv = trees.normalize(Hv)
        return Hv, eigenvalue

    return eigeniteration


def top_n_eigenanalysis(f: Callable, params: FrozenDict, top_n=1, max_iterations=100,
                        random_state=1234) -> Tuple[jnp.ndarray, List[FrozenDict]]:
    top_n, key = min(top_n, trees.count(params)), PRNGKey(random_state)
    eigenvalues, eigenvectors, eigeniteration = jnp.zeros((top_n,)), [], get_eigeniteration(f)

    for n in range(top_n):
        key, subkey = split(key)
        eigenvector = random_normal(subkey, params)
        eigenvalue = 0.0

        # TODO: will `lax.fori_loop` work here?
        for _ in range(max_iterations):
            eigenvector, eigenvalue = eigeniteration(params, eigenvector, eigenvectors)

        eigenvalues = eigenvalues.at[n].set(eigenvalue)
        eigenvectors.append(eigenvector)

    return eigenvalues, eigenvectors


def estimate_trace(f: Callable, primal: FrozenDict, n_iterations=1_000, dist=random_rademacher,
                   random_state=1234) -> jnp.ndarray:
    """
    use the skilling-hutchinson approach to estimate the trace of the hessian of ``f``, evaluated at ``primal``

    ``dist`` must have signature ``dist(key, ps: FrozenDict) -> FrozenDict``
    """
    trace = jnp.zeros(())
    key = PRNGKey(random_state)

    @jit
    def trace_update(_tangent):
        return trees.inner_product(_tangent, hvp(f, primal, _tangent)[1])

    # TODO: will `lax.fori_loop` work here?
    for _ in range(n_iterations):
        key, subkey = split(key)
        trace += trace_update(dist(subkey, primal))

    return trace / n_iterations


# adapted from: https://github.com/google/spectral-density/blob/master/jax/lanczos.py#L27
# TODO(gilmer) This function should use higher numerical precision?
def get_lanczos_alg(f: Callable, dim, order, random_state=1234) -> Callable:
    """Lanczos algorithm for tridiagonalizing a real symmetric matrix.

        This function applies Lanczos algorithm of a given order.  This function
        does full reorthogonalization to the iterates during training.

        Returns:
          tridiag: A tridiagonal matrix of size (order, order).
          vecs: A numpy array of size (order, dim) corresponding to the Lanczos
            vectors.
        """
    def _reorthogonalize(w, vecs, i):
        for j in range(i):
            tau = vecs[j, :].reshape((dim,))
            coeff = jnp.dot(w, tau)
            w += -coeff * tau
        beta = jnp.linalg.norm(w)
        return w, beta

    @jit
    def lanczos_alg(primal: FrozenDict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        key, subkey = split(PRNGKey(random_state))
        tridiag = jnp.zeros((order, order))
        vecs = jnp.zeros((order, dim))

        # initial lanczos iterate
        init_vec = normal(subkey, shape=(dim,))
        init_vec = init_vec / jnp.linalg.norm(init_vec)
        vecs = vecs.at[0].set(init_vec)

        _, unravel = ravel_pytree(primal)

        def _hvp(_tangent):
            _, Hv = hvp(f, primal, unravel(_tangent))
            return ravel_pytree(Hv)[0]

        beta = jnp.zeros(())
        # TODO(gilmer): Better to use lax.fori loop for faster compile?
        for i in range(order):
            v = vecs[i, :].reshape((dim,))
            if i == 0:
                v_old = 0
            else:
                v_old = vecs[i-1, :].reshape((dim,))

            w = _hvp(v) - beta * v_old

            alpha = jnp.dot(w, v)
            tridiag = tridiag.at[(i, i)].set(alpha)
            w = w - alpha * v

            w, beta = _reorthogonalize(w, vecs, i)

            if i + 1 < order:
                tridiag = tridiag.at[(i, i + 1)].set(beta)
                tridiag = tridiag.at[(i + 1, i)].set(beta)
                vecs = vecs.at[i + 1].set(w / beta)

        return tridiag, vecs

    return lanczos_alg
