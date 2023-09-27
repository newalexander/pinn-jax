# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

# preamble adapted from: https://flax.readthedocs.io/en/latest/_modules/flax/linen/linear.html#Dense

from jax import numpy as jnp, lax
from flax import linen as nn
from flax.linen.initializers import lecun_normal, zeros
from flax.linen.dtypes import promote_dtype

from typing import Optional, Callable, Any, Tuple, Union

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],Tuple[lax.Precision, lax.Precision]]
default_kernel_init = lecun_normal()


class CosineLayer(nn.Module):
    """cosine-featurization layer"""
    features: int
    frequency: float
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        returns u(x) = a * cos(omega x + phi)

        maps an (n_points x n_inputs) array into (n_points x n_features) via elementwise multiplication

        guarantees 1d C^infty periodicity: u(x) = u(x + L), where L = 2 pi / omega

        introduced in: https://doi.org/10.1016/j.jcp.2021.110242
        """
        kernel = self.param('kernel', self.kernel_init, (self.features, 1), self.param_dtype).squeeze(-1)
        bias = self.param('bias', self.bias_init, (self.features, ), self.param_dtype)

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        return kernel.T * jnp.cos(self.frequency * inputs + bias)


def test():
    from jax import random

    key_x, key_p = random.split(random.PRNGKey(1234))
    omega = 1.5 * jnp.pi
    length = 2. * jnp.pi / omega
    n_periodic_features = 3

    def m(): return CosineLayer(n_periodic_features, omega)

    x = random.normal(key_x, (5, 1))

    p = m().init(key_p, x)
    kernel = p['params']['kernel']
    bias = p['params']['bias']

    for idx in range(n_periodic_features):
        manual = kernel[idx] * jnp.cos(omega * x + bias[idx]).squeeze()
        mlp = m().apply(p, x)[:, idx]
        # print('manual:', manual.shape, 'mlp:', mlp.shape)
        print(jnp.linalg.norm(manual - mlp))

    print(jnp.linalg.norm(m().apply(p, x) - m().apply(p, x + length)))

    print(x.shape, m().apply(p, x).shape)


if __name__ == '__main__':
    test()
