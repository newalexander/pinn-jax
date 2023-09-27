# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp, vmap
from flax import linen as nn
from pinn_jax import trees
from pinn_jax.layers import CosineLayer
from typing import Sequence, Callable, Union


class MLP(nn.Module):
    """standard multilayer perceptron"""
    n_features: Sequence[int]  # feature size of each layer
    activation: Callable = nn.tanh
    layers = None

    # since `nn.Module`s are python 3.7 `Dataclass`es, use `setup`  rather than `__init__`
    # see: https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module
    def setup(self):
        self.layers = [nn.Dense(n_feat) for n_feat in self.n_features]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x


class MLPSet(nn.Module):
    """set of MLPs wrapped into a single `nn.Module` to make joint training easier"""
    n_mlps: int
    n_features: Union[Sequence[int], Sequence[Sequence[int]]]
    activation: Callable = nn.tanh
    mlps = None

    def setup(self):
        # if `n_features` is a just a sequence of ints, use same architecture for each MLP
        if isinstance(self.n_features[0], int):
            self.mlps = [MLP(self.n_features, self.activation) for _ in range(self.n_mlps)]
        # otherwise use individually specified architectures
        else:
            assert len(self.n_features) == self.n_mlps, "number of MLPs must match number of specified architectures!"
            self.mlps = [MLP(n_features, self.activation) for n_features in self.n_features]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jnp.hstack([mlp(inputs) for mlp in self.mlps])


class PeriodicMLP(nn.Module):
    """MLP with periodicity over a regular domain."""
    n_features: Sequence[int]  # feature size of each normal layer
    n_periodic_features: Sequence[int]  # feature sizes of each spatial dimension's encoder
    frequencies: Sequence[float]  # periodicities of each spatial dimension's encoder
    n_input: int  # input size
    activation: Callable = nn.tanh
    periodic_encoders = None
    periodic_layers = None
    layers = None

    def setup(self):
        assert len(self.n_periodic_features) == len(self.frequencies),\
            "`n_periodic_features` and `frequencies` must be the same length!!!"
        self.periodic_encoders = [CosineLayer(n_periodic_features, frequency) for
                                  (n_periodic_features, frequency) in zip(self.n_periodic_features, self.frequencies)]
        self.periodic_layers = [nn.Dense(self.n_features[0]) for _ in range(len(self.periodic_encoders))]
        self.layers = [nn.Dense(n_feat) for n_feat in self.n_features[1:]]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        for an input (x_1, ..., x_D, t), calculate:
        - w_d = sigma(CosineLayer(x_d)) for all x_d    (w_d in R^n_periodic_features[d])
        - v = sigma(Dense(w_1) + ... + Dense(w_D))     (v in R^n_features[0])
        - z = [v_1, ..., v_P, t]
        - return MLP(z)

        guarantees 1d C^infty periodicity: u(x) = u(x + L), where L = 2 pi / omega

        introduced in: https://doi.org/10.1016/j.jcp.2021.110242
        """
        def encode_inputs(_inputs: jnp.ndarray) -> jnp.ndarray:
            """encode spatial inputs to obey C^infty periodicity"""
            x, t = _inputs[:-1], _inputs[-1]

            ws = [_layer(self.activation(encoder(x[idx])))
                  for idx, (encoder, _layer) in enumerate(zip(self.periodic_encoders, self.periodic_layers))]
            v = jnp.hstack([self.activation(sum(ws)), [t]])
            return v

        z = vmap(encode_inputs)(inputs.reshape((-1, self.n_input)))
        # `inputs` needs to be a rank-2 array for `vmap` to work properly

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i != len(self.layers) - 1:
                z = self.activation(z)
        return z


class PeriodicMLPxt(nn.Module):
    """MLP with periodicity over a 1d regular domain."""
    n_features: Sequence[int]  # feature size of each normal layer
    n_periodic_features: int  # feature sizes of each spatial dimension's encoder
    frequency: float  # periodicity of each spatial dimension's encoder
    activation: Callable = nn.tanh
    layers = None
    ks = None

    def setup(self):
        self.ks = jnp.arange(1, self.n_periodic_features + 1)
        self.layers = [nn.Dense(n_feat) for n_feat in self.n_features[1:]]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        for an input (x, t), calculate:
        - v = concat([1, t], [cos(k w x): k = 1,...,m], [sin(k w x): k = 1,...,m]) in R^(2 * n_periodic_features + 2)
        - return MLP(v)

        guarantees 1d C^infty periodicity: u(x, t) = u(x + L, t), where L = 2 pi / w

        introduced in: https://arxiv.org/abs/2203.07404
        """
        def encode_inputs(_inputs: jnp.ndarray) -> jnp.ndarray:
            """encode spatial inputs to obey C^infty periodicity"""
            x, t = _inputs[0], _inputs[1]
            v = jnp.hstack([1., t, jnp.cos(self.ks * self.frequency * x), jnp.sin(self.ks * self.frequency * x)])
            return v

        z = vmap(encode_inputs)(inputs.reshape((-1, 2)))
        # `inputs` needs to be a rank-2 array for `vmap` to work properly

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i != len(self.layers) - 1:
                z = self.activation(z)
        return z


class PeriodicMLPxyt(nn.Module):
    """MLP with periodicity over a 1d regular domain."""
    n_features: Sequence[int]  # feature size of each normal layer
    n_periodic_features: Sequence[int]  # feature sizes of each spatial dimension's encoder
    frequency: Sequence[float]  # periodicity of each spatial dimension's encoder
    activation: Callable = nn.tanh
    layers = None
    ks_x, ks_xx = None, None
    ks_y, ks_yy = None, None

    def setup(self):
        self.ks_x = jnp.arange(1, self.n_periodic_features[0] + 1)
        self.ks_y = jnp.arange(1, self.n_periodic_features[1] + 1)
        self.ks_xx, self.ks_yy = jnp.meshgrid(self.ks_x, self.ks_y)
        self.ks_xx, self.ks_yy = self.ks_xx.flatten(), self.ks_yy.flatten()
        self.layers = [nn.Dense(n_feat) for n_feat in self.n_features]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        for an input (x, t), calculate:
        - v = concat([1, t], [cos(k w x): k = 1,...,m], [sin(k w x): k = 1,...,m]) in R^(2 * n_periodic_features + 2)
        - return MLP(v)

        guarantees 2d C^infty periodicity: u(x, t) = u(x + L, t), where L = 2 pi / w

        introduced in: https://arxiv.org/abs/2203.07404
        note that the causalpinn repo also implements a polynomial kernel featurization of time
        """
        def encode_inputs(_inputs: jnp.ndarray) -> jnp.ndarray:
            """encode spatial inputs to obey C^infty periodicity"""
            x, y, t = _inputs[0], _inputs[1], _inputs[2]
            v = jnp.hstack([1., t,
                            jnp.cos(self.ks_x * self.frequency[0] * x), jnp.cos(self.ks_y * self.frequency[1] * y),
                            jnp.sin(self.ks_x * self.frequency[0] * x), jnp.sin(self.ks_y * self.frequency[1] * y),
                            jnp.cos(self.ks_xx * self.frequency[0] * x) * jnp.cos(self.ks_yy * self.frequency[1] * y),
                            jnp.cos(self.ks_xx * self.frequency[0] * x) * jnp.sin(self.ks_yy * self.frequency[1] * y),
                            jnp.sin(self.ks_xx * self.frequency[0] * x) * jnp.cos(self.ks_yy * self.frequency[1] * y),
                            jnp.sin(self.ks_xx * self.frequency[0] * x) * jnp.sin(self.ks_yy * self.frequency[1] * y)])
            return v

        z = vmap(encode_inputs)(inputs.reshape((-1, 3)))
        # `inputs` needs to be a rank-2 array for `vmap` to work properly

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i != len(self.layers) - 1:
                z = self.activation(z)
        return z


class EncoderDecoderMLP(nn.Module):
    """
    Encoder-decoder MLP architecture introduced in https://doi.org/10.1137/20M1318043
    """
    n_hidden: int  # feature size of each layer
    n_output: int
    n_layers: int
    activation: Callable = nn.tanh
    u_enc, v_enc, layers, decoder = None, None, None, None

    def setup(self):
        self.u_enc, self.v_enc = nn.Dense(self.n_hidden), nn.Dense(self.n_hidden)
        self.layers = [nn.Dense(self.n_hidden) for _ in range(self.n_layers)]
        self.decoder = nn.Dense(self.n_output)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        U = sigma(X W^U + b^U)    V = sigma(X W^V + b^V)
        H^1 = sigma(H^0 W^0 + b^0)

        for k = 1, ..., L-1
            Z^k = sigma(H^k W^k + b^k)
            H^{k+1} = (1 - Z^k) odot U + Z^k odot V

        Y = H^{L+1} W + b
        """
        u, v = self.activation(self.u_enc(inputs)), self.activation(self.v_enc(inputs))
        h = self.activation(self.layers[0](inputs))

        for layer in self.layers[1:]:
            z = self.activation(layer(h))
            h = (trees.ONE - z) * u + z * v
        return self.decoder(h)


class PeriodicEncoderDecoderMLPxt(nn.Module):
    """
    Encoder-decoder MLP architecture introduced in https://doi.org/10.1137/20M1318043
    """
    n_hidden: int  # feature size of each layer
    n_output: int
    n_layers: int
    u_enc, v_enc, layers, decoder = None, None, None, None
    ks = None

    n_features: Sequence[int]  # feature size of each normal layer
    n_periodic_features: int  # feature sizes of each spatial dimension's encoder
    frequency: float  # periodicity of each spatial dimension's encoder
    activation: Callable = nn.tanh

    def setup(self):
        self.u_enc, self.v_enc = nn.Dense(self.n_hidden), nn.Dense(self.n_hidden)
        # TODO: which 'layers'
        self.layers = [nn.Dense(self.n_hidden) for _ in range(self.n_layers)]  # Keep layers from EncoderDecoderMLP
        self.decoder = nn.Dense(self.n_output)

        self.ks = jnp.arange(1, self.n_periodic_features + 1)
        # TODO: which 'layers'
        # self.layers = [nn.Dense(n_feat) for n_feat in self.n_features[1:]]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        U = sigma(X W^U + b^U)    V = sigma(X W^V + b^V)
        H^1 = sigma(H^0 W^0 + b^0)

        for k = 1, ..., L-1
            Z^k = sigma(H^k W^k + b^k)
            H^{k+1} = (1 - Z^k) odot U + Z^k odot V

        Y = H^{L+1} W + b
        """
        def encode_inputs(_inputs: jnp.ndarray) -> jnp.ndarray:
            """encode spatial inputs to obey C^infty periodicity"""
            x, t = _inputs[0], _inputs[1]
            v = jnp.hstack([1., t, jnp.cos(self.ks * self.frequency * x), jnp.sin(self.ks * self.frequency * x)])
            return v

        # inputs = encode_inputs(inputs)
        z = vmap(encode_inputs)(inputs.reshape((-1, 2)))
        u, v = self.activation(self.u_enc(z)), self.activation(self.v_enc(z))
        h = self.activation(self.layers[0](z))

        for layer in self.layers[1:]:
            z = self.activation(layer(h))
            h = (trees.ONE - z) * u + z * v
        return self.decoder(h)


class ResNet(nn.Module):
    """
    feedforward network with layerwise skip connections
    """
    n_hidden: int  # feature size of each layer
    n_output: int
    n_layers: int
    activation: Callable = nn.tanh
    encoder, decoder, layers = None, None, None

    def setup(self):
        self.encoder = nn.Dense(self.n_hidden)
        self.decoder = nn.Dense(self.n_output)
        self.layers = [nn.Dense(self.n_hidden) for _ in range(self.n_layers-1)]

    def __call__(self, inputs):
        """defines a map ``x -> h^1 -> ... -> h^(n_layers-1) -> y``,

            where h^i in R^n_hidden, y in R^n_output, and h^{i+1} = MLP(h^i) + h^i
        """
        x = self.encoder(inputs)
        for layer in self.layers:
            x = self.activation(layer(x)) + x
        return self.decoder(x)


class PeriodicResNet(nn.Module):
    """ResNet with periodicity over a regular domain."""
    n_features: Sequence[int]  # feature size of each normal layer
    n_periodic_features: Sequence[int]  # feature sizes of each spatial dimension's encoder
    frequencies: Sequence[float]  # periodicities of each spatial dimension's encoder
    n_input: int  # input size
    activation: Callable = nn.tanh
    periodic_encoders = None
    periodic_layers = None
    layers = None

    def setup(self):
        assert len(self.n_periodic_features) == len(self.frequencies),\
            "`n_periodic_features` and `frequencies` must be the same length!!!"
        self.periodic_encoders = [CosineLayer(n_periodic_features, frequency) for
                                  (n_periodic_features, frequency) in zip(self.n_periodic_features, self.frequencies)]
        self.periodic_layers = [nn.Dense(self.n_features[0]) for _ in range(len(self.periodic_encoders))]
        self.layers = [nn.Dense(n_feat) for n_feat in self.n_features[1:]]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        for an input (x_1, ..., x_D, t), calculate:
        - w_d = sigma(CosineLayer(x_d)) for all x_d    (w_d in R^n_periodic_features[d])
        - v = sigma(Dense(w_1) + ... + Dense(w_D))     (v in R^n_features[0])
        - z = [v_1, ..., v_P, t]
        - return MLP(z)

        guarantees 1d C^infty periodicity: u(x) = u(x + L), where L = 2 pi / omega

        introduced in: https://doi.org/10.1016/j.jcp.2021.110242
        used in: https://arxiv.org/abs/2203.07404
        """
        def encode_inputs(_inputs: jnp.ndarray) -> jnp.ndarray:
            """encode spatial inputs to obey C^infty periodicity"""
            x, t = _inputs[:-1], _inputs[-1]

            ws = [_layer(self.activation(encoder(x[idx])))
                  for idx, (encoder, _layer) in enumerate(zip(self.periodic_encoders, self.periodic_layers))]
            v = jnp.hstack([self.activation(sum(ws)), [t]])
            return v

        z = vmap(encode_inputs)(inputs.reshape((-1, self.n_input)))
        # `inputs` needs to be a rank-2 array for `vmap` to work properly

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i != len(self.layers) - 1:
                z = self.activation(z) + z
        return z


class DeepONet(nn.Module):
    """DeepONet https://arxiv.org/abs/1910.03193

        f(x, v) = branch(x) * trunk(v)
    """
    n_features_branch: Sequence[int]  # dimension of input features
    n_features_trunk: Sequence[int]  # dimension of space features
    activation: Callable = nn.tanh
    layers_branch = None
    layers_trunk = None
    delimiting_index: int = 2

    def setup(self):
        self.layers_branch = [nn.Dense(n_feat) for n_feat in self.n_features_branch]
        self.layers_trunk = [nn.Dense(n_feat) for n_feat in self.n_features_trunk]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:

        if inputs.ndim < 2:
            inputs = inputs.reshape(1,-1)

        x = inputs[:, self.delimiting_index:]
        # branch
        for i, layer in enumerate(self.layers_branch):
            x = layer(x)
            if i != len(self.layers_branch) - 1:
                x = self.activation(x)

        v = inputs[:, :self.delimiting_index]
        # trunk
        for i, layer in enumerate(self.layers_trunk):
            v = layer(v)
            if i != len(self.layers_trunk) - 1:
                v = self.activation(v)

        return v * x


class DeepOCNet(nn.Module):
    """ DeepONet with an additional combination module
        https://arxiv.org/abs/1910.03193

        f(x, v) = combine(branch(x) * trunk(v))
    """
    n_features_branch: Sequence[int]  # dimention of input features
    n_features_trunk: Sequence[int]  # dimension of space features
    n_features_combine: Sequence[int]  # dimension of combination layer
    activation: Callable = nn.tanh
    layers_branch = None
    layers_trunk = None
    layers_combine = None
    delimiting_index: int = 2

    def setup(self):
        self.layers_branch = [nn.Dense(n_feat) for n_feat in self.n_features_branch]
        self.layers_trunk = [nn.Dense(n_feat) for n_feat in self.n_features_trunk]
        self.layers_combine = [nn.Dense(n_feat) for n_feat in self.n_features_combine]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:

        if inputs.ndim < 2:
            inputs = inputs.reshape(1,-1)

        x = inputs[:, self.delimiting_index:]
        # branch
        for i, layer in enumerate(self.layers_branch):
            x = layer(x)
            if i != len(self.layers_branch) - 1:
                x = self.activation(x)

        v = inputs[:, :self.delimiting_index]
        # trunk
        for i, layer in enumerate(self.layers_trunk):
            v = layer(v)
            if i != len(self.layers_trunk) - 1:
                v = self.activation(v)

        z = v * x
        # combine
        for i, layer in enumerate(self.layers_combine):
            z = layer(z)
            if i != len(self.layers_combine) - 1:
                z = self.activation(z)

        return z


class DeepOCNetSet(nn.Module):
    """ DeepONet with an additional combination module
        https://arxiv.org/abs/1910.03193

        f(x, v) = combine(branch(x) * trunk(v))
    """
    n_features_branch: Sequence[int]  # dimention of input features
    n_features_trunk: Sequence[int]  # dimension of space features
    n_features_combine: Sequence[int]  # dimension of combination layer
    activation: Callable = nn.tanh
    n_output: int = 1
    layers_branch = None
    layers_trunk = None
    layers_combine = None
    delimiting_index: int = 2

    def setup(self):
        self.layers_branch = [nn.Dense(n_feat) for n_feat in self.n_features_branch]
        self.layers_trunk = [nn.Dense(n_feat) for n_feat in self.n_features_trunk]
        self.layers_combine =[[nn.Dense(n_feat) for n_feat in self.n_features_combine] for _ in range(self.n_output)]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:

        if inputs.ndim < 2:
            inputs = inputs.reshape(1,-1)

        x = inputs[:, self.delimiting_index:]
        # branch
        for i, layer in enumerate(self.layers_branch):
            x = layer(x)
            if i != len(self.layers_branch) - 1:
                x = self.activation(x)

        v = inputs[:, :self.delimiting_index]
        # trunk
        for i, layer in enumerate(self.layers_trunk):
            v = layer(v)
            if i != len(self.layers_trunk) - 1:
                v = self.activation(v)

        z = v * x
        # combine
        return jnp.hstack([self.compute_output(z, out_layers) for out_layers in self.layers_combine])

    def compute_output(self, z, layers):
        for i, layer in enumerate(layers):
            z = layer(z)
            if i != len(self.layers_combine) - 1:
                z = self.activation(z)

        return z


def get_activation(activation: str, activation_scaling: float) -> Callable:
    """construct an activation function of the form y = f(alpha x).
    there are a lot more of these we can add as needed/desired (e.g., leaky relu, sigmoid, etc)
    """
    if activation == 'tanh':
        def activation_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.tanh(activation_scaling * x)
    elif activation == 'sin':
        def activation_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sin(activation_scaling * x)
    elif activation == 'relu':
        def activation_fn(x: jnp.ndarray) -> jnp.ndarray:
            return nn.relu(activation_scaling * x)
    elif activation == 'swish':
        def activation_fn(x: jnp.ndarray) -> jnp.ndarray:
            return nn.swish(activation_scaling * x)
    else:
        raise ValueError('unsupported `activation` function specified!!!')

    return activation_fn


def test_periodic_mlp_xt():
    from jax import random, jit, config as jax_config
    from pinn_jax.derivatives import get_batch_jacobian, get_batch_hessian

    if True:
        jax_config.update('jax_enable_x64', True)

    key_x, key_p = random.split(random.PRNGKey(5678))

    n_input = 2
    n_features = (7, 11)
    n_periodic_features = 3
    frequency = jnp.pi
    lengths = jnp.array([2. * jnp.pi / frequency] + [0.])

    def mlp(): return PeriodicMLPxt(n_features, n_periodic_features, frequency)

    x = random.normal(key_x, (13, n_input))
    x_perturb = x + lengths
    p = mlp().init(key_p, x)
    # print(x.shape, mlp().apply(p, x).shape)

    print(jit(mlp().apply)(p, x).shape, jit(mlp().apply)(p, x_perturb).shape)
    print(jnp.linalg.norm(jit(mlp().apply)(p, x) - mlp().apply(p, x_perturb)))

    print('now the hard part')
    batch_jacobian = get_batch_jacobian(mlp().apply)
    print(jit(batch_jacobian)(p, x).shape)  # n_batch x n_output x n_input
    print(jnp.linalg.norm(jit(batch_jacobian)(p, x) - jit(batch_jacobian)(p, x_perturb)))

    print('now the harder part')
    batch_hessian = get_batch_hessian(mlp().apply)
    print(jit(batch_hessian)(p, x).shape)
    print(jnp.linalg.norm(jit(batch_jacobian)(p, x) - jit(batch_jacobian)(p, x_perturb)))


def test_periodic_mlp_xyt():
    from jax import random, jit, config as jax_config
    from pinn_jax.derivatives import get_batch_jacobian, get_batch_hessian

    if True:
        jax_config.update('jax_enable_x64', True)

    key_x, key_p = random.split(random.PRNGKey(5678))

    n_input = 3
    n_features = (7, 11)
    n_periodic_features = (3, 4)
    frequency = (jnp.pi, jnp.pi)
    lengths = jnp.array([2. * jnp.pi / frequency[0], 2. * jnp.pi / frequency[1]] + [0.])

    def mlp(): return PeriodicMLPxyt(n_features, n_periodic_features, frequency)

    x = random.normal(key_x, (13, n_input))
    x_perturb = x + lengths
    p = mlp().init(key_p, x)
    # print(x.shape, mlp().apply(p, x).shape)

    print(jit(mlp().apply)(p, x).shape, jit(mlp().apply)(p, x_perturb).shape)
    print(jnp.linalg.norm(jit(mlp().apply)(p, x) - mlp().apply(p, x_perturb)))

    print('now the hard part')
    batch_jacobian = get_batch_jacobian(mlp().apply)
    print(jit(batch_jacobian)(p, x).shape)  # n_batch x n_output x n_input
    print(jnp.linalg.norm(jit(batch_jacobian)(p, x) - jit(batch_jacobian)(p, x_perturb)))

    print('now the harder part')
    batch_hessian = get_batch_hessian(mlp().apply)
    print(jit(batch_hessian)(p, x).shape)
    print(jnp.linalg.norm(jit(batch_jacobian)(p, x) - jit(batch_jacobian)(p, x_perturb)))


def test_periodic_mlp():
    from jax import random, jit, config as jax_config
    from pinn_jax.derivatives import get_batch_jacobian, get_batch_hessian

    if True:
        jax_config.update('jax_enable_x64', True)

    key_x, key_p = random.split(random.PRNGKey(5678))

    n_input = 2#3
    n_features = (7, 11)
    n_periodic_features = (3, )#5)
    frequencies = (2. * jnp.pi, )#jnp.pi)
    lengths = jnp.array([2. * jnp.pi / freq for freq in frequencies] + [0.])

    def mlp(): return PeriodicMLP(n_features, n_periodic_features, frequencies, n_input)

    x = random.normal(key_x, (13, n_input))
    x_perturb = x + lengths
    p = mlp().init(key_p, x)
    # print(x.shape, mlp().apply(p, x).shape)

    print(jit(mlp().apply)(p, x).shape, jit(mlp().apply)(p, x_perturb).shape)
    print(jnp.linalg.norm(jit(mlp().apply)(p, x) - mlp().apply(p, x_perturb)))

    print('now the hard part')
    batch_jacobian = get_batch_jacobian(mlp().apply)
    print(jit(batch_jacobian)(p, x).shape)  # n_batch x n_output x n_input
    print(jnp.linalg.norm(jit(batch_jacobian)(p, x) - jit(batch_jacobian)(p, x_perturb)))

    print('now the harder part')
    batch_hessian = get_batch_hessian(mlp().apply)
    print(jit(batch_hessian)(p, x).shape)
    print(jnp.linalg.norm(jit(batch_jacobian)(p, x) - jit(batch_jacobian)(p, x_perturb)))


if __name__ == '__main__':
    test_periodic_mlp_xyt()
