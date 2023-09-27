import numpy as np
import pickle

from jax import numpy as jnp, random, jit, jacfwd, vmap, value_and_grad
from scipy.integrate import solve_ivp
from optax import apply_updates, adam

from tqdm import tqdm

from pinn_jax.models import MLP
from pinn_jax import trees
from pinn_jax.utils import get_logs_id


def ic(x, omega=2.0*trees.ONE, offset=trees.ONE):
    return jnp.sin(omega * jnp.pi * x) + offset


def ic_np(x, omega=2.0, offset=1.0):
    return np.sin(omega * np.pi * x) + offset


def get_batch_jacobian(f):
    """
    - ``points`` should be of dimension ``n_batch x n_input``
    - returns tensor ``J`` of size ``n_batch x n_output x n_input``,
            where ``J[b, o, i] = partial_i f_o(x[b])``"""
    def batch_jacobian(params, points):
        return vmap(
            jacfwd(f, argnums=1),
            in_axes=[None, 0])(params, points)

    return batch_jacobian


def get_heat_residual(u_hat):
    batch_jacobian = get_batch_jacobian(u_hat)

    def heat_residual(params, points, matrix, offset):
        """
        - `points` is `(n_time, 1)`
        - `matrix` is `(n_space, n_space)`
        - `offset` is `(n_space, )`

        - `u` is `(n_time, n_space)`
        - `u_t` is `(n_time, n_space)`

        - `u[n_t, n_x]` is the predicted value
        """

        u = u_hat(params, points)
        u_t = batch_jacobian(params, points).squeeze()

        return ((u_t - u @ matrix - offset) ** 2).sum(axis=1).mean()

    return heat_residual


def get_loss_fn(u_hat):
    heat_residual = get_heat_residual(u_hat)
    t_0 = jnp.zeros((1, 1))

    def loss_fn(params, points, matrix, offset, initial_condition):
        residual = heat_residual(params, points, matrix, offset)

        u_0 = u_hat(params, t_0).squeeze()
        ic_error = ((u_0 - initial_condition) ** 2).sum()

        return residual + 1_000. * ic_error, (residual, ic_error)

    return loss_fn


def get_train_step(loss_fn, optimizer):
    @jit
    def train_step(params, points, matrix, offset, initial_condition, opt_state):
        grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, (residual, ic_error)), grads = grad_fn(params, points, matrix, offset, initial_condition)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = apply_updates(params, updates)
        return residual, ic_error, params, opt_state
    return train_step


def train(space_idx, layer_idx, hidden_idx, loss_info,
          u_true, A, phi,
          n_space, ts, xs, ts_eval,
          n_hidden, n_layer,
          n_steps):
    u_true_norm = np.linalg.norm(u_true)

    key = random.PRNGKey(1234)
    mlp_dim = n_layer * (n_hidden,) + (n_space,)
    mlp = MLP(mlp_dim)
    p = mlp.init(key, ts)
    print(f'network has {trees.count(p)} parameters')

    ic_vec = ic(xs)
    optimizer = adam(learning_rate=1e-4)
    opt_state = optimizer.init(p)

    loss_fn = get_loss_fn(mlp.apply)
    train_step = get_train_step(loss_fn, optimizer)

    @jit
    def abs_error(params, points):
        u_pred = mlp.apply(params, points)
        return jnp.linalg.norm(u_pred - u_true)

    for step in tqdm(range(n_steps)):
        residual, ic_error, p, opt_state = train_step(p, ts, A, phi, ic_vec, opt_state)

        loss_info[space_idx, layer_idx, hidden_idx, step, 0] = float(residual)
        loss_info[space_idx, layer_idx, hidden_idx, step, 1] = float(ic_error)
        loss_info[space_idx, layer_idx, hidden_idx, step, 2] = float(abs_error(p, ts_eval))

    loss_info[space_idx, layer_idx, hidden_idx, :, 3] = loss_info[space_idx, layer_idx, hidden_idx, :, 2] / u_true_norm


def get_ode_parameters(n_space, ts, ts_eval):
    xs_np = np.linspace(0.0, 1.0, num=n_space)
    xs = jnp.array(xs_np)

    A_np = (-2 * np.eye(n_space) + np.eye(n_space, k=1) + np.eye(n_space, k=-1)) * (n_space + 1) ** 2
    A = jnp.array(A_np)

    phi_np = np.zeros((n_space,))
    phi_np[0] = 1.0
    phi_np[-1] = 1.0
    phi_np = phi_np * (n_space + 1) ** 2
    phi = jnp.array(phi_np)

    soln = solve_ivp(lambda t, u: A_np @ u + phi_np, (0, ts[-1]), ic_np(xs_np), t_eval=ts_eval)
    return A, phi, xs, soln.y.T


def main():
    logs_id = get_logs_id()

    t_max = 0.1  # max time
    n_time = 1_024
    ts_np = np.linspace(0.0, t_max, num=n_time)
    ts_np_eval = (ts_np[1:] + ts_np[:-1]) / 2
    ts, ts_eval = jnp.array(ts_np)[:, None], jnp.array(ts_np_eval)[:, None]
    n_steps = 100_000

    n_spaces = [4, 8, 16, 32, 64, 128, 256, 512]
    n_layers = [1, 2, 4, 8]
    n_hiddens = [16, 64, 128]

    loss_info = np.zeros((len(n_spaces), len(n_layers), len(n_hiddens), n_steps, 4))

    for space_idx, n_space in enumerate(n_spaces):
        for layer_idx, n_layer in enumerate(n_layers):
            for hidden_idx, n_hidden in enumerate(n_hiddens):
                print(f's: {space_idx+1}/{len(n_spaces)}\tl: {layer_idx+1}/{len(n_layers)}\th: {hidden_idx+1}/{len(n_hiddens)}')
                A, phi, xs, u_true = get_ode_parameters(n_space, ts_np, ts_np_eval)
                train(space_idx, layer_idx, hidden_idx, loss_info,
                      u_true, A, phi,
                      n_space, ts, xs, ts_eval,
                      n_hidden, n_layer,
                      n_steps)

    results = {'loss_info': loss_info,
               'n_spaces': n_spaces,
               'n_layers': n_layers,
               'n_hiddens': n_hiddens,
               't_max': t_max,
               'n_time': n_time,
               'n_steps': n_steps}

    with open(f'results_{logs_id}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()