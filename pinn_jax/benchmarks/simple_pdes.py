# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import numpy as np

from typing import Callable, Tuple


def reaction(u, rho, dt):
    """
    u_t = rho * u * (1 - u)
    adapted from: https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/systems_pbc.py#L47
    """
    factor_1 = u * np.exp(rho * dt)
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u


def diffusion(u, nu, dt, IKX2):
    """
    u_t = nu * u_xx
    https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/systems_pbc.py#L55
    """
    factor = np.exp(nu * IKX2 * dt)
    u_hat = np.fft.fft(u)
    u_hat *= factor
    u = np.real(np.fft.ifft(u_hat))
    return u


def convection_diffusion(u0: Callable, nu: float, beta: float, source: float = 0, n_x: int = 256, n_t: int = 100
                         ) -> np.ndarray:
    """Calculate the u solution for convection/diffusion (assuming periodic boundary conditions).
    Args:
        u0: Initial condition function
        nu: viscosity coefficient
        beta: wavespeed coefficient
        source: q (forcing term), option to have this be a constant
        n_x: size of the x grid
        n_t: size of the t grid
    Returns:
        u: solution

    adapted from here: https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/systems_pbc.py
    """

    h = 2 * np.pi / n_x
    x = np.arange(0, 2 * np.pi, h)  # not inclusive of the last point
    t = np.linspace(0, 1, n_t).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    u0 = u0(x).squeeze()
    G = (np.copy(u0)*0) + source  # G is the same size as u0

    IKX_pos = 1j * np.arange(0, n_x/2+1, 1)
    IKX_neg = 1j * np.arange(-n_x/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    uhat0 = np.fft.fft(u0)
    nu_factor = np.exp(nu * IKX2 * T - beta * IKX * T)
    A = uhat0 - np.fft.fft(G)*0  # at t=0, second term goes away
    uhat = A*nu_factor + np.fft.fft(G)*T  # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat)).flatten()

    return u


def reaction_diffusion(u0: Callable, nu: float, rho: float,
                       x_min: float = 0., x_max: float = 2. * np.pi,
                       t_min: float = 0., t_max: float = 1.,
                       n_x: int = 256, n_t: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the discrete solution of the reaction-diffusion PDE using
        pseudo-spectral operator splitting.

    - u0: initial condition
    - x_min, x_max, t_min, t_max: bounds of domain
    - nu: diffusion coefficient
    - rho: reaction coefficient
    - n_x, n_t: number of points in x and t discretization

    https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/systems_pbc.py#L82
    """
    L, T = 2 * np.pi, t_max - t_min
    dx, dt = L / n_x, T / n_t
    x = np.arange(x_min, x_max, dx)  # not inclusive of the last point
    t = np.linspace(t_min, t_max, num=n_t)
    u = np.zeros((n_x, n_t))
    xx, tt = np.meshgrid(x, t)
    xt = np.vstack([xx.flatten(), tt.flatten()]).T  # (n_x * n_t) x 2

    # fourier frequencies
    IKX_pos, IKX_neg = 1j * np.arange(0, n_x/2+1, 1), 1j * np.arange(-n_x/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    u0 = u0(x[:, None]).squeeze()
    u[:, 0] = u0
    u_ = u0
    for i in range(n_t-1):
        u_ = reaction(u_, rho, dt)
        u_ = diffusion(u_, nu, dt, IKX2)
        u[:, i+1] = u_

    return xt, u.T.flatten()
