# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

from jax import numpy as jnp
from flax.core.frozen_dict import FrozenDict

from pinn_jax.derivatives import get_batch_jacobian
from pinn_jax.benchmarks.gene_circuit import DEFAULT_RXN_PARAMS

from typing import Callable


def get_gene_circuit_func(u_hat: Callable, rxn_params: dict = None) -> Callable:
    """maps a time `t` to the set of state variables

        x_m, y_m, X, Y, Y_d, ES1, y_m_R, x_m_R, x_ES1, y_ES1

    implements the equations found here: https://arxiv.org/abs/1307.0178

    see here for more details: https://jhuapl.box.com/s/zn9k1fu96s30g4fx0nk826qfryxaiuj0

    in short:
    - `x_m` and `y_m` are mRNA transcribed from genes `x` and `y`
    - `X` is a generic protein
    - `Y` is a promoter
    - `Y_d` is a fluorescent reporter
    - `ES1` is a holoenyzme
    - `y_m_R` (`y_m:R`), `x_m_r` (`x_m:R`), `x_ES1` (`x:ES1`), and `y_ES1` (`y:ES1`) are intermediate products

    the model has a lot of parameters, specified by `rxn_params`:
    - `k_x_TX`, `k_x_TL`, `k_x_m_deg`, `k_x_plus`, `k_x_minus` are reaction rates for `x` determining transcription,
      translation, mRNA degradation, and #TODO
    - the `k_y_i` constants are defined similarly for `y`
    - `E_tot` is the fixed total concentration of free core RNA polymerase (RNAP)
    - `R_tot` is the fixed total concentration of free ribosome
    - `S1_tot` is the fixed total concentration of primary sigma factor
    - `x_tot` and `y_tot` are the fixed total concentration of #TODO
    - `V_TL` and `V_TX` are rates of progression (in nucleotides per second) of RNAP along the DNA, and ribosome along
      the mRNA, respectively
    """
    batch_jacobian = get_batch_jacobian(u_hat)
    if rxn_params is None:
        rxn_params = DEFAULT_RXN_PARAMS

    # validate that all the needed parameters have been supplied
    required_keys = ['k_x_TX', 'k_x_m_deg', 'k_x_TL', 'k_x_plus', 'k_x_minus',
                     'k_X_plus', 'k_X_minus', 'k_X_deg',
                     'k_y_TX', 'k_y_m_deg', 'k_y_TL', 'k_y_plus', 'k_y_minus',
                     'k_Y_plus', 'k_Y_minus', 'k_Y_deg',
                     'k_ES1_plus', 'k_ES1_minus',
                     'k_mat',
                     'L_x_prot', 'L_y',
                     'V_TL', 'V_TX',
                     'S1_tot', 'R_tot', 'E_tot', 'y_tot', 'x_tot']
    for key in required_keys:
        assert key in rxn_params.keys(), f"`rxn_params` must specify value for {key}!!!"
    n_variables = 10

    def ode_system(params: FrozenDict, points: jnp.ndarray) -> jnp.ndarray:
        u = u_hat(params, points)  # n_batch x n_output
        u_t = batch_jacobian(params, points)  # n_batch x n_output  x n_input

        # unpack time derivatives
        d_x_m, d_y_m, d_X, d_Y = u_t[:, 0, 0], u_t[:, 1, 0], u_t[:, 2, 0], u_t[:, 3, 0]
        d_Y_d, d_ES1 = u_t[:, 4, 0], u_t[:, 5, 0]
        d_y_m_R, d_x_m_R, d_x_ES1, d_y_ES1 = u_t[:, 6, 0], u_t[:, 7, 0], u_t[:, 8, 0], u_t[:, 9, 0]

        # unpack state variables
        x_m, y_m, X, Y, Y_d = u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4]
        ES1, y_m_R, x_m_R, x_ES1, y_ES1 = u[:, 5], u[:, 6], u[:, 7], u[:, 8], u[:, 9]

        # get `E` and `R` from the conservation relation
        E = (rxn_params['E_tot']
             - x_ES1 * (1. + rxn_params['k_x_TX'] * (rxn_params['L_x_prot'] / rxn_params['V_TX']))
             - y_ES1 * (1. + rxn_params['k_y_TX'] * rxn_params['L_y'] / rxn_params['V_TX']) - ES1)

        R = (rxn_params['R_tot']
             - x_m_R * (1. + rxn_params['k_x_TL'] * rxn_params['L_x_prot'] / rxn_params['V_TL'])
             - y_m_R * (1. + rxn_params['k_y_TL'] * rxn_params['L_y'] / rxn_params['V_TL']))

        # calculate ODE residuals
        residuals = jnp.zeros((u.shape[0], n_variables))

        residuals = residuals.at[:, 0].set(d_x_m - (
                rxn_params['k_x_TX'] * x_ES1
                - rxn_params['k_x_m_deg'] * x_m
                - rxn_params['k_X_plus'] * R * x_m
                + (rxn_params['k_X_minus'] + rxn_params['k_x_TL']) * x_m_R))
        residuals = residuals.at[:, 1].set(d_y_m - (
                rxn_params['k_y_TX'] * y_ES1
                - rxn_params['k_y_m_deg'] * y_m
                - rxn_params['k_Y_plus'] * R * y_m
                + (rxn_params['k_Y_minus'] + rxn_params['k_y_TL']) * y_m_R))
        residuals = residuals.at[:, 2].set(d_x_m_R - (
                rxn_params['k_X_plus'] * R * x_m
                - (rxn_params['k_X_minus'] + rxn_params['k_x_TL']) * x_m_R))
        residuals = residuals.at[:, 3].set(d_y_m_R - (
                rxn_params['k_Y_plus'] * R * y_m
                + (rxn_params['k_Y_minus'] + rxn_params['k_y_TL']) * y_m_R))
        residuals = residuals.at[:, 4].set(
            d_X
            - rxn_params['k_x_TL'] * x_m_R
            + rxn_params['k_x_m_deg'] * X)
        residuals = residuals.at[:, 5].set(
            d_Y
            - rxn_params['k_mat'] * Y_d
            + rxn_params['k_Y_deg'] * Y)
        residuals = residuals.at[:, 6].set(
            d_Y_d
            - rxn_params['k_y_TL'] * y_m_R
            + (rxn_params['k_mat'] + rxn_params['k_X_deg']) * Y_d)
        residuals = residuals.at[:, 7].set(
            d_ES1
            - rxn_params['k_ES1_plus'] * E * (rxn_params['S1_tot'] - ES1)
            + rxn_params['k_ES1_minus'] * ES1 +
            rxn_params['k_x_plus'] * ES1 * (rxn_params['x_tot'] - x_ES1) -
            (rxn_params['k_x_minus'] + rxn_params['k_x_TX']) * x_ES1 +
            rxn_params['k_y_plus'] * ES1 * (rxn_params['y_tot'] - y_ES1) -
            (rxn_params['k_y_minus'] + rxn_params['k_y_TX'] * y_ES1))
        residuals = residuals.at[:, 8].set(
            d_x_ES1
            - rxn_params['k_x_plus'] * ES1 * (rxn_params['x_tot'] - x_ES1)
            + rxn_params['k_x_minus'] * x_ES1
            - rxn_params['k_x_TX'] * x_ES1)
        residuals = residuals.at[:, 9].set(
            d_y_ES1
            - rxn_params['k_y_plus'] * ES1 * (rxn_params['y_tot'] - y_ES1)
            + rxn_params['k_y_minus'] * y_ES1
            - rxn_params['k_y_TX'] * y_ES1)

        return residuals

    return ode_system
