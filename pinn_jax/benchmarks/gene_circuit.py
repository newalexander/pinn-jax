# Copyright 2023, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved. # Distributed under the terms of the Apache 2.0 License.

import numpy as np

from scipy.integrate import solve_ivp

DEFAULT_RXN_PARAMS = {
    'k_x_TX': 0.05,  # s^-1
    'k_x_m_deg': 0.0018,  # s^-1
    'k_x_TL': 0.05,  # s^-1
    'k_x_plus': 3E7,  # M^-1 s^-1
    'k_x_minus': 0.24,  # s^-1

    'k_X_plus': 3E7,  # M^-1 s^-1
    'k_X_minus': 6.,  # s^-1
    'k_X_deg': 0.,  # s^-1 (?) -- set to zero by assumption

    'k_y_TX': 0.05,  # s^-1
    'k_y_m_deg': 0.0018,  # s^-1
    'k_y_TL': 0.05,  # s^-1
    'k_y_plus': 3E7,  # M^-1 s^-1
    'k_y_minus': 0.48,  # s^-1

    'k_Y_plus': 3E7,  # M^-1 s^-1
    'k_Y_minus': 18.,  # s^-1
    'k_Y_deg': 0.,  # s^-1 (?) -- set to zero by assumption

    'k_ES1_plus': 3E7,  # M^-1 s^-1
    'k_ES1_minus': 7.8E-3,  # s^-1

    'k_mat': 0.002,  # s^-1

    'L_x_prot': 800.,  # bp
    'L_y': 800.,  # bp

    'V_TX': 3.,  # bp s^-1
    'V_TL': 4.,  # bp s^-1

    'S1_tot': 30E-9,  # nM
    'R_tot': 1500E-9,  # nM
    'E_tot': 100E-9,  # nM
    'y_tot': 2E-9,  # nM
    'x_tot': 20E-9  # nM
}


def rk_solution(t_min=0.0, t_max=0.1):
    rxn_params = DEFAULT_RXN_PARAMS
    t_eval = np.linspace(t_min, t_max, num=32)
    ics = np.zeros((10,))

    def ode_func(_, _u):
        x_m, y_m, x_m_R, y_m_R, X = _u[0], _u[1], _u[2], _u[3], _u[4]
        Y_d, Y, ES1, x_ES1, y_ES1 = _u[5], _u[6], _u[7], _u[8], _u[9]

        E = (rxn_params['E_tot']
             - x_ES1 * (1. + rxn_params['k_x_TX'] * (rxn_params['L_x_prot'] / rxn_params['V_TX']))
             - y_ES1 * (1. + rxn_params['k_y_TX'] * rxn_params['L_y'] / rxn_params['V_TX']) - ES1)

        R = (rxn_params['R_tot']
             - x_m_R * (1. + rxn_params['k_x_TL'] * rxn_params['L_x_prot'] / rxn_params['V_TL'])
             - y_m_R * (1. + rxn_params['k_y_TL'] * rxn_params['L_y'] / rxn_params['V_TL']))

        _du = np.zeros_like(_u)
        # x_m
        _du[0] = (rxn_params['k_x_TX'] * x_ES1
                  - rxn_params['k_x_m_deg'] * x_m
                  - rxn_params['k_X_plus'] * R * x_m
                  + (rxn_params['k_X_minus'] + rxn_params['k_x_TL']) * x_m_R)
        # y_m
        _du[1] = (rxn_params['k_y_TX'] * y_ES1
                  - rxn_params['k_y_m_deg'] * y_m
                  - rxn_params['k_Y_plus'] * R * y_m
                  + (rxn_params['k_Y_minus'] + rxn_params['k_y_TL']) * y_m_R)
        # x_m_R
        _du[2] = (rxn_params['k_X_plus'] * x_m
                  - (rxn_params['k_X_minus'] + rxn_params['k_x_TX']) * x_m_R)
        # y_m_R
        _du[3] = (rxn_params['k_X_plus'] * R * y_m
                  - (rxn_params['k_Y_minus'] + rxn_params['k_y_TL']) * y_m_R)
        # X
        _du[4] = (rxn_params['k_x_TL'] * x_m_R
                  - rxn_params['k_X_deg'] * X)
        # Y_d
        _du[5] = (rxn_params['k_y_TL'] * y_m_R
                  - (rxn_params['k_mat'] + rxn_params['k_Y_deg']) * Y_d)
        # Y
        _du[6] = (rxn_params['k_mat'] * Y_d
                  - rxn_params['k_Y_deg'])
        # ES1
        _du[7] = (rxn_params['k_ES1_plus'] * E * (rxn_params['S1_tot'] - ES1)
                  - rxn_params['k_ES1_minus'] * ES1
                  - rxn_params['k_x_plus'] * ES1 * (rxn_params['x_tot'] - x_ES1)
                  - (rxn_params['k_x_minus'] + rxn_params['k_x_TX']) * x_ES1
                  + rxn_params['k_y_plus'] * ES1 * (rxn_params['y_tot'] - y_ES1)
                  - (rxn_params['k_y_minus'] + rxn_params['k_y_TX'] * y_ES1))
        # x_ES1
        _du[8] = (rxn_params['k_x_plus'] * ES1 * (rxn_params['x_tot'] - x_ES1)
                  - rxn_params['k_x_minus'] * x_ES1
                  + rxn_params['k_x_TX'] * x_ES1)
        # y_ES1
        _du[9] = (rxn_params['k_y_plus'] * ES1 * (rxn_params['y_tot'] - y_ES1)
                  - rxn_params['k_y_minus'] * y_ES1
                  + rxn_params['k_y_TX'] * y_ES1)
        return _du

    return solve_ivp(ode_func, (t_min, t_max), ics, t_eval=t_eval)


def test_rk():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('darkgrid')

    cols = ['x_m', 'y_m', 'x_m_R', 'y_m_R', 'X', 'Y_d', 'Y', 'ES1', 'x_ES1', 'y_ES1']
    t_max = 10
    solution = rk_solution(t_max=t_max)
    u = pd.DataFrame(solution.y.T, columns=cols)
    u = u.reset_index().assign(t=solution.t).melt(id_vars=['index', 't'])
    u.to_csv(f'foo_{t_max}.csv', index=False)
    sns.relplot(x='t', y='value', col='variable', col_wrap=5, data=u, kind='line')
    plt.waitforbuttonpress()


if __name__ == '__main__':
    test_rk()
