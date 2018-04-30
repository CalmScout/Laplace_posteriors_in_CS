"""
Module with functions for data visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from system import g
from constants import MIN_MU_VISUALIZATION, MAX_MU_VISUALIZATION
from constants import MIN_DELTA_VISUALIZATION, MAX_DELTA_VISUALIZATION


def plot_cost_func(data, eq_idx, mu_plot_step=0.1, delta_plot_step=0.01):
    """
    Plot cost function generated from system of equations.
    """
    mu_plot = np.arange(MIN_MU_VISUALIZATION, MAX_MU_VISUALIZATION, mu_plot_step)
    delta_plot = np.arange(MIN_DELTA_VISUALIZATION, MAX_DELTA_VISUALIZATION, delta_plot_step)
    MU, DELTA = np.meshgrid(mu_plot, delta_plot)
    # x = np.array([data.mu[eq_idx], data.delta[eq_idx]])
    # G = g(x, data, eq_idx)

    # G = g([MU, DELTA], data, eq_idx)

    G = np.zeros((len(delta_plot), len(mu_plot)))
    print("G.shape ==", G.shape)
    for i in range(len(delta_plot)):
        for j in range(len(mu_plot)):
            G[i][j] = g([MU[i][j], DELTA[i][j]], data, eq_idx)
    plt.figure()
    CS = plt.contour(MU, DELTA, G)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()
