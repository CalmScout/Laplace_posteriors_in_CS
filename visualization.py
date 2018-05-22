"""
Module with functions for data visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from system import g
from constants import MIN_MU_VISUALIZATION, MAX_MU_VISUALIZATION
from constants import MIN_DELTA_VISUALIZATION, MAX_DELTA_VISUALIZATION


def plot_cost_func(data, eq_idx, mu_plot_step=0.1, delta_plot_step=0.01, number_of_contours = 10):
    """
    Plot cost function generated from system of equations.
    """
    mu_plot = np.arange(MIN_MU_VISUALIZATION, MAX_MU_VISUALIZATION, mu_plot_step)
    delta_plot = np.arange(MIN_DELTA_VISUALIZATION, MAX_DELTA_VISUALIZATION, delta_plot_step)
    MU, DELTA = np.meshgrid(mu_plot, delta_plot)
    vis_arr = np.zeros(MU.shape)
    for i in range(MU.shape[0]):
        for j in range(MU.shape[1]):
            vis_arr[i][j] = g((MU[i][j], DELTA[i][j]), data, eq_idx)
    CS = plt.contour(MU, DELTA, vis_arr, number_of_contours)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\delta$')
    plt.show()
