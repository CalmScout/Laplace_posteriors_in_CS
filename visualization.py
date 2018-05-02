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
    # plt.figure()
    # number_of_contours = 10
    # CS = plt.contour(MU, DELTA, G, number_of_contours)
    # plt.clabel(CS, inline=1, fontsize=10)
    # plt.show()
    plt.contour(MU, DELTA, g((MU, DELTA), data, eq_idx))
    plt.show()
