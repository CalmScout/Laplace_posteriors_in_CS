"""
Module with functions for data visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
from system import g
from constants import MIN_MU_VISUALIZATION, MAX_MU_VISUALIZATION
from constants import MIN_DELTA_VISUALIZATION, MAX_DELTA_VISUALIZATION


def plot_cost_func(data, eq_idx, mu_plot_step=0.1, delta_plot_step=0.003, number_of_contours=20, mark_min_val=True):
    """
    Plot cost function generated from system of equations and marks minimum value in the grid.
    """
    mu_plot = np.arange(MIN_MU_VISUALIZATION, MAX_MU_VISUALIZATION + mu_plot_step, mu_plot_step)
    delta_plot = np.arange(MIN_DELTA_VISUALIZATION, MAX_DELTA_VISUALIZATION, delta_plot_step)
    MU, DELTA = np.meshgrid(mu_plot, delta_plot)
    vis_arr = np.zeros(MU.shape)
    for i in range(MU.shape[0]):
        for j in range(MU.shape[1]):
            vis_arr[i][j] = g((MU[i][j], DELTA[i][j]), data, eq_idx)
    im = plt.imshow(vis_arr, cmap=cm.RdBu)  # drawing the function
    print('vis_arr.shape:', vis_arr.shape)
    cset = contour(vis_arr, number_of_contours, linewidths=2, cmap=cm.Set2)
    clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    colorbar(im)
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\delta$')
    print('Minimal value in plotting grid:', np.amin(vis_arr))
    # plot point we are looking for - point of function minimum
    if mark_min_val:
        delta_min_idx, mu_min_idx = divmod(vis_arr.argmin(), vis_arr.shape[1])
        plt.plot(mu_min_idx, delta_min_idx, 'wo')
    show()


if __name__ == "__main__":
    from data_generator import data_generator
    data = data_generator()
    eq_idx = 3
    plot_cost_func(data, eq_idx)
