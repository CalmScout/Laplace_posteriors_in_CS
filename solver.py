"""
Module contains methods for solving systems.
"""
import numpy as np
import scipy.optimize as opt
from system import f, g, g_derivative
from constants import MIN_DELTA_INIT, MAX_DELTA_INIT, DELTA_GRID_DENSE, MIN_MU_INIT,\
    MAX_MU_INIT, MU_GRID_DENSE, TOL_THRESHOLD_SYSTEM_SOLUTION


def check(data, eq_idx):
    """
    :return: error of system of equations solution
    """
    x = np.array([data.mu[eq_idx], data.delta[eq_idx]])
    return f(x, data, eq_idx)


def check_scalar(data, eq_idx):
    """
    Compute l2 norm of error of current system of equations solution.
    """
    res = check(data, eq_idx)
    return np.linalg.norm(res)


def compute_scipy(data, eq_idx):
    """
    Performs solving system of equations 'eq_idx'.
    """
    x = np.array([data.mu[eq_idx], data.delta[eq_idx]])
    methods = {0: "Nelder-Mead",
               1: "Powell"}
    res = opt.minimize(g, x, args=(data, eq_idx), jac=g_derivative, method=methods[0])
    data.mu[eq_idx], data.delta[eq_idx] = res.x


def compute_manual_gdsc(data, eq_idx, learning_rate=1e-3, tol=TOL_THRESHOLD_SYSTEM_SOLUTION * 1e-3):
    """
    Performs solving system of equations 'eq_idx' by gradient descent method.
    """
    x_init = _compute_init_approx_for_gdsc(data, eq_idx)
    curr_x = np.array(x_init)
    previous_step_size = 1 / tol    # some big value
    hist = []   # list of approximations that can be used for plotting
    while previous_step_size > tol:
        prev_x = curr_x
        hist.append(curr_x)
        step = g_derivative(prev_x, data, eq_idx).reshape((2, 1))
        curr_x = curr_x - learning_rate * step
        previous_step_size = np.abs(np.linalg.norm(curr_x - prev_x))
    data.mu[eq_idx], data.delta[eq_idx] = curr_x
    return hist


def _compute_init_approx_for_gdsc(data, eq_idx):
    """
    Computes initial value of gradient descent approximation algorithm.
    """
    mu = np.linspace(MIN_MU_INIT, MAX_MU_INIT, MU_GRID_DENSE)
    delta = np.linspace(MIN_DELTA_INIT, MAX_DELTA_INIT, DELTA_GRID_DENSE)
    curr_g_min = float("inf")
    mu_init = None
    delta_init = None
    for mu_el in mu:
        for delta_el in delta:
            x = np.array([mu_el, delta_el])
            if g(x, data, eq_idx) < curr_g_min:
                mu_init = mu_el
                delta_init = delta_el
                curr_g_min = g([mu_el, delta_el], data, eq_idx)
    return np.array([mu_init, delta_init]).reshape((2, 1))


if __name__ == "__main__":
    from data_generator import data_generator
    data = data_generator()
    print('Start!')
    eq_idx = 3
    compute_manual_gdsc(data, eq_idx)
    print('End!')
    print("Error:", check_scalar(data, eq_idx))
