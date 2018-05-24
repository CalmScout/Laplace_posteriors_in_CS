"""
Module contains methods for solving systems.
"""
import numpy as np
import scipy.optimize as opt
from system import f, g, g_derivative


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


def compute_manual_gdsc(data, eq_idx, learning_rate=1e-2, tol=1e-4):
    """
    Performs solving system of equations 'eq_idx' by gradient descent method.
    """
    x_init = np.array([data.mu[eq_idx], data.delta[eq_idx]])
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


if __name__ == "__main__":
    from data_generator import data_generator
    data = data_generator()
    print('Start!')
    eq_idx = 3
    compute_manual_gdsc(data, eq_idx)
    print('End!')
    print("Error:", check_scalar(data, eq_idx))
