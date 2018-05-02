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
    res = opt.minimize(g, x, args=(data, eq_idx), jac=g_derivative, method=methods[1])
    data.mu[eq_idx], data.delta[eq_idx] = res.x


def compute_manual_gdsc(data, eq_idx, learning_rate=1e-2, tol=1e-11):
    """
    Performs solving system of equations 'eq_idx' by gradient descent method.
    """
    x_init = np.array([data.mu[eq_idx], data.delta[eq_idx]])
    curr_x = np.array(x_init)
    previous_step_size = 1 / tol    # some big value
    while previous_step_size > tol:
        prev_x = curr_x
        # print("curr_x.shape", curr_x.shape)
        # print("g_derivative(prev_x, data, eq_idx).shape", g_derivative(prev_x, data, eq_idx).shape)
        step = g_derivative(prev_x, data, eq_idx).reshape((2, 1))
        curr_x = curr_x - learning_rate * step
        # print("g_derivative:", type(g_derivative(prev_x, data, eq_idx)))
        # print("learning_rate * g_derivative:", type(learning_rate * g_derivative(prev_x, data, eq_idx)))
        # print("curr_x - learning_rate * g_derivative(prev_x, data, eq_idx)", type(curr_x - learning_rate * g_derivative(prev_x, data, eq_idx)))
        # print("curr_x after change:", type(curr_x))
        previous_step_size = np.abs(np.linalg.norm(curr_x - prev_x))
    data.mu[eq_idx], data.delta[eq_idx] = curr_x
