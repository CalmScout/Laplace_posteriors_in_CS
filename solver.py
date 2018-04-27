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


def compute(data, eq_idx):
    """
    Performs solving system of equations 'eq_idx'.
    """
    x = np.array([data.mu[eq_idx], data.delta[eq_idx]])
    methods = {0: "Nelder-Mead",
               1: "Powell"}
    res = opt.minimize(g, x, args=(data, eq_idx), jac=g_derivative, method=methods[1])
    data.mu[eq_idx], data.delta[eq_idx] = res.x
