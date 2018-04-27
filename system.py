"""
Module contains system of equations and objective function, generated from it.
"""
import numpy as np


def f(x, data, eq_idx):
    """
    System of equations.
    :return:
    """
    res = np.zeros(len(x))
    a, b, c = data.systems_params[eq_idx]
    res[0] = a * x[0] + b + c * np.sign(x[0]) * (1 - np.exp(-np.abs(x[0]) / x[1]))
    res[1] = -1 / x[1] + 2 * a * x[1] + c * np.exp(-np.abs(x[0]) / x[1]) * (1 + np.abs(x[0]) / x[1])
    return res


def g(x, data, eq_idx):
    """
    Objective function, opens 'sign' and 'abs'.
    """
    if x[0] > 0:
        return _g_1(x, data, eq_idx)
    elif x[0] < 0:
        return _g_3(x, data, eq_idx)
    return _g_2(x, data, eq_idx)


def g_derivative(x, data, eq_idx):
    """
    Derivative(gradient) of objective function.
    """
    if x[0] > 0:
        return _g_1_derivative(x, data, eq_idx)
    elif x[0] < 0:
        return _g_3_derivative(x, data, eq_idx)
    return _g_2_derivative(x, data, eq_idx)


def _g_1(x, data, eq_idx):
    """
    Objective function for constraints mu > 0, delta > 0
    Notation: mu == x[0], delta == x[1]
    """
    a, b, c = data.systems_params[eq_idx]
    return 0.5 * ((a * x[0] + b + c * (1 - np.exp(-x[0] / x[1]))) ** 2 +
                  (-1 / x[1] + 2 * a * x[1] + c * np.exp(-x[0] / x[1]) * (1 + x[0] / x[1])) ** 2)


def _g_1_derivative(x, data, eq_idx):
    """
    Derivative of g_1. Returns [d g_1 / d mu, d g_1 / d delta]
    Notation: mu == x[0], delta == x[1]
    """
    a, b, c = data.systems_params[eq_idx]
    res = np.zeros(len(x))
    res[0] = (a * x[0] + b + c * (1 - np.exp(-x[0] / x[1]))) * \
             (a + c * np.exp(-x[0] / x[1]) / x[1]) - \
             (-1 / x[1] + 2 * a * x[1] + c * np.exp(-x[0] / x[1]) * (1 + x[0] / x[1])) * \
             c * x[0] * np.exp(-x[0] / x[1]) / (x[1] ** 2)
    res[1] = (a * x[0] + b + c * (1 - np.exp(-x[0] / x[1]))) * np.exp(-x[0] / x[1]) * \
             (-x[0] / x[1] ** 2) * c + \
             (-1 / x[1] + 2 * a * x[1] + c * np.exp(-x[0] / x[1]) * (1 + x[0] / x[1])) * \
             (1 / x[1] ** 2 + 2 * a + c * np.exp(-x[0] / x[1]) * (1 + x[0] / x[1]) *
              x[0] / x[1] ** 2 - c * np.exp(-x[0] / x[1]) * x[0] / x[1] ** 2)
    return res


def _g_2(x, data, eq_idx):
    """
    Objective function for constraints mu == 0, delta > 0
    Notation: mu == x[0], delta == x[1]
    """
    a, b, c = data.systems_params[eq_idx]
    return 0.5 * (b ** 2 + (-1 / x[1] + 2 * a * x[1] + c) ** 2)


def _g_2_derivative(x, data, eq_idx):
    """
    Derivative of g_2. Returns [d g_2 / d mu, d g_2 / d delta]
    Notation: mu == x[0], delta == x[1]
    """
    res = np.zeros(len(x))
    a, b, c = data.systems_params[eq_idx]
    res[0] = 0
    res[1] = 1 / x[1] ** 2 + 2 * a
    return res


def _g_3(x, data, eq_idx):
    """
    Objective function for constraints mu < 0, delta > 0
    Notation: mu == x[0], delta == x[1]
    """
    a, b, c = data.systems_params[eq_idx]
    return 0.5 * ((a * x[0] + b - c * (1 - np.exp(x[0] / x[1]))) ** 2 +
                  (-1 / x[1] + 2 * a * x[1] + c * np.exp(x[0] / x[1]) * (1 - x[0] / x[1])) ** 2)


def _g_3_derivative(x, data, eq_idx):
    """
    Derivative of g_3. Returns [d g_3 / d mu, d g_3 / d delta]
    Notation: mu == x[0], delta == x[1]
    """
    res = np.zeros(len(x))
    a, b, c = data.systems_params[eq_idx]
    res[0] = (a * x[0] + b - c * (1 - np.exp(x[0] / x[1]))) * \
             (a + c * np.exp(x[0] / x[1]) / x[1]) + \
             (-1 / x[1] + 2 * a * x[1] + c * np.exp(x[0] / x[1]) *
              (1 - x[0] / x[1])) * (c * np.exp(x[0] / x[1]) / x[1] *
                                    (1 - x[0] / x[1]) + c * np.exp(x[0] / x[1]) * (-1 / x[1]))
    res[1] = (a * x[0] + b - c * (1 - np.exp(x[0] / x[1]))) * \
             c * np.exp(x[0] / x[1]) * (- x[0] / x[1] ** 2) + \
             (-1 / x[1] + 2 * a * x[1] + c * np.exp(x[0] / x[1]) * (1 - x[0] / x[1])) * \
             (1 / x[1] ** 2 + 2 * a + c * np.exp(x[0] / x[1]) * (-x[0] / x[1] ** 2) * (1 - x[0] / x[1]) +
              c * np.exp(x[0] / x[1]) * (x[0] / x[1] ** 2))
    return res
