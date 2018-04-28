"""
Additional tools.
"""
from constants import N, M, TOL_THRESHOLD
from solver import check_scalar
import numpy as np


def check_solutions(data):
    """
    Check solutions of all systems and raise exception in case of error.
    """
    for i in range(N):
        if data.delta[i] <= 0:
            raise ValueError("delta have to be positive!")
        if check_scalar(data, i) > TOL_THRESHOLD:
            raise ValueError("Solution is not precise! Error: {}".format(check_scalar(data, i)))


def elbo(data):
    """
    Computes Evidence Lower Bound (ELBO).
    """
    acc = 0.0
    for i in range(N):
        # in paper formulas (13), (14)
        term_1 = data.delta[i] * np.exp(-np.abs(data.mu[i]) / data.delta[i]) + np.abs(data.mu[i])
        # in paper formula (16)
        term_2 = -(data.beta / 2) * ((2 * data.delta[i] * data.delta[i] + data.mu[i] * data.mu[i]) * np.linalg.norm(data.Phi[:, i]) ** 2 - \
                                     2 * np.dot(data.z[:, i], data.Phi[:, i]) * data.mu[i])
        # in paper formula (12)
        term_3 = np.log(data.delta[i])
        acc += term_1 + term_2 + term_3
    return acc


def recompute_parameters(data):
    """
    Recomputes gamma and beta parameters
    """
    new_gamma = 0.0
    for i in range(N):
        new_gamma += data.delta[i] * np.exp(-np.abs(data.mu[i]) / data.delta[i]) + np.abs(data.mu[i])
    new_gamma /= N
    acc = 0.0
    for i in range(N):
        acc += np.dot(data.Phi[:, i], data.Phi[:, i]) * data.delta[i] * data.delta[i]
    new_beta = M / (np.linalg.norm(np.dot(data.Phi, data.mu) - data.y) + 4 * acc)
    return new_gamma, new_beta
