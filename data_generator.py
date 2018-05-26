import numpy as np
from constants import N, M, BETTA_INIT, PART_NONZERO, W_BOUND, MIN_DELTA_INIT, MAX_DELTA_INIT
from data import Data


def data_generator():
    """
    Generates data for experiment run.
    :return: Data object
    """
    data = Data()
    # assume that sparse representation have only 'part_nonzero' nonzero elements
    number_nonzero = int(N * PART_NONZERO)
    # we don't keep noise in object instance because we don't know it values in real life
    noise = np.random.normal(0.0, 1.0 / BETTA_INIT, (M, 1))
    # Each column of acquisition matrix Phi is normalized random vector, each element of which has standard Gauss
    # distribution N(0, 1)
    data.Phi = np.random.randn(M, N)
    for j in range(data.Phi.shape[1]):
        curr_norm = np.linalg.norm(data.Phi[:, j])
        data.Phi[:, j] /= curr_norm
    data.w = np.zeros((N, 1))
    # generate positions for nonzero elements in hidden parameters state vector
    positions_nonzero = np.random.permutation(np.arange(N))[:number_nonzero]
    # fill in nonzero elements from uniform(-100, 100) distribution
    for i_nonzero in positions_nonzero:
        data.w[i_nonzero] = np.random.rand() * 2 * W_BOUND - W_BOUND
    # generate y
    data.y = np.matmul(data.Phi, data.w) + noise
    data.mu, data.delta = init_values_generator()
    # each vector z_i contains M components and we have N such vectors
    # so in our notation z_i means z[:, i]
    data.z = np.zeros((M, N))
    data.y = np.reshape(data.y, (len(data.y),))
    for j in range(data.z.shape[1]):  # iterate over columns
        data.z[:, j] = data.y
        for i in range(data.z.shape[1]):
            if not i == j:
                data.z[:, j] -= data.Phi[:, i] * data.mu[i]
    data.systems_params = system_generator(data)
    return data


def init_values_generator():
    """
    Generates initial approximations for vectors 'mu' and 'delta'
    :return:
    """
    mu_init = np.random.normal(0.0, 1.0, (N, 1))
    delta_init = np.random.uniform(low=MIN_DELTA_INIT, high=MAX_DELTA_INIT, size=(N, 1))
    return mu_init, delta_init


def system_generator(data):
    """
    Computes parameters for all systems of equations.
    """

    def _generate_params(i):
        """
        Generates parameters (a, b, c) for system with unknowns 'mu_i', 'delta_i'
        """
        a = data.beta * np.linalg.norm(data.Phi[:, i])
        b = -data.beta * np.dot(data.z[:, i], data.Phi[:, i])
        c = 1 / data.gamma
        return a, b, c

    result = []
    for i in range(N):
        result.append(_generate_params(i))
    return result
