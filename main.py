import numpy as np
from data_generator import data_generator
from constants import N, M
from solver import compute_scipy
from data_generator import system_generator
from solver import compute_manual_gdsc
from tools import check_solutions, elbo
import matplotlib.pyplot as plt
from visualization import plot_cost_func
from tools import check_measurement_approximation

# generate data one time for current experiment
data = data_generator()
# remember generated vector of hidden parameters 'w'
w_original = data.w
# start iterative process for adjusting parameters mu, delta, gamma, beta
elbo_lst = []
for iter in range(10):
    print('Iteration:', iter)
    # solve systems with current parameters
    for i in range(N):
        compute_scipy(data, i)
    # check all solutions and rise exceptions if necessary
    check_solutions(data)
    curr_elbo = elbo(data)
    elbo_lst.append(curr_elbo)
    # recompute gamma - formula (19) in the paper
    data.gamma = 0
    for i in range(N):
        data.gamma += data.delta[i] * np.exp(-np.abs(data.mu[i]) / data.delta[i]) + np.abs(data.mu[i])
    data.gamma /= N
    # recompute beta - formula (20) in the paper
    beta_denominator = 0
    for j in range(N):
        beta_denominator += np.matmul(data.Phi[:, j], data.Phi[:, j]) * np.power(data.delta[j], 2)
    beta_denominator *= 4
    beta_denominator += np.power(np.linalg.norm(np.matmul(data.Phi, data.mu) - data.y), 2)
    data.beta = M / beta_denominator
    # update coefficient of system of equations for the next iteration
    data.systems_params = system_generator(data)

# build graph of dependency of elbo from iteration number
print(elbo_lst)
plt.plot(elbo_lst, 'o')
# plt.show()
plt.savefig('output/elbo.png', dpi=100)

# print("Result:")
# for i in range(N):
#     print("{} --> Laplace({}, {})".format(w_original[i], data.mu[i], data.delta[i]))

# print("Error between original and restored signals: {}".format(check_measurement_approximation(data)))
