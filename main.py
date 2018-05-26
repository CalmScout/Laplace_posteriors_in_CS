import numpy as np
from data_generator import data_generator
from constants import N, M
# from solver import compute_scipy
from data_generator import system_generator
from solver import compute_manual_gdsc
from tools import check_solutions, elbo
import matplotlib.pyplot as plt
from visualization import plot_cost_func

# generate data one time for current experiment
data = data_generator()
# remember generated vector of hidden parameters 'w'
# start iterative process for adjusting parameters mu, delta, gamma, beta
elbo_lst = []
for iter in range(3):
    print('Iteration:', iter)
    # solve systems with current parameters
    for i in range(N):
        compute_manual_gdsc(data, i)
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
plt.figure(1)
print(elbo_lst)
plt.plot(elbo_lst, 'o')
plt.savefig('output/elbo.png', dpi=100)

# print comparison of original values and their approximations
plt.figure(2)
plt.plot(data.w, '^')
plt.plot(data.mu, 'rx')
plt.show()
