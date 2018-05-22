from data_generator import data_generator
from constants import N
# from solver import compute_scipy
from solver import compute_manual_gdsc
from tools import check_solutions, elbo
import matplotlib.pyplot as plt
from visualization import plot_cost_func
from tools import check_measurement_approximation

# generate data one time for current experiment
data = data_generator()
# # remember generated vector of hidden parameters 'w'
# w_original = data.w
# # start iterative process for adjusting parameters mu, delta, gamma, beta
# elbo_lst = []
# for iter in range(10):
#     # solve systems with current parameters
#     for i in range(N):
#         compute_manual_gdsc(data, i)
#     # check all solutions and rise exceptions if necessary
#     check_solutions(data)
#     curr_elbo = elbo(data)
#     elbo_lst.append(curr_elbo)

# # build graph of dependency of elbo from iteration number
# print(elbo_lst)
# plt.plot(elbo_lst, 'o')
# # plt.show()
# plt.savefig('output/elbo.png', dpi=100)
#
# print("Result:")
# for i in range(N):
#     print("{} --> Laplace({}, {})".format(w_original[i], data.mu[i], data.delta[i]))

# test of plotting cost function we minimize
eq_idx = 3  # system number 3
plot_cost_func(data, eq_idx)
#
# # print("Error between original and restored signals: {}".format(check_measurement_approximation(data)))
