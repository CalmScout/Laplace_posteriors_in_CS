from data_generator import data_generator
from constants import N
from solver import compute
from tools import check_solutions, elbo
import matplotlib.pyplot as plt

# generate data one time for current experiment
data = data_generator()
# remember generated vector of hidden parameters 'w'
w_original = data.w
# start iterative process for adjusting parameters mu, delta, gamma, beta
elbo_lst = []
for iter in range(10):
    # solve systems with current parameters
    for i in range(N):
        compute(data, i)
    # check all solutions and rise exceptions if necessary
    check_solutions(data)
    curr_elbo = elbo(data)
    elbo_lst.append(curr_elbo)

# build graph of dependency of elbo from iteration number
print(elbo_lst)
plt.plot(elbo_lst, 'o')
# plt.show()
plt.savefig('elbo.png', dpi=100)

print("Result:")
for i in range(N):
    print("{} --> Laplace({}, {})".format(w_original[i], data.mu[i], data.delta[i]))
