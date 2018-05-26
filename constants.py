"""
Module with constants we use to generate data and solve system.
"""
# number of hidden parameters
N = 100
# number of measurements
M = 20
# noise in acquisition system has normal distribution N(0, 1/beta)
# according to graphs and physical sense we have such constraints: 0 < betta <= 50
BETTA_INIT = 1.0
# 1/GAMMA is regularization term at l1 norm in initial optimization problem
GAMMA_INIT = 0.1
# part of nonzero elements in sparse representation 'w'
PART_NONZERO = 0.05
# nonzero elements in w have uniform distribution U[-W_BOUND, W_BOUND]
W_BOUND = 100
# limits for generating of init values for 'delta'
MIN_DELTA_INIT = 0.1
MAX_DELTA_INIT = 1.0
# each system are solved starting from several initial points (grid of initial approximations)
DELTA_GRID_DENSE = 10
# limits for generating init values for 'mu' variable
# boundaries of mu = [-50, 50] - otherwise machine lacks of precision
MIN_MU_INIT = -30.0
MAX_MU_INIT = 30.0
# number of nodes for mu in generating of grid of initial approximations
MU_GRID_DENSE = 10
# tolerance threshold for system solution - if system solution exceed threshold -
# recompute with other initial approximation
TOL_THRESHOLD_SYSTEM_SOLUTION = 1e-6
# computer zero for solver_scipy module - to separate different cases of equations
COMPUTER_ZERO = 1e-13
# value of delta we replace negative delta_opt
DEFAULT_POSITIVE_DELTA = 0.5
# learning rate for manual gradient descent implementation
LEARNING_RATE = 0.001
# visualization range of mu
MIN_MU_VISUALIZATION = MIN_MU_INIT - abs(MIN_MU_INIT) * 0.3
MAX_MU_VISUALIZATION = MAX_MU_INIT + abs(MAX_MU_INIT) * 0.3
# visualization range of delta
MIN_DELTA_VISUALIZATION = 0.03
MAX_DELTA_VISUALIZATION = 1.0
