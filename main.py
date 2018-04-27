from data_generator import data_generator
from constants import N
from solver import compute, check_scalar

data = data_generator()
for i in range(N):
    compute(data, i)

# checking all solutions
for i in range(N):
    print(check_scalar(data, i))
