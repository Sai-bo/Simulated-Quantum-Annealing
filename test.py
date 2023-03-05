import numpy as np

np.random.seed(0)
num_par = np.random.normal(0, 1, 5)
N = len(num_par)

J = np.outer(num_par, num_par)

print(np.linalg.eigvals(J))