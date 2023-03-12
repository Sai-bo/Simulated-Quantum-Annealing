import numpy as np
# import numba as nb
import time
from matplotlib import pyplot as plt
from solver import *

def one_MA_run(J, h, temp_sched, c_k = None, p_k = None, sd = None, init_state = None):
    """
    One momentum annealing run over the full temparature schedule.
    The goal is to find a state such that sum(J[i, j]*state[i]*state[j]) + sum(h[i]*state[i]) is minimized.
    
    Parameters:
        J (2-D array of float): The matrix representing the coupling field of the problem.
        h (1-D array of float): The vector representing the local field of the problem.
        temp_sched (list[float]): The temparature schedule for SA.
                                  The number of iterations is implicitly the length of temp_schedule.
        c_k (1-D array of float): momentum scaling factor. Multiply momentum coupling by this factor.
                                  If None, c_k = [1, 1, ...]
        p_k (1-D array of float): dropout probability. Randomly decrease momentum coupling to zero with this probability.
                                  If None, p_k = [0, 0, ...]
        sd (default=None): Seed for numpy.random.default_rng().
        init_state (1-D array of int, default=None): The boolean vector representing the initial state.
                                                     If None, a random state is chosen.
    
    Return: final_state (1-D array of int)
    """

    rng = np.random.default_rng(seed = sd)

    N = J.shape[0]
    steps = len(temp_sched)

    ### normalize
    # norm_coef = np.sqrt(N / (np.sum(J**2) + 0.5 * np.sum(h**2))) # normalization
    # J = J * norm_coef
    # h = h * norm_coef
    J = 0.5 * (J + J.T)

    ### initialize momentum couplings
    la = np.max(np.linalg.eigvals(-J))
    w = np.zeros(N)
    coupling_sum = np.zeros(N)
    coupling_sum_C = np.zeros(N)
    C = []

    for i in range(N):
        for j in range(N):
            coupling_sum[i] += abs(J[i][j])
        if la >= coupling_sum[i]:
            C.append(i)

    for i in range(N):
        for j in C:
            coupling_sum_C[i] += abs(J[i][j])
        if i in C:
            w[i] = coupling_sum[i] - (0.5 * coupling_sum_C[i])
        else:
            w[i] = la / 2

    ### initial state
    if init_state is None:
        state = 2 * rng.binomial(1, 0.5, N) - np.ones(N)
        last_state = 2 * rng.binomial(1, 0.5, N) - np.ones(N)
    else:
        state = init_state
        last_state = state

    state_best = state
    last_state_best = last_state

    ### momentum scaling factor and dropout probability
    if c_k is None:
        c_k = np.ones(steps)
    if p_k is None:
        p_k = np.zeros(steps)

    record = []

    ### annealing
    state_best = []
    E_best = -999999
    for i in range(steps):
        T_k = temp_sched[i]
        w_k = np.multiply(w * c_k[i], rng.binomial(1, 1 - p_k[i], N))

        gamma_k = np.random.gamma(shape=1, scale=1, size=N)
        temp_state = np.sign(h + (J + np.diag(w_k)) @ state - (T_k / 2) * np.diag(gamma_k) @ last_state)
        last_state = state
        state = temp_state

        E_current = np.sum(np.multiply(J, np.outer(state, state))) + np.dot(h, state)
        record.append(E_current)
        if E_current > E_best:
            E_best = E_current
            state_best = state
            last_state_best = last_state

    return state_best, last_state_best, record


##########################################################################################################

def temparature_schedule(init_temp, decay_rate, steps, mode = 'EXPONENTIAL'):
    if mode == 'EXPONENTIAL':
        schedule = [init_temp * (1 - decay_rate) ** i for i in range(steps)]
        return schedule

    if mode == 'LINEAR':
        schedule = [init_temp - decay_rate * i for i in range(steps)]
        return schedule

    if mode == 'LOGARITHM':
        schedule = [init_temp * np.log(2) / np.log(2 + i) for i in range(steps)]
        return schedule

##########################################################################################################

def main():
    ##########################################################################################################

    # np.random.seed(100)

    # density = 0.6
    N = 10

    # J = np.random.binomial(1, density, (N, N))
    J = np.random.uniform(-1, 1, (N, N))
    # J = np.random.randint(low=-1,high=2, size=(N,N))
    np.fill_diagonal(J, 0)
    J = 0.5 * (J + J.T)
    h = np.zeros(N)

    # norm_coef = np.sqrt(J.shape[0] / (np.sum(J**2) + 0.5 * np.sum(h**2))) # normalization
    # J = J * norm_coef
    # h = h * norm_coef

    ### annealing steps
    steps = 30000

    ### temparature schedule
    # init_temp = 100
    # schedule = temparature_schedule(init_temp, 0.0001, steps, mode='LOGARITHM')
    schedule = [10 / np.log(2 + i) for i in range(steps)]

    ### momentum scaling factor
    msf = [min(1, np.sqrt((i + 1) / 1000)) for i in range(steps)]

    ### drop out probability
    dropout = [max(0, 0.5 - ((i + 1) / 2000)) for i in range(steps)]

    ### annealing test
    state, last_state, record = one_MA_run(-J, h, schedule, msf, dropout)
    E_MA = np.sum(np.multiply(J, np.outer(state, state))) + np.dot(h, state)
    print(state)
    print(last_state)
    print(E_MA)


    right_sol = solver(J, h)
    E_solver = np.sum(np.multiply(J, np.outer(right_sol, right_sol))) + np.dot(h, right_sol)
    print(right_sol)
    print(E_solver)

    plt.figure(figsize=(15, 3))
    plt.plot(record)
    plt.show()


if __name__ == "__main__":
    main()