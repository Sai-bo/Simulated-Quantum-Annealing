import numpy as np
# import numba as nb
import time
from solver import solver

def one_MA_run(J, h, temp_sched, momentum_scaling_factor, sd = None, init_state = None):
    """
    One momentum annealing run over the full temparature schedule.
    The goal is to find a state such that sum(J[i, j]*state[i]*state[j]) + sum(h[i]*state[i]) is minimized.
    
    Parameters:
        J (2-D array of float): The matrix representing the coupling field of the problem.
        h (1-D array of float): The vector representing the local field of the problem.
        temp_sched (list[float]): The temparature schedule for SA.
                                  The number of iterations is implicitly the length of temp_schedule.
        sd (default=None): Seed for numpy.random.default_rng().
        init_state (1-D array of int, default=None): The boolean vector representing the initial state.
                                                     If None, a random state is chosen.
    
    Return: final_state (1-D array of int)
    """

    rng = np.random.default_rng(seed = sd)

    N = J.shape[0]
    steps = len(temp_sched)

    # normalize
    norm_coef = np.sqrt(N / (np.sum(J**2) + 0.5 * np.sum(h**2))) # normalization
    J = J * norm_coef
    h = h * norm_coef
    J = 0.5 * (J + J.T)

    # initialize momentum couplings
    la = np.max(np.linalg.eigvals(J))
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
            w[i] = coupling_sum[i] - 0.5 * coupling_sum_C[i]
        else:
            w[i] = la / 2
    
    if init_state is None:
        state = 2 * rng.binomial(1, 0.5, N) - np.ones(N)
        last_state = 2 * rng.binomial(1, 0.5, N) - np.ones(N)
    else:
        state = init_state
        last_state = state

    # annealing
    for i in range(steps):
        T = temp_sched[i]
        w_i = w * momentum_scaling_factor[i]
        gamma = np.random.gamma(shape=1, scale=1, size=N)
        temp_state = np.sign(h + (J + np.diag(w_i)).dot(state) - T / 2 * np.diag(gamma).dot(last_state))
        last_state = state
        state = temp_state

    return state


##########################################################################################################

def temparature_schedule(init_temp, decay_rate, steps, mode = 'EXPONENTIAL'):
    if mode == 'EXPONENTIAL':
        schedule = [init_temp * (1 - decay_rate) ** i for i in range(steps)]
        return schedule

    if mode == 'LINEAR':
        schedule = [init_temp - decay_rate * i for i in range(steps)]
        return schedule

    # if mode == 'REVERSE':

    #     return schedule

##########################################################################################################


def main():
    np.random.seed(0)
    num_par = np.random.normal(0, 1, 5)
    N = len(num_par)

    J = np.outer(num_par, num_par)
    np.fill_diagonal(J, 0)
    h = np.zeros(N)

    norm_coef = np.sqrt(J.shape[0] / (np.sum(J**2) + 0.5 * np.sum(h**2))) # normalization
    J = J * norm_coef
    h = h * norm_coef

    steps = 10000

    # schedule
    schedule = temparature_schedule(100, 0.001, steps)

    # momentum scaling factor
    msf = [1 - (1 / (i + 1)) for i in range(steps)]


    # annealing test
    sol = one_MA_run(J, h, schedule, msf)
    print(sol)
    right_sol = solver(J, h)
    print(right_sol)




if __name__ == "__main__":
    main()