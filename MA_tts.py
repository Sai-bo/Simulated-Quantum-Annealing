import numpy as np

def one_MA_tts_run(J, h, temp_sched, solution, c_k = None, p_k = None, sd = None, init_state = None):
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

    ### annealing
    record = []
    E_best = 1e7

    for i in range(steps):
        T_k = temp_sched[i]
        w_k = np.multiply(w * c_k[i], rng.binomial(1, 1 - p_k[i], N))

        gamma_k = np.random.gamma(shape=1, scale=1, size=N)
        temp_state = np.sign(h + (J + np.diag(w_k)) @ state - (T_k / 2) * np.diag(gamma_k) @ last_state)
        last_state = state
        state = temp_state

        E_current = np.sum(np.multiply(-J, np.outer(state, state))) + np.dot(h, state)
        record.append(E_current)
        if E_current < E_best:
            E_best = E_current
            state_best = state
            last_state_best = last_state
        
        E = 0
        for i in range(N):
            for j in range(N):
                if J[i][j] == -1:
                    E += (1 - (state_best[i] * state_best[j])) / 2
        
        if E == solution:
            return state_best, last_state_best, record, steps

    return state_best, last_state_best, record, steps, E
