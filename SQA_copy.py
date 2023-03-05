from dataclasses import field
import numpy as np
from scipy.sparse import block_diag
import time
import matplotlib.pyplot as plt


def one_SQA_run_new(J, h, trans_fld_sched, M, T, field_cycling = 1, sd = None, init_state = None, return_pauli_z = False, enable_global_move = False):
    """
    One path-integral Monte Carlo simulated quantum annealing run over the full transverse field strength schedule.
    The goal is to find a state such that sum(J[i, j]*state[i]*state[j]) + sum(h[i]*state[i]) is minimized.
    
    Parameters:
        J (2-D array of float): The matrix representing the coupling field of the problem.
        h (1-D array of float): The vector representing the local field of the problem.
        trans_fld_sched (list[float]): The transeverse field strength schedule for QA.
                                       The number of iterations is implicitly the length of trans_fld_schedule.
        M (int): Number of Trotter replicas. To simulate QA precisely, M should be chosen such that T M / Gamma >> 1.
        T (float): Temperature parameter. Smaller T leads to higher probability of finding ground state.
        sd (default=None): Seed for numpy.random.default_rng().
        init_state (1-D array of int, default=None): The boolean vector representing the initial state.
                                                     If None, a random state is chosen.
        return_pauli_z (bool, default=False): If True, returns a N-spin state averaged over the imaginary time dimension.
                                              If False, returns the raw N*M-spin state.
        enable_global_move (bool, default=Falss): If True, apply global move technique.
        field_cycling (int, default=1): Numbers of cycles in field-cycling. field_cycling=1 is equivalent to not apply field-cycling technique.
    
    Return: final_state (1-D array of int)
    """

    rng = np.random.default_rng(seed = sd)

    # normalize
    norm_coef = np.sqrt(J.shape[0] / (np.sum(J**2) + 0.5 * np.sum(h**2))) # normalization
    J_n = J * norm_coef
    h_n = h * norm_coef


    N = J_n.shape[0]

    J_n = 0.5 * (J_n + J_n.T)
    j = block_diag([J_n / M] * M).toarray()
    h_extended = np.repeat(h_n / M, M)

    Jp_terms = np.eye(N*M, k=N) + np.eye(N*M, k=N*(1-M))
    Jp_terms = 0.5 * (Jp_terms + Jp_terms.T)

    steps = len(trans_fld_sched)
    # Jp_increase_rate = trans_fld_sched[1] - trans_fld_sched[0]
    T_decrement = T / field_cycling

    if init_state is None:
        state = 2 * rng.binomial(1, 0.5, N*M) - np.ones(N*M)
    else:
        state = np.tile(init_state, M)

    dE = np.zeros(N*M)
    Jp_coef = -0.5 * T * np.log(np.tanh(trans_fld_sched[0] / (M * T)))
    for flip in range(N*M):
        dE[flip] = -4 * (j[flip] + Jp_coef * Jp_terms[flip]).dot(state) * state[flip] - 2 * h_extended[flip] * state[flip]

    # Annealing
    for cycle in range(field_cycling):

        for Gamma in trans_fld_sched[0::field_cycling]:
            Jp_last = Jp_coef
            Jp_coef = -0.5 * T * np.log(np.tanh(Gamma / (M * T)))
            Jp_increment = Jp_coef - Jp_last

            # Update Jp terms in dE
            for flip in range(N*M):
                flip_layer = flip // N
                flip_pos = flip % N
                flip_below = (flip_layer - 1) % M * N + flip_pos
                flip_above = (flip_layer + 1) % M * N + flip_pos
                dE[flip] += -4 * Jp_increment * state[flip] * (state[flip_below] + state[flip_above])

            # Global move
            if enable_global_move:
                candidate_pos = 0
                dE_min = 0
                for layer in range(M):
                    dE_min += dE[layer * N]
                for pos in range(N):
                    dE_temp = 0
                    for layer in range(M):
                        dE_temp += dE[layer * N + pos]
                    if dE_temp < dE_min:
                        dE_min = dE_temp
                        candidate_pos = pos

                if rng.binomial(1, np.minimum(np.exp(-dE_min/T), 1.)):
                    for layer in range(M):
                        flip = N * layer + candidate_pos
                        state[flip] *= -1

                        # Update dE (N spins at the same layer for every spins)
                        for flip_to_update in range(N*layer, N*(layer+1)):
                            if flip_to_update == flip:
                                dE[flip_to_update] *= -1
                            else:
                                dE[flip_to_update] += -8 * j[flip_to_update][flip] * state[flip_to_update] * state[flip]


            # Local move
            for flip in range(N*M):
                if rng.binomial(1, np.minimum(np.exp(-dE[flip]/T), 1.)):
                    state[flip] *= -1

                    # Update dE (N spins at the same layer and 2 spins at the layer right above/below it)
                    flip_layer = flip // N
                    flip_pos = flip % N
                    for flip_to_update in range(N*flip_layer, N*(flip_layer+1)):
                        if flip_to_update == flip:
                            dE[flip_to_update] *= -1
                        else:
                            dE[flip_to_update] += -8 * j[flip_to_update][flip] * state[flip_to_update] * state[flip]

                    flip_to_update1 = (flip_layer - 1) % M * N + flip_pos
                    flip_to_update2 = (flip_layer + 1) % M * N + flip_pos
                    dE[flip_to_update1] += -8 * Jp_coef * state[flip] * state[flip_to_update1]
                    dE[flip_to_update2] += -8 * Jp_coef * state[flip] * state[flip_to_update2]

            

        # Field-cycling
        # if cycle != field_cycling - 1:
        #     T -= T_decrement
        #     for Gamma in trans_fld_sched[-1::-100 * field_cycling]:
        #         Jp_coef = -0.5 * T * np.log(np.tanh(Gamma / (M * T)))

        #         for flip in range(N*M):
        #             delta_E = -4 * (j[flip] + Jp_coef * Jp_terms[flip]).dot(state) * state[flip] - 2 * h_extended[flip] * state[flip]
        #             if rng.binomial(1, np.minimum(np.exp(-delta_E/T), 1.)):
        #                 state[flip] *= -1


        state = [int(i) for i in state]
        

    if return_pauli_z:
        final_state = state[:N]
        final_E = np.sum(np.multiply(J,np.outer(final_state,final_state))) + np.dot(h,final_state)
        for i in range(M):
            temp_state = state[i*N:(i+1)*N]
            temp_E = np.sum(np.multiply(J,np.outer(temp_state,temp_state))) + np.dot(h,temp_state)
            if temp_E < final_E:
                final_state = temp_state.copy()
                final_E = temp_E
        return final_state
    else:
        return state


##########################################################################################################


def main():

    sd = 7

    np.random.seed(sd)
    num_par = np.random.normal(0, 1, 5)
    # num_par = np.zeros(3)
    # for i in range(3):
    #     num_par[i] = i+1
    N = len(num_par)

    J = np.outer(num_par, num_par)
    np.fill_diagonal(J, 0)
    h = np.zeros(N)

    norm_coef = np.sqrt(J.shape[0] / (np.sum(J**2) + 0.5 * np.sum(h**2))) # normalization
    J = J * norm_coef
    h = h * norm_coef

    M = 5
    T = 0.05

    steps = 10
    Gamma0 = 10
    Gamma1 = 1e-8
    decay_rate = (Gamma1 / Gamma0)**(1/(steps-1))
    schedule = [Gamma0 * decay_rate**i for i in range(steps)]
    # J_plus0 = 0
    # J_plus1 = 0.5
    # increase_rate = (J_plus1 - J_plus0) / steps
    # schedule = [J_plus0 + increase_rate * i for i in range(steps)]


    # SQA
    start_time = time.time()
    ans = one_SQA_run(J, h, schedule, M, T, sd=sd, return_pauli_z=True)
    total_time = time.time() - start_time

    E_sqa = np.sum(np.multiply(J,np.outer(ans,ans))) + np.dot(h,ans)

    print("-----simulated quantum annealing-----")
    print(f"final state: {ans}")
    print(f"final energy: {E_sqa}; time: {total_time} s")


    # deterministic solver
    from solver import solver
    start_time_solver = time.time()
    ans_solver = solver(J,h)
    total_time_solver = time.time() - start_time_solver

    E_solver = np.sum(np.multiply(J,np.outer(ans_solver,ans_solver))) + np.dot(h,ans_solver)

    print("-----solver-----")
    print(f"ground state: {ans_solver}")
    print(f"ground energy: {E_solver}; time: {total_time_solver} s")

if __name__ == "__main__":
    main()