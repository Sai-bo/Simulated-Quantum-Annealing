import numpy as np

def solver(J, h):
    """
    A simple deterministic solver that used to compare the performance of SQA algorithm.
    """

    N = J.shape[0]
    state = np.ones(N)
    E = np.sum(np.multiply(J,np.outer(state,state))) + np.dot(h,state)
    E_min = E
    final_state = state

    for i in range(2**N):
        temp_num = 2**N + i
        temp_str = bin(temp_num)[3:]
        for j in range(len(temp_str)):
            state[j] = 2 * int(temp_str[j]) - 1

        E = np.sum(np.multiply(J,np.outer(state,state))) + np.dot(h,state)

        if E < E_min:
            E_min = E
            final_state = state.copy()

    return final_state