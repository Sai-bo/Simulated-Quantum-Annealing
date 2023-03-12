import numpy as np
from MA import one_MA_run
import time
import csv
import matplotlib.pyplot as plt

def bm_MA(schedule, input_path):
    
    ### from input file to J matrix
    problem_file = open(input_path)
    r = problem_file.readlines()
    problem_file.close()

    N = int(r[0].split()[0])
    J = np.zeros((N, N))
    for i in range(1, int(r[0].split()[1])+1):
        J[int(r[i].split()[0])-1][int(r[i].split()[1])-1] = int(r[i].split()[2])

    ### annealing
    start_time = time.time()
    ans, _, _ = one_MA_run(-J, np.zeros(N), schedule)
    total_time = time.time() - start_time

    ### computing energy
    E = 0
    for i in range(N):
        for j in range(N):
            if J[i][j] == 1:
                E += (1 - (ans[i] * ans[j])) / 2
                
    return ans, total_time, E


##########################################################################################################
##########################################################################################################


if __name__ == "__main__":

    ### annealing steps
    steps = 100000

    ### temperature schedule
    schedule = [10 / np.log(2 + i) for i in range(steps)]

    ### momentum scaling factor
    msf = [min(1, np.sqrt((i + 1) / 10000)) for i in range(steps)]

    ### drop out probability
    dropout = [max(0, 0.5 - ((i + 1) / 20000)) for i in range(steps)]

    ### benchmark times
    bm_times = 10

##########################################################################################################

    ### g05 file
    input_path = []
    output_path = []
    problem_size = [60]

    for size in problem_size:
        for index in range(10):
            input_path.append(f"./data/g05/g05_{size}.{index}")
            output_path.append(f"./result/MA/g05_{size}.{index}_{steps}")

##########################################################################################################

    for j in range(len(input_path)):
        current_input_path = input_path[j]
        current_output_path = output_path[j]
        with open(current_output_path, "w", newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['', 'cut', 'time', 'ans'])

            for i in range(bm_times):
                ans, total_time, E = bm_MA(schedule, current_input_path)
                writer.writerow([i+1, E, total_time, ans])

