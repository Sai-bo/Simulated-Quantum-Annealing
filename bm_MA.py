import numpy as np
from MA import one_MA_run
from MA_tts import one_MA_tts_run
import time
import csv
import matplotlib.pyplot as plt

def bm_MA(schedule, msf, dropout, input_path):
    
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
    ans, _, _ = one_MA_run(-J, np.zeros(N), schedule, msf, dropout)
    total_time = time.time() - start_time

    ### computing energy
    E = 0
    for i in range(N):
        for j in range(N):
            if J[i][j] == 1:
                E += (1 - (ans[i] * ans[j])) / 2
                
    return ans, total_time, E

##########################################################################################################

def bm_MA_tts(schedule, msf, dropout, input_path, solution):
    
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
    ans, _, _, needed_steps = one_MA_tts_run(-J, np.zeros(N), schedule, solution, msf, dropout)
    total_time = time.time() - start_time
                
    return ans, total_time, needed_steps


##########################################################################################################
##########################################################################################################


if __name__ == "__main__":

    ### annealing steps
    steps = 100000

    ### temperature schedule
    mode = 'exp'

    if mode == "log":
        schedule = [10 / np.log(2 + i) for i in range(steps)]
    elif mode == 'exp':
        schedule = [100 * (1 - 0.0001) ** i for i in range(steps)]
    elif mode == 'linear':
        schedule = [100 * (1 - (i / steps)) for i in range(steps)]

    # init_temp = 100
    # final_temp = 1e-3
    # schedule = [init_temp * ((final_temp / init_temp) ** (i / steps)) for i in range(steps)]
    print(schedule)

    ### factor
    factor = steps

    ### momentum scaling factor
    msf = [min(1, np.sqrt((i + 1) / (steps / factor))) for i in range(steps)]
    # msf = [1 for i in range(steps)]

    ### drop out probability
    dropout = [max(0, 0.5 - ((i + 1) / (2 * steps / factor))) for i in range(steps)]
    # dropout = [0 for i in range(steps)]

    ### benchmark times
    bm_times = 100

##########################################################################################################

    ### g05 file
    input_path = []
    output_path = []
    problem_size = [60]

    for size in problem_size:
        for index in range(1):
            input_path.append(f"./data/g05/g05_{size}.{index}")
            output_path.append(f"./result/MA_result/g05_{size}.{index}_{steps}_{factor}_{mode}")

##########################################################################################################

    for j in range(len(input_path)):
        current_input_path = input_path[j]
        current_output_path = output_path[j]
        with open(current_output_path, "w", newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['', 'cut', 'time', 'ans'])

            for i in range(bm_times):
                ans, total_time, E = bm_MA(schedule, msf, dropout, current_input_path)
                writer.writerow([i+1, E, total_time, ans])

##########################################################################################################

    # ### g05 file
    # input_path = []
    # output_path = []
    # problem_size = [60]

    # for size in problem_size:
    #     for index in range(1):
    #         input_path.append(f"./data/g05/g05_{size}.{index}")
    #         output_path.append(f"./result/MA_result/g05_{size}.{index}_0_{mode}_tts")

##########################################################################################################

    # for j in range(len(input_path)):
    #     current_input_path = input_path[j]
    #     current_output_path = output_path[j]
    #     with open(current_output_path, "w", newline='') as output:
    #         writer = csv.writer(output)
    #         writer.writerow(['', 'tts', 'time', 'ans'])

    #         for i in range(bm_times):
    #             ans, total_time, needed_steps = bm_MA_tts(schedule, msf, dropout, current_input_path, 536)
    #             writer.writerow([i+1, needed_steps, total_time, ans])