import numpy as np
from SQA import one_SQA_run
from SQA_copy import one_SQA_run_new
import time
import csv
import matplotlib.pyplot as plt

def bm(schedule, M, T, path, F=1, G=False):

    problem_file = open(path)
    r = problem_file.readlines()
    problem_file.close()
    N = int(r[0].split()[0])
    J = np.zeros((N, N))
    for i in range(1, int(r[0].split()[1])+1):
        J[int(r[i].split()[0])-1][int(r[i].split()[1])-1] = int(r[i].split()[2])

    start_time = time.time()
    # ans, count, non_count = one_SQA_run(J, np.zeros(N), schedule, M, T, field_cycling=F, return_pauli_z=True, enable_global_move=G)
    ans = one_SQA_run(J, np.zeros(N), schedule, M, T, field_cycling=F, return_pauli_z=True, enable_global_move=G)
    total_time = time.time() - start_time

    E = 0
    for i in range(N):
        for j in range(N):
            if J[i][j] == 1:
                E += (1 - (ans[i] * ans[j])) / 2

    return ans, total_time, E

def bm_new(schedule, M, T, path, F=1, G=False):

    problem_file = open(path)
    r = problem_file.readlines()
    problem_file.close()
    N = int(r[0].split()[0])
    J = np.zeros((N, N))
    for i in range(1, int(r[0].split()[1])+1):
        J[int(r[i].split()[0])-1][int(r[i].split()[1])-1] = int(r[i].split()[2])

    start_time = time.time()
    ans = one_SQA_run_new(J, np.zeros(N), schedule, M, T, field_cycling=F, return_pauli_z=True, enable_global_move=G)
    total_time = time.time() - start_time

    E = 0
    for i in range(N):
        for j in range(N):
            if J[i][j] == 1:
                E += (1 - (ans[i] * ans[j])) / 2

    return ans, total_time, E

if __name__ == "__main__":

    M = 20
    T = 0.03

    steps_array = [200, 400, 600]
    for steps in steps_array:
    # steps = 1000
        Gamma0 = 10
        Gamma1 = 1e-8
        decay_rate = (Gamma1 / Gamma0)**(1/(steps-1))
        schedule = [Gamma0 * decay_rate**i for i in range(steps)]
        # J_plus0 = 0
        # J_plus1 = 0.3
        # increase_rate = (J_plus1 - J_plus0) / steps
        # schedule = [J_plus0 + increase_rate * i for i in range(steps)]

        bm_times = 10
        
        # g05
        path = []
        problem_size = [100]

        for size in problem_size:
            for index in range(10):
                path.append(f"./data/g05/g05_{size}.{index}")

        # ans, total_time, E = bm(schedule, M, T, path[0])
        # print(ans, total_time, E)

        # for current_path in path:
        #     # output_file_name = current_path.split("/")[-1]
        #     with open(f"./result/new_result/{steps}_{M}_{T}_old.csv", "w", newline='') as output:
        #         writer = csv.writer(output)
        #         writer.writerow(['', 'cut', 'time', 'ans'])

        #         f_count = 0
        #         non_f_count = 0
        #         for i in range(bm_times):
        #             ans, total_time, E, count, non_count = bm(schedule, M, T, current_path, G=False)
        #             writer.writerow([i+1, E, total_time, ans])
        #             f_count += count
        #             non_f_count += non_count
        #         print(f_count)
        #         print(non_count)

        # for current_path in path:
        # for j in range(len(path)):
        #     current_path = path[j]
        #     # output_file_name = current_path.split("/")[-1]
        #     with open(f"./result/MaxCut_Benchmark/{steps}_{M}_{T}_{j}.csv", "w", newline='') as output:
        #         writer = csv.writer(output)
        #         writer.writerow(['', 'cut', 'time', 'ans'])

        #         for i in range(bm_times):
        #             ans, total_time, E = bm(schedule, M, T, current_path, G=False)
        #             writer.writerow([i+1, E, total_time, ans])

        # for current_path in path:
        #     # output_file_name = current_path.split("/")[-1]
        #     with open(f"./result/new_result/{steps}_{M}_{T}_G_old.csv", "w", newline='') as output:
        #         writer = csv.writer(output)
        #         writer.writerow(['', 'cut', 'time', 'ans'])

        #         for i in range(bm_times):
        #             ans, total_time, E = bm(schedule, M, T, current_path, G=True)
        #             writer.writerow([i+1, E, total_time, ans])

        # for current_path in path:
        for j in range(len(path)):
            current_path = path[j]
            # output_file_name = current_path.split("/")[-1]
            with open(f"./result/new_result/{steps}_{M}_{T}_{j}_G.csv", "w", newline='') as output:
                writer = csv.writer(output)
                writer.writerow(['', 'cut', 'time', 'ans'])

                for i in range(bm_times):
                    ans, total_time, E = bm(schedule, M, T, current_path, G=True)
                    writer.writerow([i+1, E, total_time, ans])

    # for steps in steps_array:
    # # steps = 1000
    #     Gamma0 = 10
    #     Gamma1 = 1e-8
    #     decay_rate = Gamma1 - Gamma0 / steps
    #     schedule = [Gamma0 + decay_rate*i for i in range(steps)]
    #     # J_plus0 = 0
    #     # J_plus1 = 0.3
    #     # increase_rate = (J_plus1 - J_plus0) / steps
    #     # schedule = [J_plus0 + increase_rate * i for i in range(steps)]

    #     bm_times = 100
        
    #     # g05
    #     path = []
    #     problem_size = [100]

    #     for size in problem_size:
    #         for index in range(1):
    #             path.append(f"./data/g05/g05_{size}.{index}")

    #     # ans, total_time, E = bm(schedule, M, T, path[0])
    #     # print(ans, total_time, E)

    #     for current_path in path:
    #         # output_file_name = current_path.split("/")[-1]
    #         with open(f"./result/new_result/{steps}_{M}_{T}_old_l.csv", "w", newline='') as output:
    #             writer = csv.writer(output)
    #             writer.writerow(['', 'cut', 'time', 'ans'])

    #             f_count = 0
    #             non_f_count = 0
    #             for i in range(bm_times):
    #                 ans, total_time, E, count, non_count = bm(schedule, M, T, current_path, G=False)
    #                 writer.writerow([i+1, E, total_time, ans])
    #                 f_count += count
    #                 non_f_count += non_count
    #             print(f_count)
    #             print(non_count)


        # for current_path in path:
        #     # output_file_name = current_path.split("/")[-1]
        #     with open(f"./result/new_result/{steps}_{M}_{T}_new_l.csv", "w", newline='') as output:
        #         writer = csv.writer(output)
        #         writer.writerow(['', 'cut', 'time', 'ans'])

        #         for i in range(bm_times):
        #             ans, total_time, E = bm_new(schedule, M, T, current_path, G=False)
        #             writer.writerow([i+1, E, total_time, ans])

        # for current_path in path:
        #     # output_file_name = current_path.split("/")[-1]
        #     with open(f"./result/new_result/{steps}_{M}_{T}_G_old_l.csv", "w", newline='') as output:
        #         writer = csv.writer(output)
        #         writer.writerow(['', 'cut', 'time', 'ans'])

        #         for i in range(bm_times):
        #             ans, total_time, E = bm(schedule, M, T, current_path, G=True)
        #             writer.writerow([i+1, E, total_time, ans])

        # for current_path in path:
        #     # output_file_name = current_path.split("/")[-1]
        #     with open(f"./result/new_result/{steps}_{M}_{T}_G_new_l.csv", "w", newline='') as output:
        #         writer = csv.writer(output)
        #         writer.writerow(['', 'cut', 'time', 'ans'])

        #         for i in range(bm_times):
        #             ans, total_time, E = bm_new(schedule, M, T, current_path, G=True)
        #             writer.writerow([i+1, E, total_time, ans])

    # for current_path in path:
    #     output_file_name = current_path.split("/")[-1]
    #     with open(f"./result/g05_F/{output_file_name}_2000_output.csv", "w", newline='') as output:
    #         writer = csv.writer(output)
    #         writer.writerow(['', 'cut', 'time', 'ans'])

    #         for i in range(bm_times):
    #             ans, total_time, E = bm(schedule, M, T, current_path, F=4)
    #             writer.writerow([i+1, E, total_time, ans])


    
    # gset
    # path = "./data/gset/G1.txt"
    # # for j in range(1,2):
    # with open(f"./result/new_result/gset/{steps}_{M}_{T}_G_new.csv", "w", newline='') as output:
    #     writer = csv.writer(output)
    #     writer.writerow(['', 'cut', 'time', 'ans'])

    #     for i in range(bm_times):
    #         ans, total_time, E = bm(schedule, M, T, path, G=True)
    #         writer.writerow([i+1, E, total_time, ans])

    # with open("./result/gset_F/G22_4000_output.csv", "w", newline='') as output:
    #     writer = csv.writer(output)
    #     writer.writerow(['', 'cut', 'time', 'ans'])

    #     for i in range(bm_times):
    #         ans, total_time, E = bm(schedule, M, T, path, F=4)
    #         writer.writerow([i+1, E, total_time, ans])



    # plt.title("Gamma")
    # plt.xlabel("Annealing step")
    # plt.ylabel("Gamma")
    # plt.plot(range(1000), schedule)
    # plt.show()

    # J_sche = [-0.5 * T * np.log(np.tanh(Gamma / (M * T))) for Gamma in schedule[0::4]]
    # J_sche += [-0.5 * T * np.log(np.tanh(Gamma / (M * T))) for Gamma in schedule[-1::-400]]
    # J_sche += [-0.5 * T * np.log(np.tanh(Gamma / (M * T))) for Gamma in schedule[0::4]]
    # J_sche += [-0.5 * T * np.log(np.tanh(Gamma / (M * T))) for Gamma in schedule[-1::-400]]
    # J_sche += [-0.5 * T * np.log(np.tanh(Gamma / (M * T))) for Gamma in schedule[0::4]]
    # J_sche += [-0.5 * T * np.log(np.tanh(Gamma / (M * T))) for Gamma in schedule[-1::-400]]
    # J_sche += [-0.5 * T * np.log(np.tanh(Gamma / (M * T))) for Gamma in schedule[0::4]]

    # plt.title("J plus")
    # plt.xlabel("Annealing step")
    # plt.ylabel("J plus")
    # plt.plot(range(len(J_sche)), J_sche)
    # plt.show()
