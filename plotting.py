import matplotlib.pyplot as plt
import numpy as np
import csv

path = []
solution = []
with open("./data/g05/solution.txt", "r", newline='') as solution_file:
    lines = solution_file.readlines()
    for line in lines:
        solution.append(int(line.split()[1]))

# Accuracy

# accuracy = np.zeros(1001)

# for index in range(1):
#     with open(f"./result/new_result/400_20_0.03_old.csv", "r", newline='') as output:
#         rows = csv.reader(output)

#         for row in list(rows)[1:]:
#             r = float(row[1]) / solution[index+20]
#             for i in range(1000,-1,-1):
#                 if r >= i / 1000:
#                     accuracy[i] += 1
#                     break


# plt.title("Steps = 400")
# plt.xlabel("Solution accuracy")
# plt.ylabel("Frequency")
# plt.xticks([i/10 for i in range(980,1001)], fontsize = 6)
# # plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 900, 1000])
# plt.bar([i/10 for i in range(980,1001)], accuracy[980:], width = 0.06)
# plt.show()



# Time

# problem_size = np.array([60, 80, 100])
steps = np.array([200, 400, 600, 800, 1000])

# x = problem_size.repeat([1000, 1000, 1000])
x = steps.repeat([100, 100, 100, 100, 100])
time = []
average_time = []
# for size in problem_size:
for step in steps:
    total_time = 0
    for index in range(1):
        # with open(f"./result/g05_G/g05_{size}.{index}_output.csv", "r", newline="") as output:
        with open(f"./result/new_result/{step}_20_0.03_old.csv", "r", newline="") as output:
            rows = csv.reader(output)
            for row in list(rows)[1:]:
                time.append(float(row[2]))
                total_time += float(row[2])

    average_time.append(total_time / 100)

# m = np.polyfit(problem_size, average_time, 2)
m = np.polyfit(steps, average_time, 1)

time_2 = []
average_time_2 = []
# for size in problem_size:
for step in steps:
    total_time_2 = 0
    for index in range(1):
        # with open(f"./result/g05_G/g05_{size}.{index}_output.csv", "r", newline="") as output:
        with open(f"./result/new_result/{step}_20_0.03_new.csv", "r", newline="") as output:
            rows = csv.reader(output)
            for row in list(rows)[1:]:
                time_2.append(float(row[2]))
                total_time_2 += float(row[2])

    average_time_2.append(total_time_2 / 100)

# m = np.polyfit(problem_size, average_time, 2)
m_2 = np.polyfit(steps, average_time_2, 1)

plt.title("Time")
# plt.xlabel("Problem size")
plt.xlabel("steps")
plt.ylabel("time (s)")
# plt.xticks([60, 80, 100])
plt.xticks([200, 400, 600, 800, 1000])
plt.scatter(x, time, s=10)
plt.scatter(x, time_2, s=10)
# plt.scatter(problem_size, average_time, s=30)
plt.scatter(steps, average_time, s=30)
plt.scatter(steps, average_time_2, s=30)
# plt.plot(np.array(range(55, 105)), m[0]*np.multiply(np.array(range(55, 105)), np.array(range(55, 105))) + m[1]*np.array(range(55, 105)) + m[2], color="orange")
plt.plot(np.array(range(90, 1010)), m[0]*np.array(range(90, 1010)) + m[1], color="orange")
plt.plot(np.array(range(90, 1010)), m_2[0]*np.array(range(90, 1010)) + m_2[1], color="orange")
plt.show()

# M = np.array([5, 10, 20, 30])

# x = M.repeat([100, 100, 100, 100])
# time = []
# average_time = []
# for m in M:
#     with open(f"./result/g05/{m}/g05_100.0_output.csv", "r", newline="") as output:
#         total_time = 0
#         rows = csv.reader(output)
#         for row in list(rows)[1:]:
#             total_time += float(row[2])
#             time.append(float(row[2]))
#     average_time.append(total_time / 100)

# m, k = np.polyfit(M, average_time, 1)

# plt.title("Time")
# plt.xlabel("M")
# plt.ylabel("time (s)")
# plt.xticks([5, 10, 20, 30])
# plt.scatter(x, time, s=10)
# plt.scatter(M, average_time, s=30)
# plt.plot(M, m*M + k, color="orange")
# plt.show()



# Schedule

# M = 20
# T = 0.03

# steps = 1000
# Gamma0 = 10
# Gamma1 = 1e-8
# decay_rate = Gamma1 - Gamma0 / steps
# schedule = [Gamma0 + decay_rate*i for i in range(steps)]
# Jp = [-0.5 * T * np.log(np.tanh(Gamma / (M * T))) for Gamma in schedule]
# plt.title("J plus")
# plt.xlabel("Annealing step")
# plt.ylabel("J plus")
# # plt.plot(range(steps), schedule)
# plt.plot(range(steps), Jp)
# plt.show()

# plt.title("Flip count")
# plt.bar(["flip", "no flip"], [0.78375418582, 0.21624581417])
# plt.show()