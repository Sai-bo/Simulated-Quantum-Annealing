import matplotlib.pyplot as plt
import numpy as np
import csv

path = []
solution = []
with open("./data/g05/solution.txt", "r", newline='') as solution_file:
    lines = solution_file.readlines()
    for line in lines:
        solution.append(int(line.split()[1]))

##########################################################################################################

# Accuracy

accuracy = np.zeros(1001)
size = 60
index = 0
steps = 100000
factor = 100000
mode = "exp"


with open(f"result\MA_result\g05_{size}.{index}_{steps}_{factor}_{mode}", "r", newline='') as output:
    rows = csv.reader(output)

    for row in list(rows)[1:]:
        r = float(row[1]) / solution[10 * ((size // 20) - 3) + index]
        for i in range(1000,-1,-1):
            if r >= i / 1000:
                accuracy[i] += 1
                break


plt.title(f"Steps = {steps}")
plt.xlabel("Solution accuracy")
plt.ylabel("Frequency")
plt.xticks([i for i in range(90,101)], fontsize = 6)
# plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 900, 1000])
plt.bar([i/10 for i in range(900,1001)], accuracy[900:], width = 0.1)
plt.show()

##########################################################################################################

# Time

# problem_size = np.array([60, 80, 100])
steps = np.array([300, 3000, 10000, 30000])

# x = problem_size.repeat([1000, 1000, 1000])
x = steps.repeat([100, 100, 100, 100])
time = []
average_time = []
# for size in problem_size:
for step in steps:
    total_time = 0
    for index in range(1):
        # with open(f"./result/g05_G/g05_{size}.{index}_output.csv", "r", newline="") as output:
        with open(f"result\MA_result\g05_60.0_{step}_3_linear", "r", newline="") as output:
            rows = csv.reader(output)
            for row in list(rows)[1:]:
                time.append(float(row[2]))
                total_time += float(row[2])

    average_time.append(total_time / 100)

# m = np.polyfit(problem_size, average_time, 2)
m = np.polyfit(steps, average_time, 1)

plt.title("Time")
# plt.xlabel("Problem size")
plt.xlabel("steps")
plt.ylabel("time (s)")
# plt.xticks([60, 80, 100])
plt.xticks([300, 3000, 10000, 30000])
plt.scatter(x, time, s=10)
# plt.scatter(problem_size, average_time, s=30)
plt.scatter(steps, average_time, s=30)
# plt.plot(np.array(range(55, 105)), m[0]*np.multiply(np.array(range(55, 105)), np.array(range(55, 105))) + m[1]*np.array(range(55, 105)) + m[2], color="orange")
plt.plot(np.array(range(200, 30100)), m[0]*np.array(range(200, 30100)) + m[1], color="orange")
plt.show()

##########################################################################################################

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