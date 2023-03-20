steps = 1000

schedule = [100 * (1 - 0.01) ** i for i in range(steps)]

print(schedule)