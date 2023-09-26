import matplotlib.pyplot as plt
import csv

import matplotlib.ticker
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter


def read_from_file(filename):
    x = []
    y = []
    count = 0
    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            count += 1
            y.append(int(float(row[0])))
            x.append(count)
    return x, y


x_max_fitness, y_max_fitness = read_from_file('data/max_fitness_helicopter.csv')
x_min_fitness, y_min_fitness = read_from_file('data/min_fitness_helicopter.csv')
x_avg_fitness, y_avg_fitness = read_from_file('data/avg_fitness_helicopter.csv')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(x_max_fitness, y_max_fitness)
ax2.plot(x_min_fitness, y_min_fitness)
ax3.plot(x_avg_fitness, y_avg_fitness)
plt.show()