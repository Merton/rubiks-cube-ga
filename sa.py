from scipy.constants import k
from ga_config import *
from copy import deepcopy
from rubiks_operations import rand_move
from random import random, randint
from ga import plot_gas
import matplotlib.pyplot as plt

t = 2000
start_t = t
a = 0.05

fit = starting_cube.fitness()
print("Starting fitness: {}".format(fit))

moves = rand_moves(num_moves)

cube = deepcopy(starting_cube)
fit_max = []
fit_cur = []
temps = []
N = 100
i = 0
prev = ''
# moves = []
max_fitness = 0
while t > 1:
    for n in range(N):
        if len(moves) > 0:
            anneal_i = randint(0, len(moves)-1)
        else:
            anneal_i = 0
        if anneal_i > 0:
            prev = moves[anneal_i-1]
        else:
            prev = ''

        if anneal_i < len(moves)-1:
            next = moves[anneal_i+1]
        else:
            next = ''

        cube_current = deepcopy(cube)
        current_fit = cube_current.perform_moves(moves)

        cube_new = deepcopy(cube)
        new_move = rand_move(prev, next)
        try_moves = np.copy(moves)
        try_moves[anneal_i] = new_move
        new_fit = cube_new.perform_moves(try_moves)

        dE = new_fit - current_fit
        if dE > 0:
            moves[anneal_i] = new_move
        elif dE <= 0:
            P_E = np.exp(dE / t)
            rand = random()
            if P_E < rand:
                moves[anneal_i] = new_move

        if new_fit > max_fitness:
            max_fitness = new_fit

    print("Max fitness: {}, Current Fitness: {}".format(max_fitness, new_fit))
    print(moves)
    print("Temp: {} Move length: {}".format(t, len(moves)))
    fit_max.append(max_fitness)
    fit_cur.append(new_fit)
    t -= t*a
    temps.append(t)
    i += 1

plt.plot(temps, fit_max, 'r-', label="SA Max")
# plt.plot(temps, fit_cur, 'b-', label="SA Current")

plt.xlabel('Temperature')
plt.ylabel('Fitness')
plt.xlim(2000, 0)
plt.legend()
plt.show()
