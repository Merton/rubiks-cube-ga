import sys
from random import random, randint
import matplotlib.pyplot as plt
from copy import deepcopy
from ga_config import *
from rubiks_operations import rand_move


# To configure the different starting values and GAs features, use the ga_config.py file.
def random_pop_selection(demes_size=-1):
    """
    Selects 2 random phenotypes from the population
    :param demes_size: The deme size for selecting the population within a cyclic loop
    :return: The two selected indexes of the population
    """
    g1_i = randint(0, population_size - 1)
    if demes_size > 0:
        g2_i = (g1_i + 1 + randint(0, demes_size)) % population_size
    else:
        g2_i = randint(0, population_size - 1)
    return g1_i, g2_i


def mutate(genotype):
    """
    Randomly flips one gene in the genotype with probability of mutation_rate
    :param genotype: The list of bits representing the genotype
    :return: The possibly mutated genotype
    """
    if random() > 1 - mutation_rate:
        mutation_index = randint(0, len(genotype) - 1)
        prev = ''
        if mutation_index > 0:
            prev = genotype[mutation_index-1]
        genotype[mutation_index] = rand_move(prev)

    return genotype


def fitness(genotype):
    """
    Calculated the fitness of the genotype. Creates a FxF matrix, multiplies it by the relationships then sums
    along to result the fitness.
    :param genotype: The genes to be measured
    :return: The fitness value (+/-)
    """
    current_state = deepcopy(starting_cube.state)
    starting_cube.perform_moves(genotype)

    fit = starting_cube.fitness()
    starting_cube.state = current_state

    return fit


def tournament(pop):
    """
    Select two contestants at random and determine
    the highest fitness of them. The loser gets replaced by the winner.
    :param pop: The population that the contestants get selected from.
    :return: The new population
    """
    g1_i, g2_i = random_pop_selection(demes_size)

    g1 = pop[g1_i]
    g2 = pop[g2_i]

    if fitness(g1) >= fitness(g2):
        pop[g2_i] = g1
    else:
        pop[g1_i] = g2

    return pop


def microbial_co(pop):
    """
    Implements microbial crossover. Selects two random genotypes and determines greatest fitness.
    The loser with some chance Pc, gets crossed over bit by bit with the winner. Each bit mutates with
    some probability Pm.
    :param pop: The population that the samples get taken from
    :return: The updated population
    """
    g1_i, g2_i = random_pop_selection(demes_size)

    g1 = pop[g1_i]
    g2 = pop[g2_i]

    Pc = crossover_rate
    Pm = mutation_rate

    g1_fit = fitness(g1)
    g2_fit = fitness(g2)
    if g1_fit >= g2_fit:
        winner = g1
        loser = g2
        loser_i = g2_i
    else:
        winner = g2
        loser = g1
        loser_i = g1_i

    for i in range(len(winner)):
        if random() > 1 - Pc:
            loser[i] = winner[i]
        if random() > 1 - Pm:
            prev = ''
            if i > 0:
                prev = loser[i-1]

            loser[i] = rand_move(prev)

    pop[loser_i] = loser
    return pop


def ga(config):
    """
    Calculates the fitness of the population, then applies the relevant GA features
    determined by the config dictionary.
    :param config: The configuration settings for the GA
    :return: The updated config for the GA
    """
    print("\n")
    if config['has_mutation']:
        print("\tApplying mutation")
        config['population'] = np.apply_along_axis(mutate, 1, config['population'])

    if config['has_tournament']:
        print("\tApplying tournament selection")
        for _ in range(population_size):
            config['population'] = tournament(config['population'])

    if config['has_microbial_co']:
        print("\tApplying microbial crossover")
        for _ in range(population_size):
            config['population'] = microbial_co(config['population'])

    print("\tCalculating fitness")
    pop_fitness = [fitness(genotype) for genotype in config['population']]
    config['fitness'] = pop_fitness

    f_min = np.amin(config['fitness'])
    f_max = np.amax(config['fitness'])
    f_avg = np.sum(config['fitness']) / population_size

    config['f_min'].append(f_min)
    config['f_max'].append(f_max)
    config['f_avg'].append(f_avg)


    print("\tIntroducing new population with best performers")
    best_split, new_split = int(population_size*top_percent_thres), int(population_size*(1-top_percent_thres))

    pop_fit_arr = sorted(zip(config['population'], pop_fitness), key=lambda x: x[1], reverse=True)

    best_performers = [n[0] for n in pop_fit_arr][:best_split]
    new_pop = np.array([rand_moves(num_moves) for _ in range(new_split)])

    config['population'] = np.concatenate((best_performers, new_pop))
    config['fitness'] = [fitness(genotype) for genotype in config['population']]

    return config


def run_gas(num_generations):
    """
    Runs all GAs in the ga_types dictionary, for n generations.
    :param num_generations: The number of generations to run
    :return:
    """
    config = ga_types['Microbial']
    initial_pop = deepcopy(config['population'])

    for i in range(num_generations):
        sys.stdout.write("\rRunning gen {0} of {1}".format(i+1, generations))
        sys.stdout.flush()

        for ga_type, config, in ga_types.items():
            ga(config)
            ga_types[ga_type] = config

            best_fitness = np.amax(config['fitness'])
            best_fitness_index = np.where(config['fitness'] == best_fitness)
            print("Best moves: ", config['population'][best_fitness_index][0])
            print("Best fitness: ", best_fitness)
            print("\n")
    print(f'\nMax (solved) fitness: 48')
    print()


def plot_gas():
    """
    Uses the populated lists for each GA type stored in the ga_types dictionary to
    plot them on the same figure. Plot label is inferred from the type name, and plot colour
    can be configured in ga_config.py.
    :return:
    """
    xs = range(generations)
    print_results()

    for type, values in ga_types.items():
        plt.plot(xs, values['f_max'], values['plot_color'], label=type+" Max")

    plt.legend()
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()


def print_results():
    for ga_name, config in ga_types.items():
        print(
            f'{ga_name: <25}: '
            f'best: {max(config["f_max"]): <3}, '
            f'worst: {min(config["f_min"]): <3}, '
            f'avg: {sum(config["f_avg"]) // generations: <4}'
        )


if __name__ == '__main__':
    run_gas(generations)
    plot_gas()
