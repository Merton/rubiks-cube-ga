import numpy as np
from rubiks_operations import Cube, rand_moves

# === Global GA Config === #
max_generations = 200
population_size = 100
mutation_rate = 0.2
crossover_rate = 0.8
num_moves = 50
demes_size = 20
# demes_size = -1

elitism = False
top_percent_thres = 0.1  # Keep top, bring new pop in

# Create the starting population
starting_cube = Cube()
starting_cube.shuffle(rigor=5)

print("Cube:")
print(starting_cube)

starting_moveset = np.array([rand_moves(num_moves) for n in range(population_size)])

"""
=== GA Features ===
The ga_types dictionary is used to store the GA config for each combination of features.
The key represents the label of the GA for maintenance and plot labels.
The value is a dictionary of settings for the GA to enable features like 
mutation, tournament selection and microbial crossover.

The f_min, f_max and f_avg can be accessed after the ga has been ran to view results and plot.
"""
ga_types = {
    'Microbial': {
        'has_mutation': False,
        'has_tournament': False,
        'has_microbial_co': True,
        'has_k_co': False,
        'population': starting_moveset,
        'fitness': [0],
        'f_min': [0],
        'f_max': [0],
        'f_avg': [0],
        'plot_color': 'b-'
    },
}