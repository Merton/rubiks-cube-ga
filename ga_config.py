import numpy as np
from rubiks_operations import Cube, rand_moves

# === Global GA Config === #
max_generations = 500
population_size = 100
mutation_rate = 0.2
crossover_rate = 0.85
num_moves = 10
top_percent_thres = 0.1
demes_size = population_size

# Create the starting population
starting_cube = Cube()
starting_cube.shuffle(rigor=num_moves)

# print("Shuffling cube")
# starting_cube.shuffle()
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
        'has_mutation': True,
        'has_tournament': False,
        'has_microbial_co': True,
        'population': starting_moveset,
        'fitness': [0],
        'f_min': [0],
        'f_max': [0],
        'f_avg': [0],
        'plot_color': 'b-'
    },
}