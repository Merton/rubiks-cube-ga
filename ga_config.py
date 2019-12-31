from matrix_helpers import random_matrix
import numpy as np

# === Global GA Config === #
generations = 100
population_size = 100
num_friends   = 20
mutation_rate = 0.05

demes_size = population_size / 5

# Get the relationship matrix
relationships = random_matrix(num_friends, num_friends, True, sym=True)

# Create the starting population
starting_population = random_matrix(population_size, num_friends)


"""
=== GA Features ===
The ga_types dictionary is used to store the GA config for each combination of features.
The key represents the label of the GA for maintenance and plot labels.
The value is a dictionary of settings for the GA to enable features like 
mutation, tournament selection and microbial crossover.

The f_min, f_max and f_avg can be accessed after the ga has been ran to view results and plot.
"""
ga_types = {
    'Mutation Only': {
        'has_mutation': True,
        'has_tournament': False,
        'has_microbial_co': False,
        'population': starting_population.copy(),
        'f_min': [],
        'f_max': [],
        'f_avg': [],
        'plot_color': 'b-'
    },
    # 'Tournament Only': {
    #     'has_mutation': False,
    #     'has_tournament': True,
    #     'has_microbial_co': False,
    #     'population': starting_population.copy(),
    #     'f_min': [],
    #     'f_max': [],
    #     'f_avg': [],
    #     'plot_color': 'r-'
    # },
    'Microbial Only': {
        'has_mutation': False,
        'has_tournament': False,
        'has_microbial_co': True,
        'population': starting_population.copy(),
        'f_min': [],
        'f_max': [],
        'f_avg': [],
        'plot_color': 'k-'
    },
    'Mutation + Tournament': {
        'has_mutation': True,
        'has_tournament': True,
        'has_microbial_co': False,
        'population': starting_population.copy(),
        'f_min': [],
        'f_max': [],
        'f_avg': [],
        'plot_color': 'y-'
    },
    'Mutation + Microbial CO': {
        'has_mutation': True,
        'has_tournament': False,
        'has_microbial_co': True,
        'population': starting_population.copy(),
        'f_min': [],
        'f_max': [],
        'f_avg': [],
        'plot_color': 'm-'
    },
    'Microbial + Tournament': {
        'has_mutation': False,
        'has_tournament': True,
        'has_microbial_co': True,
        'population': starting_population.copy(),
        'f_min': [],
        'f_max': [],
        'f_avg': [],
        'plot_color': 'c-'
    },
    # Uncomment to enable all features GA
    # 'All Features': {
    #     'has_mutation': True,
    #     'has_tournament': True,
    #     'has_microbial_co': True,
    #     'population': starting_population.copy(),
    #     'f_min': [],
    #     'f_max': [],
    #     'f_avg': [],
    #     'plot_color': 'r-'
    # },
}