import numpy as np
from random import random


def random_bit():
    """
    :return: Returns either 1 or -1 with probability 0.5.
    """
    return 1 if random() >= 0.5 else -1


def flip_bit(bit):
    """
    Returns +/- 2 from the original bit. -1 if 1, 1 if -1.
    :param bit: Either -1 or 1
    :return: The flipped bit
    """
    return 1 if bit < 0 else -1


def random_matrix(m, n, zero_diagonals=False, sym=False):
    """
    Creates a matrix of size (m * n) populated with a random distribution of
    -1 and 1's.
    :param m: The number of rows of the matrix
    :param n: The number of columns of the matrix
    :param zero_diagonals: Populates the matrix with diagonal zeroes
    :param sym: Ensures that the matrix is symmetrical along diagonal axis.
    Default is asymmetric.
    :return: The random matrix
    """
    matrix = np.array([[random_bit() for _ in range(n)] for _ in range(m)])
    if sym:
        matrix = np.maximum(matrix, matrix.transpose())
    if zero_diagonals:
        np.fill_diagonal(matrix, 0)
    return matrix
