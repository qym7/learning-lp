"""
Module contains some useful mathematical functions that are used in several other modules.
"""


import numpy as np
import random


def get_weights_for_convex_comb(nb):
    """
    Returns weights for a convex combination of nb elements, of which none is favoured.

    Arguments
    ---------
    nb : int

    Returns
    -------
    weights : numpy.array
    """
    aux = (nb + 1) * [None]
    aux[0] = 0
    aux[nb] = 1
    for i in range(nb - 1):
        aux[i + 1] = random.random()
    aux = np.array(aux)
    aux.sort()

    weights = nb * [None]
    for i in range(nb):
        weights[i] = aux[i + 1] - aux[i]

    return np.array(weights)


def convex_comb(weights, points):
    """
    Computes convex combination of points with given weights and returns the result.

    Arguments
    ---------
    weights : numpy.array
    points : numpy.array

    Returns
    -------
    conv_comb : numpy.array
    """
    points = points.copy()
    conv_comb = np.dot(np.transpose(points), weights)
    return conv_comb
