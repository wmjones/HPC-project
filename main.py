import numpy as np
import tensorflow as tf


def generate_data(m, d):
    # generate an array for a grid of m equidistant points per dimension for the hypercube [0, 4pi]^d
    data = np.zeros((m**d, d))
    for i in range(0, d):
        data[:, i] = np.tile(np.repeat(np.linspace(0, 4*np.pi, m), m**(d-(i+1))), m**i)
    return(data)


def generate_label(data):
    return(np.apply_along_axis(lambda x: sum(np.sin(x)), 1, data))


data = generate_data(10, 3)
label = generate_label(data)
