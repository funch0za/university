"""
Замените списки с фиксированными весами на массивы numpy,
причём с генерацией случайных значений в них.
"""

import numpy as np


def neuralNetwork(inp, weights):
    prediction_h1 = inp.dot(weights[0])
    prediction_h2 = prediction_h1.dot(weights[1])
    prediction_out = prediction_h2.dot(weights[2])
    return prediction_out


def get_random_arr(size):
    return np.random.rand(size)


inp = np.array([23, 45])

weight_h_1 = get_random_arr(2)
weight_h_2 = get_random_arr(2)
weight_h_3 = get_random_arr(2)

weight_out_1 = get_random_arr(2)
weight_out_2 = get_random_arr(2)
weight_out_3 = get_random_arr(2)

weights_h = np.array([weight_h_1, weight_h_2, weight_h_3]).T
weights_out = np.array([weight_out_1, weight_out_2, weight_out_3]).T

weights_1 = np.array([weight_h_1, weight_h_2]).T
weights_2 = np.array([weight_h_3, weight_out_1]).T
weights_3 = np.array([weight_out_2, weight_out_3]).T

weights = [weights_1, weights_2, weights_3]

print(neuralNetwork(inp, weights))
