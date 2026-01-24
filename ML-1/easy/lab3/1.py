"""
Напишите по памяти код из урока "Оценка ошибки".
"""

import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


prediction = neural_networks(np.array([150, 40]), [0.2, 0.3])
print(prediction)

true_prediction = 50

print(true_prediction - prediction)
print(get_error(true_prediction, prediction))
