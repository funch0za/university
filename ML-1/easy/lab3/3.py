"""
Решите предыдущее задание, но уже с использованием кода и цикла в нём,
последовательно меняя вес на заданный шаг на каждой итерации.
Выполнять цикл надо до тех пор, пока ошибка не станет меньше 0.001.
Выведите получившиеся веса.
"""

import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_predict, cur_predict):
    return (true_predict - cur_predict) ** 2


true_predict = 50
weights = np.array([0.2, 0.3])
inp = np.array([150, 40])


ideal_error = 0.001
while get_error(true_predict, neural_networks(inp, weights)) > ideal_error:
    weights[0] += 0.0001

print(weights)
print(get_error(true_predict, neural_networks(inp, weights)))
