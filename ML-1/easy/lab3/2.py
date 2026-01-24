"""
Измените один из весов нейросети так, чтобы ошибка (error) была меньше 0.001.
Решите это методом проб и ошибок.
"""

import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_predict, cur_predict):
    return (true_predict - cur_predict) ** 2


true_predict = 50
weights = np.array([0.2532, 0.3])
predict = neural_networks(np.array([150, 40]), weights)
print(predict)
print(get_error(true_predict, predict))
