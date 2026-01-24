"""
Добавьте еще один скрытый слой к нейросети.
Создайте третий набор весов weights_h_3 = [0.6, 0.2] и weights_out_3 = [0.7, 0.4].
Добавьте его в переменные weights_h и weights_out.
Измените функцию neural_network так, чтобы она работала с этим третьим слоем.
"""

import numpy as np


def neuralNetwork(inp, weights):
    prediction_h1 = inp.dot(weights[0])
    prediction_h2 = prediction_h1.dot(weights[1])
    prediction_out = prediction_h2.dot(weights[2])
    return prediction_out


inp = np.array([23, 45])

weight_h_1 = [0.4, 0.1]
weight_h_2 = [0.3, 0.2]
weight_h_3 = [0.6, 0.2]

weight_out_1 = [0.4, 0.1]
weight_out_2 = [0.3, 0.1]
weight_out_3 = [0.7, 0.4]

weights_h = np.array([weight_h_1, weight_h_2, weight_h_3]).T
weights_out = np.array([weight_out_1, weight_out_2, weight_out_3]).T

weights_1 = np.array([weight_h_1, weight_h_2]).T
weights_2 = np.array([weight_h_3, weight_out_1]).T
weights_3 = np.array([weight_out_2, weight_out_3]).T

weights = [weights_1, weights_2, weights_3]

print(neuralNetwork(inp, weights))
