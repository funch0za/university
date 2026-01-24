"""
Измените начальное значение веса и входных данных.
Запустите код несколько раз с разными начальными значениями и выведите результаты.
Как начальные значения влияют на обучение нейросети?
"""

import numpy as np


def neural_networks(inp, weight):
    return inp * weight


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def gradient(inp, weight, true_prediction):
    for i in range(10):
        prediction = neural_networks(inp, weight)
        error = get_error(true_prediction, prediction)
        print(
            "Prediction: %.10f, Weight: %.5f, Error: %.20f"
            % (prediction, weight, error)
        )
        delta = (prediction - true_prediction) * inp
        weight = weight - delta


true_prediction = 0.4

gradient(0.9, 0.2, true_prediction)
print()
gradient(0.2, 0.001, true_prediction)
print()
gradient(0.9999, 2, true_prediction)
print()
gradient(0.9999, 20, true_prediction)
print()
gradient(0.00001, 200, true_prediction)
print()
gradient(0.00001, 0.2, true_prediction)

"""
Большие веса - нестабильность.
Маленькие веса - медленное обучение.
"""
