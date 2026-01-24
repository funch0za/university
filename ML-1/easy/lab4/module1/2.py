"""
Измените значение learning_rate на 0.01, 0.1 и 0.0001.
Как это влияет на скорость обучения?
Попробуйте найти оптимальное значение learning_rate, которое обеспечивает быстрое и точное обучение.
"""

import numpy as np


def neural_networks(inp, weight):
    return inp * weight


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def gradient(inp, weight, true_prediction, count_iters, learning_rate):
    for i in range(count_iters):
        prediction = neural_networks(inp, weight)
        error = get_error(true_prediction, prediction)
        print(
            "Prediction: %.10f, Weight: %.5f, Error: %.20f"
            % (prediction, weight, error)
        )
        delta = (prediction - true_prediction) * inp * learning_rate
        weight = weight - delta


print("1")
gradient(30, 0.2, 70, 20, 0.01)  # обучение ломается
print("2")
gradient(30, 0.2, 70, 20, 0.1)  # обучение ломается
print("3")
gradient(30, 0.2, 70, 20, 0.0001)  # слишком медленное обучение
print("4")
gradient(30, 0.2, 70, 20, 0.001)  # подходящий коэффициент
