"""
Измените значение true_prediction на другое значение (например, 0.8 или 0.5) и запустите код снова.
Как изменение желаемого выходного значения влияет на обучение нейросети?
Объясните результаты.
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


gradient(0.9, 0.2, 0.18)
print()
gradient(0.9, 0.2, 0.5)

"""
Влияет на скорость обучения т.к. влияет на величину ошибки.
"""
