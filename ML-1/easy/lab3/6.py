"""
Измените количество итераций в цикле (например, увеличьте до 100).
Как изменение числа итераций влияет на точность предсказаний?
Попробуйте найти оптимальное количество итераций для данной задачи.
"""

import numpy as np


def neural_networks(inp, weight):
    return inp * weight


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def gradient(inp, weight, true_prediction, n):
    for i in range(n):
        prediction = neural_networks(inp, weight)
        error = get_error(true_prediction, prediction)
        print(
            "Prediction: %.10f, Weight: %.5f, Error: %.20f"
            % (prediction, weight, error)
        )
        delta = (prediction - true_prediction) * inp
        weight = weight - delta


# достаточно 15
gradient(0.9, 0.2, 0.5, 15)
print()
# нужно больше 10**5
gradient(0.009, 0.0001, 0.9, 2 * 10**5)

"""
Влияет на точность результата. Чем больше эпох, тем меньше ошибка.
"""
