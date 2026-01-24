"""
Увеличьте количество итераций цикла (например, до 100).
Как это влияет на точность предсказаний?
Сколько итераций необходимо для достижения близкой к идеальной точности предсказаний?
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


# слишком медленное обучение
# поэтому нужно больше итераций
gradient(30, 0.2, 70, 100, 0.0001)
print()
# подходящий коэффициент
# можно использовать меньше итераций
gradient(30, 0.2, 70, 20, 0.001)
# чем меньше learning_rate, тем больше нужно итераций
