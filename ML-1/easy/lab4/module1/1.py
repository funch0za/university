"""
Напишите по памяти код из урока "Решение проблемы с расхождением".
"""

import numpy as np


def neural_networks(inp, weight):
    return inp * weight


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def gradient(inp, weight, true_prediction, n):
    learning_rate = 0.001
    for i in range(n):
        prediction = neural_networks(inp, weight)
        error = get_error(true_prediction, prediction)
        print(
            "Prediction: %.10f, Weight: %.5f, Error: %.20f"
            % (prediction, weight, error)
        )
        delta = (
            (prediction - true_prediction) * inp * learning_rate
        )  # умножаем нашу проивзодную на скорость обучения
        weight = weight - delta


gradient(30, 0.2, 70, 10)
