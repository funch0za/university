"""
Увеличьте количество итераций цикла (например, до 100).
Как это влияет на точность предсказаний?
Сколько итераций необходимо для достижения близкой к идеальной точности предсказаний?
"""

import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def gradient(inp, weights, true_prediction, count_iters, learning_rate):
    for i in range(count_iters):
        prediction = neural_networks(inp, weights)
        error = get_error(true_prediction, prediction)
        print(
            "Prediction: %.10f, Weights: %s, Error: %.20f"
            % (prediction, weights, error)
        )
        delta = (prediction - true_prediction) * inp * learning_rate
        delta[0] = 0
        weights = weights - delta


inp = np.array([150, 40])
weights = np.array([0.2, 0.3])
true_prediction = 1

count_iters = 30

print("1")
learning_rate = 0.00001
count_iters = 1000
gradient(inp, weights, true_prediction, count_iters, learning_rate)

"""
Как это влияет на точность предсказания? 
При таком большом количестве итераций нейросеть смогла обучиться.
Даже учитывая такую маленькую скорость обучения.

Когда нейросеть перестает улучшать предсказание?
Когда ошибка становится равной 0.
А также если поставить слишком большую
"""
