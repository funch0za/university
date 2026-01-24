"""
Напишите по памяти код из урока "Градиентный спуск с несколькими выходами".
"""

import numpy as np


def neural_networks(inp, weights):
    return inp * weights


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def gradient(inp, weights, true_predictions, count_iters, learning_rate):
    for i in range(count_iters):
        prediction = neural_networks(inp, weights)
        error = get_error(true_predictions, prediction)
        print("Prediction: %s, Weights: %s, Error: %s" % (prediction, weights, error))
        delta = (prediction - true_predictions) * inp * learning_rate
        weights = weights - delta


inp = 200
weights = np.array([0.2, 0.3])
true_predictions = np.array([50, 120])
learning_rate = 0.00001
count_iters = 30

gradient(inp, weights, true_predictions, count_iters, learning_rate)

"""
Значения inp, которые слишком велики, могут сделать обучение более сложным, 
так как большие входные значения увеличивают величину шага обновления весов, 
что может привести к скачкам весов и затруднению сходимости. 
Слишком маленькие значения inp могут замедлить обучение, 
потому что величина обновления весов становится очень малой.

Большие значения inp - потенциальная нестабильность.
Нужно лучше адаптировать шаг обучения.
"""
