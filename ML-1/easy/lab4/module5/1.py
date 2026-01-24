"""
Напишите по памяти код из урока "Обучение на нескольких наборах данных".
"""

import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def gradient(inp, weights, true_predictions, count_iters, learning_rate):
    for i in range(count_iters):
        for j in range(len(inp)):
            cur_inp = inp[j]
            cur_predict = true_predictions[j]
            prediction = neural_networks(cur_inp, weights)
            error = get_error(cur_predict, prediction)
            print("Prediction: ", prediction)
            print("Weights: ", weights)
            print("Error: ", error)
            print("-------------------")
            delta = (prediction - cur_predict) * cur_inp * learning_rate
            weights = weights - delta


inp = np.array([[150, 40], [170, 80], [160, 90]])
true_predictions = np.array([50, 120, 140])
weights = np.array([0.2, 0.3])
count_iters = 50
learning_rate = 0.00001
gradient(inp, weights, true_predictions, count_iters, learning_rate)
