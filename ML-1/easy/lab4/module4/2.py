"""
Измените true_predictions на другие значения (например, [[70, 90]] или [[30, 110]]).
Запустите код и определите, как это влияет на обучение нейросети.
Какие значения true_predictions делают обучение более сложным, а какие более простым?
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
        print("Prediction: ", prediction)
        print("Weights: ", weights)
        print("Error: ", error)
        delta = (prediction - true_predictions) * inp * learning_rate
        weights = weights - delta


inp = np.array([150, 40])
weights = np.array(
    [[0.2, 0.3], [0.5, 0.7]]
).T  # Транспонируем данную весовую матрицу векторов
learning_rate = 0.00001
count_iters = 50

true_predictions = np.array([1000, 2000])
gradient(inp, weights, true_predictions, count_iters, learning_rate)
print()
true_predictions = np.array([70, 90])
# gradient(inp, weights, true_predictions, count_iters, learning_rate)
true_predictions = np.array([30, 110])
# gradient(inp, weights, true_predictions, count_iters, learning_rate)
true_predictions = np.array([100, 101])
# gradient(inp, weights, true_predictions, count_iters, learning_rate)

"""
1) Большое расхождение с предсказаниями
    Большие отличия от входных значений приводят к нестабильности.
2) Малые значения увеличивают величину ошибки. Это требует уменьшение скорости обучения
3) Разброс значений приводит к более сложному обучению.
"""
