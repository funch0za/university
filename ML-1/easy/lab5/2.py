"""
Измените скорость обучения (learning_rate).
Попробуйте более высокие или более низкие значения (например, 0.000001 или 0.0001).
Как это влияет на скорость сходимости и точность предсказания?
"""

import numpy as np

weights = np.array([0.2, 0.3])


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def gradient(inp, true_predictions, weights, learning_rate, epochs):
    for i in range(epochs):
        error = 0
        delta = np.zeros_like(weights)
        for j in range(len(inp)):
            current_inp = inp[j]
            true_prediction = true_predictions[j]
            prediction = neural_networks(current_inp, weights)
            error += get_error(true_prediction, prediction)
            print(
                "Prediction: %.10f, True_prediction: %.10f, Weights: %s"
                % (prediction, true_prediction, weights)
            )
            delta += (prediction - true_prediction) * current_inp * learning_rate
        weights -= delta / len(inp)
        print("Errors: %.10f" % error)
        print("-------------------")
    return weights


def calc_prob(person_h, person_w):
    return neural_networks(np.array([person_h, person_w]), weights)


inp = np.array(
    [
        [150, 40],
        [140, 35],
        [155, 45],
        [185, 95],
        [145, 40],
        [195, 100],
        [180, 95],
        [170, 80],
        [160, 90],
    ]
)

true_predictions = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])

learning_rate = 0.00001
# learning_rate = 0.000001 # ошибка становится больше так как нужно больше эпох для обучения
# learning_rate = 0.0001 # ошибки при рассчетах
epochs = 500

# обучение
weights = gradient(inp, true_predictions, weights, learning_rate, epochs)

print(calc_prob(150, 45))
print(calc_prob(170, 85))
