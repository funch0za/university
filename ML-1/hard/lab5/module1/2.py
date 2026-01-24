"""
Попробуйте изменить количество нейронов в скрытом слое (hidden_size).
Как это влияет на сходимость обучения и точность предсказания?
"""

import numpy as np

DEBUG_MODE = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def relu(x):
    return (x < 0) * x


def relu_deriv(x):
    return x > 0


def linear(x):
    return x


def linear_deriv(x):
    return 1


def th(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def th_deriv(x):
    return 1 - (np.exp(4 * x) - 2 * np.exp(2 * x) + 1) / (
        np.exp(4 * x) + 2 * np.exp(2 * x) + 1
    )


def get_random_weights(input_size, output_size):
    np.random.seed(1)
    return np.random.uniform(size=(input_size, output_size))


def train(
    data,
    predict,
    weight_out,
    weight_hid,
    epochs,
    learning_rate,
    activation,
    activation_deriv,
):
    for epoch in range(epochs):
        layer_hid = activation(np.dot(data, weight_hid))

        layer_out = activation(np.dot(layer_hid, weight_out))
        error = (layer_out - predict) ** 2

        layer_out_delta = (layer_out - predict) * activation_deriv(layer_out)
        layer_hidden_delta = layer_out_delta.dot(weight_out.T) * activation_deriv(
            layer_hid
        )
        # подгоняем веса
        weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
        weight_hid -= learning_rate * data.T.dot(layer_hidden_delta)
        # каждую 1000 эпоху будем выводить ошибку
        if epoch % 1000 == 0:
            error = np.mean(error)
            if DEBUG_MODE:
                print(f"Epoch: {epoch}, Error: {error}")
    return weight_hid, weight_out


def run(activation, activation_deriv, hidden_size):
    EPOCHS = 100000
    LEARNING_RATE = 0.1
    HIDDEN_SIZE = hidden_size

    # реализуем чтото вроде "исключающего или" XOR
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # входные данные
    y = np.array([[0], [1], [1], [0]])  # ожидаемый прогноз

    # задаем параметры нейронной сети
    input_size = len(x[0])
    output_size = len(y[0])

    # фиксируем генератор случайных чисел
    weight_hid = get_random_weights(input_size, HIDDEN_SIZE)
    weight_out = get_random_weights(HIDDEN_SIZE, output_size)

    weight_hid, weight_out = train(
        x,
        y,
        weight_out,
        weight_hid,
        EPOCHS,
        LEARNING_RATE,
        activation,
        activation_deriv,
    )

    data = np.array([[0, 1]])
    layer_hid = activation(data.dot(weight_hid))
    layer_out = activation(layer_hid.dot(weight_out))

    print("Prediciton: ", layer_out)


hidden_sizes = (4, 10, 40, 50, 51, 60, 80, 100)

for sz in hidden_sizes:
    print("hidden size = ", sz)
    run(th, th_deriv, sz)
    print()

"""
После размера 50 началось переобучение,
результаты стали хуже
"""
