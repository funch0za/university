"""
Напишите по памяти код "Функция активации softmax".
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


x = np.array(
    [
        [0, 0, 0, 0],  # 0
        [0, 0, 0, 1],  # 1
        [0, 0, 1, 0],  # 2
        [0, 0, 1, 1],  # 3
        [0, 1, 0, 0],  # 4
        [0, 1, 0, 1],  # 5
        [0, 1, 1, 0],  # 6
        [0, 1, 1, 1],  # 7
        [1, 0, 0, 0],  # 8
        [1, 0, 0, 1],  # 9
    ]
)

y = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

input_size = len(x[0])
hidden_size = 15
output_size = len(y[0])

np.random.seed()
weight_hid = np.random.randn(input_size, hidden_size) * 0.1
weight_out = np.random.randn(hidden_size, output_size) * 0.1

for i in range(10000):
    layer_hid = sigmoid(x.dot(weight_hid))
    layer_out = softmax(layer_hid.dot(weight_out))

    error = layer_out - y

    layer_out_delta = error / output_size
    layer_hid_delta = layer_out_delta.dot(weight_out.T) * layer_hid * (1 - layer_hid)

    weight_out -= 0.1 * layer_hid.T.dot(layer_out_delta)
    weight_hid -= 0.1 * x.T.dot(layer_hid_delta)

    if i % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"epoch {i}, error: {loss:.4f}")


def predict(inp):
    layer_hid = sigmoid(inp.dot(weight_hid))
    layer_out = softmax(layer_hid.dot(weight_out))
    return np.argmax(layer_out)


print("=" * 40)

for i, inp in enumerate(x):
    predicted = predict(np.array([inp]))
    actual = np.argmax(y[i])
    print(f"input {inp} -> predict: {predicted}, correct: {actual}")
