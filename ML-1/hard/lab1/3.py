"""
Добавьте ещё один скрытый слой.
Его значения надо также пропустить через функцию ReLU.
Проанализируйте результат.
"""

import numpy as np


def get_error(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2))


def relu(x):
    return (x > 0) * x


inp = np.array([[15, 10], [15, 15], [15, 20], [25, 10]])


true_prediction = np.array([[10, 20, 15, 20]]).T

layer_in_size = inp.shape[1]
layer_hid1_size = 10
layer_hid2_size = 7
layer_out_size = true_prediction.shape[1]

weights_hid1 = 2 * np.random.random((layer_in_size, layer_hid1_size)) - 1
weights_hid2 = 2 * np.random.random((layer_hid1_size, layer_hid2_size)) - 1
weights_out = 2 * np.random.random((layer_hid2_size, layer_out_size)) - 1

prediction_hid1 = relu(np.dot(inp, weights_hid1))
prediction_hid2 = relu(np.dot(prediction_hid1, weights_hid2))
prediction = np.dot(prediction_hid2, weights_out)

print("layer number 1")
print(prediction_hid1)
print("layer number 2")
print(prediction_hid2)
print("output")
print(prediction)

error = get_error(true_prediction, prediction)
print("RMSE:", error)
