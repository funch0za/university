"""
Увеличьте количество нейронов в скрытом слое.
"""

import numpy as np


def get_error(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2))


def relu(x):
    return (x > 0) * x


inp = np.array([[15, 10], [15, 15], [15, 20], [25, 10]])
true_prediction = np.array([[10, 20, 15, 20]]).T


layer_hid_size = 5
layer_in_size = len(inp[0])
layer_out_size = len(true_prediction[0])


weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weights_out = 2 * np.random.random((layer_hid_size, layer_out_size)) - 1


prediction_hid = relu(np.dot(inp[0], weights_hid))
print(prediction_hid)


prediction = prediction_hid.dot(weights_out)
print(prediction)
