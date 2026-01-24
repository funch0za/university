"""
Увеличьте или уменьшите количество скрытых нейронов в первом скрытом слое (layer_hid_1_size).
Как это влияет на скорость обучения и точность предсказания?
Какой размер слоя приводит к наилучшим результатам?
"""

import numpy as np


def relu(x):
    return (x > 0) * x


def reluderif(x):
    return x > 0


def gradient(inp, true_prediction, weight_hid, weight_out, learning_rate, num_epoch):
    for i in range(num_epoch):
        layer_out_error = 0
        for i in range(len(inp)):
            layer_in = inp[i : i + 1]
            layer_hid = relu(layer_in.dot(weight_hid))
            layer_out = layer_hid.dot(weight_out)
            layer_out_error += np.sum(layer_out - true_prediction[i : i + 1]) ** 2
            layer_out_delta = true_prediction[i : i + 1] - layer_out
            layer_hid_delta = layer_out_delta.dot(weight_out.T) * reluderif(layer_hid)
            weight_out += learning_rate * layer_hid.T.dot(layer_out_delta)
            weight_hid += learning_rate * layer_in.T.dot(layer_hid_delta)
            print(
                "Predictions: %s, true_predictions: %s"
                % (layer_out, true_prediction[i : i + 1])
            )
        print("Errors: %.4f" % layer_out_error)
        print("----------------------")
    return weight_hid, weight_hid


def neuro_training(inp, true_predict, learning_rate, num_epoch, layer_hid_size):
    print(f"learning rate = {learning_rate}")
    print(f"count of epochs = {num_epoch}")
    print(f"size of hidden layer = {layer_hid_size}")
    print("----------------------")

    layer_in_size = len(inp[0])
    layer_out_size = 1

    np.random.seed(100)
    weight_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
    weight_out = np.random.random((layer_hid_size, layer_out_size))

    gradient(inp, true_predict, weight_hid, weight_out, learning_rate, num_epoch)


def main():
    LEARNIN_RATE = 0.0001

    inp = np.array([[15, 10], [15, 18], [15, 20], [25, 10]])
    true_prediction = np.array([15, 18, 20, 25])

    neuro_training(inp, true_prediction, LEARNIN_RATE, 100, 3)

    # результат стал лучше
    neuro_training(inp, true_prediction, LEARNIN_RATE, 100, 5)

    # результат лучше не смотря на уменьшение кол-ва эпох
    neuro_training(inp, true_prediction, LEARNIN_RATE, 50, 5)

    # результат стал хуже
    neuro_training(inp, true_prediction, LEARNIN_RATE, 100, 2)

    # результат стал лучше, но потребовалось больше эпох
    neuro_training(inp, true_prediction, LEARNIN_RATE, 500, 2)


main()
