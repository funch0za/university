"""
Напишите по памяти код из урока "Распознавание рукописных цифр".
Примечание: да, это сложновато, но нужно.
Необязательно всё делать точь-в-точь как было предложено
"""

from keras.src.datasets import mnist
import numpy as np


def relu(x):
    return (x < 0) * x


def reluderiv(x):
    return x > 0


def gradient(epochs, train_img, train_labels, learning_rate, weight_hid, weight_out):
    for i in range(epochs):
        correct_answers = 0
        for j in range(len(train_img)):
            layer_in = train_img[j : j + 1]
            layer_hid = relu(np.dot(layer_in, weight_hid))
            layer_out = np.dot(layer_hid, weight_out)

            # np.argmax - находим индекс максимального элемента из массива полученных прогнозов
            correct_answers += bool(
                np.argmax(layer_out) == np.argmax(train_labels[j : j + 1])
            )

            layer_out_delta = layer_out - train_labels[j : j + 1]
            layer_hid_delta = layer_out_delta.dot(weight_out.T) * reluderiv(layer_hid)

            weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
            weight_hid -= learning_rate * layer_in.T.dot(layer_hid_delta)
            print("Epoch: ", i)
            print("Accuracy: %.2f" % (correct_answers * 100 / len(train_img)))
    return weight_hid, weight_out


def check(test_img, test_labels):
    correct_answers = 0
    for j in range(len(test_img)):
        layer_in = test_img[j : j + 1]
        layer_hid = relu(np.dot(layer_in, weight_hid))
        layer_out = np.dot(layer_hid, weight_out)
        correct_answers += bool(
            np.argmax(layer_out) == np.argmax(test_labels[j : j + 1])
        )
    print("Accuracy: %.2f" % (correct_answers * 100 / len(test_img)))


def preparing_labels(labels):
    """
    бинарная категоризация
    преобразуем каждое значение в вектор
    например
    3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    10 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    one_hot_labels = np.zeros((len(labels), DIGITS_CNT))
    for j in range(len(labels)):
        print(labels[j])
        one_hot_labels[j][labels[j]] = 1
    return one_hot_labels


def generate_weights(input_size, output_size):
    np.random.seed(2)
    return 0.2 * np.random.random((input_size, output_size)) - 0.1


TRAIN_IMG_CNT = 1000
TEST_IMG_CNT = 10000
IMG_SIZE = 28 * 28
DIGITS_CNT = 10  # выходных нейронов столько же
IMG_DEEP = 255
LEARNING_RATE = 0.01
EPOCHS_CNT = 100
HIDDEN_SIZE = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_img = x_train[0:TRAIN_IMG_CNT].reshape(TRAIN_IMG_CNT, IMG_SIZE) / IMG_DEEP
train_labels = y_train[0:TRAIN_IMG_CNT]

test_img = x_test[0:TEST_IMG_CNT].reshape(TEST_IMG_CNT, IMG_SIZE) / IMG_DEEP
test_labels = y_test[0:TEST_IMG_CNT]

train_labels = preparing_labels(train_labels)
test_labels = preparing_labels(test_labels)

weight_hid = generate_weights(IMG_SIZE, HIDDEN_SIZE)
weight_out = generate_weights(HIDDEN_SIZE, DIGITS_CNT)

weight_hid, weight_out = gradient(
    EPOCHS_CNT, train_img, train_labels, LEARNING_RATE, weight_hid, weight_out
)
check(test_img, test_labels)
