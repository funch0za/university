"""
Попробуйте увеличить количество эпох (num_epochs).
Как это влияет на точность модели?
Определите оптимальное количество эпох для этой модели.
"""

from keras.src.datasets import mnist
import numpy as np

DEBUG_MODE = False


def relu(x):
    return (x < 0) * x


def reluderiv(x):
    return x > 0


def gradient(
    epochs,
    train_img,
    train_labels,
    learning_rate,
    weight_hid,
    weight_out,
    dropout_rate,
    batch_size,
):
    middle_accuracy = 0
    for i in range(epochs):
        correct_answers = 0

        for j in range(len(train_img) // batch_size):
            """
            нужно переопределить индексы массивов
            так как теперь у нас пакетный градиентный спуск
            чтобы понять с каким пакетом данных мы работаем
            """
            batch_start = batch_size * j
            batch_end = batch_size * (j + 1)

            layer_in = train_img[batch_start:batch_end]
            layer_hid = relu(np.dot(layer_in, weight_hid))

            """
            Напишем dropout-маску для ввода в наш алгоритм регуляризации 
            по типу отсева некоторых данных для обучения. 
            Данную маску будем применять для регулировки прогноза на скрытом слое.
            """
            dropout_mask = np.random.rand(*layer_hid.shape) < dropout_rate
            layer_hid *= dropout_mask
            layer_out = np.dot(layer_hid, weight_out)

            for k in range(batch_size):
                correct_answers += bool(
                    np.argmax(layer_out[k : k + 1])
                    == np.argmax(train_labels[batch_start + k : batch_end + k + 1])
                )

            layer_out_delta = (
                layer_out - train_labels[batch_start:batch_end]
            ) / batch_size

            """
            Используем маску при обучении в процессе обратного распространения.
            """
            layer_hid_delta = (
                layer_out_delta.dot(weight_out.T) * reluderiv(layer_hid) * dropout_mask
            )

            weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
            weight_hid -= learning_rate * layer_in.T.dot(layer_hid_delta)
        current_accuracy = correct_answers * 100 / len(train_img)
        middle_accuracy += current_accuracy
        if DEBUG_MODE:
            print("Epoch: ", i)
            print("Accuracy: %.2f" % (current_accuracy))
    return weight_hid, weight_out, middle_accuracy / epochs


def check(test_img, test_labels, weight_hid, weight_out):
    correct_answers = 0
    for j in range(len(test_img)):
        layer_in = test_img[j : j + 1]
        layer_hid = relu(np.dot(layer_in, weight_hid))
        layer_out = np.dot(layer_hid, weight_out)
        correct_answers += bool(
            np.argmax(layer_out) == np.argmax(test_labels[j : j + 1])
        )
    current_accuracy = correct_answers * 100 / len(test_img)
    if DEBUG_MODE:
        print("Accuracy: %.2f" % (correct_answers * 100 / len(test_img)))
    return current_accuracy


def preparing_labels(labels, size):
    """
    бинарная категоризация
    преобразуем каждое значение в вектор
    например
    3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    10 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    one_hot_labels = np.zeros((len(labels), size))
    for j in range(len(labels)):
        one_hot_labels[j][labels[j]] = 1
    return one_hot_labels


def generate_weights(input_size, output_size):
    np.random.seed(2)
    return 0.2 * np.random.random((input_size, output_size)) - 0.1


def handwritten_detection(hidden_size, dropout_rate, epochs):
    TRAIN_IMG_CNT = 1000
    TEST_IMG_CNT = 10000
    IMG_SIZE = 28 * 28
    DIGITS_CNT = 10  # выходных нейронов столько же
    IMG_DEEP = 255
    BATCH_SIZE = 50
    LEARNING_RATE = 0.01

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_img = x_train[0:TRAIN_IMG_CNT].reshape(TRAIN_IMG_CNT, IMG_SIZE) / IMG_DEEP
    train_labels = y_train[0:TRAIN_IMG_CNT]

    test_img = x_test[0:TEST_IMG_CNT].reshape(TEST_IMG_CNT, IMG_SIZE) / IMG_DEEP
    test_labels = y_test[0:TEST_IMG_CNT]

    train_labels = preparing_labels(train_labels, DIGITS_CNT)
    test_labels = preparing_labels(test_labels, DIGITS_CNT)

    weight_hid = generate_weights(IMG_SIZE, hidden_size)
    weight_out = generate_weights(hidden_size, DIGITS_CNT)

    weight_hid, weight_out, middle_accuracy = gradient(
        epochs,
        train_img,
        train_labels,
        LEARNING_RATE,
        weight_hid,
        weight_out,
        dropout_rate,
        BATCH_SIZE,
    )
    check_accuracy = check(test_img, test_labels, weight_hid, weight_out)
    return (middle_accuracy, check_accuracy)


def run_set(dropout_rate):
    print(f"dropout_rate = {dropout_rate}")
    ans = []
    sizes = (50, 100)
    epochs_cnt = (10, 30, 100, 500, 1000)
    for size in sizes:
        for epochs in epochs_cnt:
            res = handwritten_detection(size, dropout_rate, epochs)
            if DEBUG_MODE:
                ans.append(
                    f"epochs = {epochs}, size = {size}, train accuracy = {res[0]}, test accuracy = {res[1]}"
                )
            else:
                print(
                    f"epochs = {epochs}, size = {size}, train accuracy = {res[0]}, test accuracy = {res[1]}"
                )

    if DEBUG_MODE:
        for i in ans:
            print(i)


def main():
    # run_set(0.2)
    run_set(0.5)
    # run_set(0)
    # run_set(1)


main()


"""
увеличение числа эпох привело к более лучшим результатам
но, если от 0 к 500 был большой скачок в качестве, 
то от 500 к 1000 скачок всего лишь на 1 %
"""
