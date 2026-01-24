"""
Напишите нейросеть, которая будет суммировать 2 числа.
То есть на входе у неё будет 2 нейрона, например: [10, 5], [0, -5], [2, 6],
а на выходе должен быть 1 нейрон. Для представленных входов,
выходы будут: [15], [-5], [8] соответственно.
Проверьте правильность работы нейросети, передав ей тестовые данные [12, 4]
на вход, а потом [3, -8].
Примечание: регулируйте параметры нейросети до тех пор,
пока она не будет выдавать результат для тестовых данных очень близкий к правильному
(ошибка не должна быть больше 1%).
"""

import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2))


def gradient(inp, weights, true_predict, learn_rate, epochs):
    for _ in range(epochs):
        error = 0
        delta = np.zeros_like(weights)
        for j in range(len(inp)):
            cur_inp = inp[j]
            cur_true_predict = true_predict[j]
            predict = neural_networks(cur_inp, weights)
            error += get_error(cur_true_predict, predict)
            print(
                "Prediction: %.10f, True_prediction: %.10f, Weights: %s"
                % (predict, cur_true_predict, weights)
            )
            delta += (predict - cur_true_predict) * cur_inp * learn_rate
        weights -= delta / len(inp)
        print("Errors: %.10f" % error)
        print("-------------------")
    return weights


def calc_sum(a, b, weights):
    return neural_networks(np.array([a, b]), weights)


def get_diff(a, b):
    return abs(a - b) / 100


def solve(a, b, weights):
    ans_from_NN = calc_sum(a, b, weights)
    ans = a + b
    diff_ans = get_diff(ans, ans_from_NN)
    print("data: ", a, b)
    print("ans from neural network: ", ans_from_NN)
    print("correct ans: ", ans)
    print("difference: ", diff_ans)
    print("-------------------")


def main():
    inp = np.array([[10, 5], [0, -5], [2, 6]])
    weights = np.array([0.2, 0.3])
    true_predict = np.array([15, -5, 8])
    learn_rate = 0.00001
    epochs = 10**4

    weights = gradient(inp, weights, true_predict, learn_rate, epochs)

    solve(12, 4, weights)
    solve(3, -8, weights)


main()
