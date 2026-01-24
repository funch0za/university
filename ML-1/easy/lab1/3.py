"""
Модифицируйте функцию neural_network так, чтобы она принимала два входных параметра: inp и bias.
Результат будет задан как inp * weight + bias.
Запустите функцию с новыми значениями inp, weight и bias. Как изменится выходная переменная? Почему?
"""


def neuralNetwork(inp, weight, bias):
    prediction = inp * weight + bias
    return prediction


out_1 = neuralNetwork(150, 0.3, 0.2)
out_2 = neuralNetwork(130, 0.4, 0.9)
print(out_1)
print(out_2)
