"""
Измените входные данные и вес нейросети в коде.
Запустите программу с новыми значениями и опишите, как это повлияло на выходные данные.
Объясните, почему это произошло с точки зрения работы нейронной сети.
"""


def neuralNetwork(inp, weight):
    prediction = inp * weight
    return prediction


out_1 = neuralNetwork(110, 0.9)
out_2 = neuralNetwork(10, 0.2)
print(out_1)
print(out_2)
