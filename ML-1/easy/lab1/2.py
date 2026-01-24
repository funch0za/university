"""
Создайте список входных данных (например, inputs = [150, 160, 170, 180, 190])
и используйте цикл for для вычисления выходных данных нейросети для каждого значения в списке.
Распечатайте выходные данные для каждого входного значения.
"""


def neuralNetwork(inp, weight):
    prediction = inp * weight
    return prediction


inputs = [150, 160, 170, 180, 190]
for inp in inputs:
    print(neuralNetwork(inp, 12))
