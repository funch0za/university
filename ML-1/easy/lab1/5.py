"""
Измените веса нейросети в коде и определите,
при каких значениях весов выходные данные для каждого элемента становятся больше 0.5.
Решите это методом проб и ошибок, меняя веса с небольшими шагами.
"""


def network(inp, weight):
    predict = [0] * len(weight)
    for i in range(len(predict)):
        predict[i] = inp * weight[i]
    return predict


weight = [0.001, 0.5]
inp = 4

STEP = 0.09
MAX_PREDICT = 0.5
predict = network(inp, weight)
while predict[0] <= MAX_PREDICT or predict[1] <= MAX_PREDICT:
    weight[0] += STEP
    weight[1] += STEP
    predict = network(inp, weight)

print(predict)
