"""
Напишите код с циклом, где значение веса будет увеличиваться до тех пор, пока выходное значение меньше 0.5.
Как только один выход стал больше 0.5, то изменение его веса останавливается.
Как только второй выход стал больше 0.5, то изменение его веса также останавливается, а цикл завершается.
Выведите получившиеся веса.
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
    weight[0] += STEP if predict[0] <= MAX_PREDICT else 0
    weight[1] += STEP if predict[1] <= MAX_PREDICT else 0
    predict = network(inp, weight)

print(predict)
