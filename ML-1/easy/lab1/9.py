"""
Выполните предыдущее задание, но с помощью цикла. После цикла выведите получившиеся веса.
"""


def network(inp, weight):
    predict = [0] * len(weight)
    for i in range(len(weight)):
        predict[i] = sum([inp[j] * weight[i][j] for j in range(len(inp))])
    return predict


inp = [50, 165, 45]
weights_1 = [0.2, 0.1, 0.65]
weights_2 = [0.3, 0.1, 0.7]
weights = [weights_1, weights_2]

STEP = 0.1
COUNT = 100
for i in range(COUNT):
    for j in range(COUNT):
        for k in range(COUNT):
            new_weights_1 = [i * STEP, j * STEP, k * STEP]
            new_weights = [new_weights_1, weights_2]
            predict = network(inp, new_weights)
            if len(set(predict)) == 1:
                i = COUNT
                j = COUNT
                k = COUNT
                weights = new_weights

print(network(inp, weights))
print(weights)
