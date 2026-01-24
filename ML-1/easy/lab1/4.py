"""
Измените функцию neural_network так, чтобы она возвращала не только предсказание (prediction),
но и весь список промежуточных значений
(произведение каждого элемента входных данных на соответствующий вес).
Выведите обе переменные (prediction и список промежуточных значений) на экран.
"""


def network(inp, weight):
    predict = 0
    inter = []
    for i in range(len(weight)):
        plus = weight[i] * inp[i]
        inter.append(plus)
        predict += plus
    return (predict, inter)


out1 = network([150, 40], [0.3, 0.4])
out2 = network([80, 60], [0.2, 0.4])

print(out1, out2)
