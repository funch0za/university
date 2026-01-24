import numpy as np
from tensor import Tensor
from SGD import SGD

np.random.seed(0)  # для того чтобы были постоянно одни и теже данные при каждом запуске
inp = Tensor(
    [[2, 3], [5, 10]], autograd=True
)  # два набора входных данных, два нейрона на вход
true_predictions = Tensor(
    [[5], [15]], autograd=True
)  # прогноз для двух наборов данных, один выходной нейрон

weights = [  # список, состоящий из двух классов
    Tensor(np.random.rand(2, 2), autograd=True),
    Tensor(np.random.rand(2, 1), autograd=True),
]
sgd = SGD(weights, 0.001)
num_epochs = 10


for i in range(num_epochs):
    # рассчитываем прогноз, реализуя прямое распространение
    prediction = inp.dot(weights[0]).dot(weights[1])
    # поэтому выполним перемножение на самого себя. Операция умножения реализована
    error = (prediction - true_predictions) * (prediction - true_predictions)
    error.backward(Tensor(np.ones_like(error.data)))
    sgd.step()
    print("Error: ", error)

print("weights = ", weights)

print(Tensor([3, 4]).dot(weights[0]).dot(weights[1]))


