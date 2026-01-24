from SGD import SGD
from sequential import Sequential
from mse_loss import MSELoss
from tensor import Tensor
from linear import Linear
from layer import Layer
from sigmoid import *
from tanh import *
from softmax import *
import numpy as np

#Example
np.random.seed(0)
#входные данные
x = Tensor([
    [0,0,0,0], # 0
    [0,0,0,1], # 1
    [0,0,1,0], # 2
    [0,0,1,1], # 3
    [0,1,0,0], # 4
    [0,1,0,1], # 5
    [0,1,1,0], # 6
    [0,1,1,1], # 7
    [1,0,0,0], # 8
    [1,0,0,1]  # 9
], autograd=True)
# массив правильных ответов
y = Tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], autograd=True)

#Cоздаем модель нейронной сети
model = Sequential([Linear(4,15), Sigmoid(), Linear(15,10), Softmax()])
sgd = SGD(model.get_parameters(), 0.01) #оптимайзер
loss = MSELoss()
epochs = 10000

for epoch in range(epochs):
    predictions = model.forward(x)
    error = loss.forward(predictions, y)
    error.backward(Tensor(np.ones_like(error.data)))
    sgd.step()

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Error: {error}")


def predict(inp):
    output_layer = model.forward(inp)
    # print(output_layer)
    return np.argmax(output_layer.data)
# так как тензор нельзя перебрать с помощью цикла for изменим входные данные
x = ([
    [0,0,0,0], # 0
    [0,0,0,1], # 1
    [0,0,1,0], # 2
    [0,0,1,1], # 3
    [0,1,0,0], # 4
    [0,1,0,1], # 5
    [0,1,1,0], # 6
    [0,1,1,1], # 7
    [1,0,0,0], # 8
    [1,0,0,1]  # 9
])

for inp in x:
    print("------------------------------------")
    print(f"Предсказанная цифра для {inp}:", predict(Tensor([inp])))


