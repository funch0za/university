"""
С помощью нашего фреймворка создайте нейросеть, 
которая будет выводить результат перемножения 3 входных чисел.
После обучения убедитесь, что при подаче на вход (3, 5, 4), 
нейросеть выдаёт примерно 60. Примечание: обучение должно вестись на других числах.
"""

import numpy as np
from tensor import Tensor
from SGD import SGD

class Net:
    def __init__(self):
        """
        Конструктор
        """
        self.W = Tensor([[0.3], [0.3], [0.3]], autograd=True) # веса
        self.weights = [self.W] 
    
    def forward(self, x):
        return x.dot(self.W)  # без смещения

X = np.array([[1., 1., 1.],
              [2., 2., 2.],
              [0.5, 0.5, 0.5],
              [1., 2., 3.],
              [3, 5, 4]])

y = np.prod(X, axis=1, keepdims=True)

X_tensor = Tensor(X, autograd=True)
y_tensor = Tensor(y, autograd=True)

model = Net()
opt = SGD(model.weights, 0.001)

print("weights:", model.W.data.flatten())

for epoch in range(1000):
    pred = model.forward(X_tensor)
    loss = ((pred - y_tensor) * (pred - y_tensor)).__sum__(0)
    
    for w in model.weights:
        w.grad = None
    loss.backward()
    
    opt.step()
    
    if epoch % 200 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data.mean():.6f}')

print("finaly weighs:", model.W.data)

print("-" * 20 + " TEST " + "-" * 20)
for i in range(len(X)):
    test = Tensor(X[i:i+1], autograd=True)
    pred = model.forward(test)
    print(f"{X[i]} -> predict: {pred.data[0][0]:.4f}, ok: {y[i][0]:.4f}")
