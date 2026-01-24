import numpy as np
from tensor import Tensor
from linear import Linear
from mse_loss import MSELoss
from SGD import SGD

class Net:
    def __init__(self):
        self.layer = Linear(3, 1)
        self.weights = self.layer.get_parameters()
    
    def forward(self, x):
        return self.layer.forward(x)

X = np.array([[1., 1., 1.],
              [2., 2., 2.],
              [0.5, 0.5, 0.5],
              [1., 2., 3.],
              [3., 5., 4.]])

y = np.prod(X, axis=1, keepdims=True)

X_tensor = Tensor(X, autograd=True)
y_tensor = Tensor(y, autograd=True)

model = Net()
opt = SGD(model.weights, 0.001)

loss_fn = MSELoss()

print("weight:", model.layer.weight.data.flatten())

for epoch in range(1000):
    pred = model.forward(X_tensor)
    loss = loss_fn.forward(pred, y_tensor)
    
    for w in model.weights:
        w.grad = None
    
    loss.backward()
    
    opt.step()
    
    if epoch % 200 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data.mean():.6f}')

print("\nFinal weight:", model.layer.weight.data.flatten())

print("-" * 20 + " TEST " + "-" * 20)
for i in range(len(X)):
    test = Tensor(X[i:i+1], autograd=True)
    pred = model.forward(test)
    print(f"{X[i]} -> predict: {pred.data[0][0]:.4f}, ok: {y[i][0]:.4f}")
