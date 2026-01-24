import numpy as np
from tensor import Tensor
from linear import Linear
from rmse_loss import RMSELoss
from SGD import SGD

class Net:
    def __init__(self):
        self.layer = Linear(2, 1)
        self.weights = self.layer.get_parameters()
    
    def forward(self, x):
        return self.layer.forward(x)

model = Net()
loss_fn = RMSELoss()
optimizer = SGD(model.weights, learning_rate=0.01)

inp_data = np.array([
    [150, 40],
    [140, 35],
    [155, 45],
    [185, 95],
    [145, 40],
    [195, 100],
    [180, 95],
    [170, 80],
    [160, 90]
])

true_predictions = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100]).reshape(-1, 1)

X_min = inp_data.min(axis=0)
X_max = inp_data.max(axis=0)
inp_data_norm = (inp_data - X_min) / (X_max - X_min + 1e-8)

y_min = true_predictions.min()
y_max = true_predictions.max()
true_predictions_norm = (true_predictions - y_min) / (y_max - y_min + 1e-8)

X_tensor = Tensor(inp_data_norm, autograd=True)
y_tensor = Tensor(true_predictions_norm, autograd=True)

for i in range(5000):
    predictions = model.forward(X_tensor)
    loss = loss_fn.forward(predictions, y_tensor)
    
    for w in model.weights:
        w.grad = None
    
    loss.backward()
    optimizer.step()
    
    if i % 1000 == 0:
        pred_denorm = predictions.data * (y_max - y_min) + y_min
        error = np.sqrt(np.mean((pred_denorm - true_predictions) ** 2))
        print(f"epoch {i}, Loss: {loss.data.mean()}, Error: {error}")

def denormalize(val):
    return val * (y_max - y_min) + y_min

for j in range(len(inp_data)):
    test_norm = (inp_data[j] - X_min) / (X_max - X_min)
    test = Tensor([test_norm], autograd=True)
    pred_norm = model.forward(test)
    pred = denormalize(pred_norm.data[0][0])
    print(f"Input: {inp_data[j]}, Prediction: {pred}, True: {true_predictions[j][0]}")

def calc_prob(person_h, person_w):
    test_norm = np.array([(person_h - X_min[0]) / (X_max[0] - X_min[0]),
                          (person_w - X_min[1]) / (X_max[1] - X_min[1])])
    test_input = Tensor([test_norm], autograd=True)
    prediction_norm = model.forward(test_input)
    prediction = denormalize(prediction_norm.data[0][0])
    return prediction

print(f"\ncalc_prob(150, 45) = {calc_prob(150, 45)}")
print(f"calc_prob(170, 85) = {calc_prob(170, 85)}")
