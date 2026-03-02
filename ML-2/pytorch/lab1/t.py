'''
Используя фреймворк PyTorch, создайте нейросеть, которая будет перемножать входные 2 нейрона. 
Например, если подаёте (3, 4), то выходной нейрон должен быть примерно равен 12. 
В обучающем наборе должно быть 100 примеров. 
Примечание: набор обучающих данных можно сгенерировать с помощью random и операции перемножения
'''

import torch
from torch import nn
import random

class SimpleModel(nn.Module):
    def __init__(self, inp_size, hid_size, out_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(inp_size, hid_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_size, out_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def gen_data(num, low, high):
    inp = torch.randint(low, high, (num, 2))
    out = inp[:, 0] * inp[:, 1]
    return inp, out

inp_size = 2  
hid_size = 100
out_size = 1
learning_rate = 0.01
num_epochs = 200

model = SimpleModel(inp_size, hid_size, out_size)
loss = nn.MSELoss()  # MSE для регрессии
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam работает лучше

train_inp, train_out = gen_data(100, -10, 10)
