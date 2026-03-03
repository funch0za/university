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
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hid_size, out_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out

class MultNetwork:
    def __init__(self, num_of_data, low_limit, high_limit):
        self.num_of_data = num_of_data
        self.low_limit = low_limit
        self.high_limit = high_limit

    def generate_data_for_training(self):
        raw_inp = torch.randint(self.low_limit, self.high_limit, (self.num_of_data, 2), dtype=torch.float32)
        raw_out = (raw_inp[:, 0] * raw_inp[:, 1]).unsqueeze(1)

        self.inp_mean = raw_inp.mean(dim=0, keepdim=True)
        self.inp_std = raw_inp.std(dim=0, keepdim=True) + 1e-8
        self.out_mean = raw_out.mean()
        self.out_std = raw_out.std() + 1e-8

        inp_data = (raw_inp - self.inp_mean) / self.inp_std
        out_data = (raw_out - self.out_mean) / self.out_std

        return inp_data, out_data

    def config_for_training(self):
        self.num_train_epoch = 500
        self.train_learning_rate = 0.01
        self.model = SimpleModel(2, 128, 1)
        self.optimizator = torch.optim.Adam(self.model.parameters(), lr=self.train_learning_rate) 
        self.loss_func = nn.MSELoss()

    def training(self):
        self.config_for_training()
        self.train_inp_data, self.train_out_data = self.generate_data_for_training()

        for epoch in range(self.num_train_epoch):
            self.optimizator.zero_grad()
            out = self.model(self.train_inp_data)
            error = self.loss_func(out, self.train_out_data)
            error.backward()
            self.optimizator.step()
            
            if epoch % 50 == 0:
                print("Эпоха - ", epoch, "Ошибка - ", error.item())

    def get_prediction(self, a, b):
        test_input = torch.tensor([[a, b]], dtype=torch.float32)
        test_input_std = (test_input - self.inp_mean) / self.inp_std
        normalized_pred = self.model(test_input_std).item()
        return normalized_pred * self.out_std + self.out_mean

network = MultNetwork(100, -10, 10)
network.training()

data = ((3, 4), (3, 3), (2, 3), (9, 2), (0, 1), (5, 6), (0, 0))
for d in data:
    print("Данные - ", d, "\n\tОжидается - ", d[0] * d[1], "\n\tПолучен - ", network.get_prediction(d[0], d[1]))
    print("Данные - ", (-d[0], d[1]), "\n\tОжидается - ", -d[0] * d[1], "\n\tПолучен - ", network.get_prediction(-d[0], d[1]))