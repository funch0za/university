'''
Используя фреймворк PyTorch, создайте нейросеть, которая будет перемножать входные 2 нейрона. 
Например, если подаёте (3, 4), то выходной нейрон должен быть примерно равен 12. 
В обучающем наборе должно быть 100 примеров. 
Примечание: набор обучающих данных можно сгенерировать с помощью random и операции перемножения
'''

import torch
from torch import nn
import random

LOW = -30
HIGH = 30

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
        self.num_train_epoch = 2000
        self.train_learning_rate = 0.01
        self.model = SimpleModel(2, 128, 1)
        self.optimizator = torch.optim.Adam(self.model.parameters(), lr=self.train_learning_rate) 
        self.loss_func = nn.MSELoss()

    def generate_data_for_training(self):
        raw_inp = torch.randint(self.low_limit, self.high_limit, (self.num_of_data, 2), dtype=torch.float32)
        raw_out = (raw_inp[:, 0] * raw_inp[:, 1]).unsqueeze(1)

        self.inp_mean = raw_inp.mean(dim=0, keepdim=True)
        self.inp_std = raw_inp.std(dim=0, keepdim=True)
        self.out_mean = raw_out.mean()
        self.out_std = raw_out.std()

        inp_data = (raw_inp - self.inp_mean) / self.inp_std
        out_data = (raw_out - self.out_mean) / self.out_std

        return inp_data, out_data

    def training(self, DEBUG=False):
        self.train_inp_data, self.train_out_data = self.generate_data_for_training()

        for epoch in range(self.num_train_epoch):
            self.optimizator.zero_grad()
            out = self.model(self.train_inp_data)
            error = self.loss_func(out, self.train_out_data)
            error.backward()
            self.optimizator.step()
            
            if epoch % 50 == 0 and DEBUG:
                print("Эпоха - ", epoch, "Ошибка - ", error.item())

    def get_prediction(self, a, b):
        test_input = torch.tensor([[a, b]], dtype=torch.float32)
        test_input_std = (test_input - self.inp_mean) / self.inp_std
        normalized_pred = self.model(test_input_std).item()
        return normalized_pred * self.out_std + self.out_mean

def test_network(network):
    print("-" * 100)
    DATA = ((3, 4), (3, 3), (2, 3), (9, 2), (0, 1), (5, 6), (0, 0), (14, 2), (20, 10), (30, 0))
    for d in DATA:
        print("Данные - ", d, "\n\tОжидается - ", d[0] * d[1], "\n\tПолучен - ", network.get_prediction(d[0], d[1]))
        print("Данные - ", (-d[0], d[1]), "\n\tОжидается - ", -d[0] * d[1], "\n\tПолучен - ", network.get_prediction(-d[0], d[1]))
    print("-" * 100)

def save_model(network, filename):
    torch.save({
        'model_state_dict': network.model.state_dict(),
        'inp_mean': network.inp_mean,
        'inp_std': network.inp_std,
        'out_mean': network.out_mean,
        'out_std': network.out_std
    }, filename)

def task_8_11():
    network100 = MultNetwork(100, LOW, HIGH)
    network100.training()
    test_network(network100)
    save_model(network100, "network100.pth")

    network10 = MultNetwork(10, LOW, HIGH)
    network10.training()
    test_network(network10)
    save_model(network10, "network10.pth")

    network1000 = MultNetwork(1000, LOW, HIGH)
    network1000.training()
    test_network(network1000)
    save_model(network1000, "network1000.pth")

def load_model(filename):
    struct = torch.load(filename)
    network = MultNetwork(10, LOW, HIGH)

    network.model.load_state_dict(struct['model_state_dict'])
    network.inp_mean = struct['inp_mean']
    network.inp_std = struct['inp_std']
    network.out_mean = struct['out_mean']
    network.out_std = struct['out_std']
    
    return network

def task_12_13():
    network100 = load_model("network100.pth")
    test_network(network100)
    network10 = load_model("network10.pth")
    test_network(network10)
    network1000 = load_model("network1000.pth")
    test_network(network1000)

task_8_11()
task_12_13()