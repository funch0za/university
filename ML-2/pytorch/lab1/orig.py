
import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, inp_size, hid_size, out_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(inp_size, hid_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_size, out_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


inp_size = 3
hid_size = 4
out_size = 3
learning_rate = 0.001
num_epochs = 100

model = SimpleModel(inp_size, hid_size, out_size)

loss = nn.CrossEntropyLoss() # функция нахождения перекрестной энтропии.
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

x = torch.randn(100, inp_size)
print(x)

y = torch.randint(0, out_size, (100, ))

print(y)

for epoch in range(num_epochs): 
    optim.zero_grad()
    out = model(x)
    error = loss(out, y)
    error. backward()
    optim.step()
    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch + 1}], Потери: {error.item(): .4f}')

    
print(model(torch.randn(10, inp_size)))