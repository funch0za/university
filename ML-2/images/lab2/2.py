'''
Измените модель, сделав её такой: свёрточный слой, 
слой нормализации, слой ReLU, слой max-pooling, 
свёрточный слой, слой нормализации, слой ReLU, 
линейный слой. Как изменился результат обучения
в плане точности и скорости? 
Примечание: не забудьте поменять другие параметры 
в модели, иначе обучение не начнётся.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

ROOT = "../data"
MODEL_FILENAME = "model2.pth"

class Model(nn.Module):
    CONV_KERNEL_SZ = 5
    CONV_STRIDE = 1
    CONV_PADDING = 2

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=self.CONV_KERNEL_SZ,
            stride=self.CONV_STRIDE,
            padding=self.CONV_PADDING,
        )

        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=self.CONV_KERNEL_SZ,
            stride=self.CONV_STRIDE,
            padding=self.CONV_PADDING,
        )
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.fc = nn.Linear(32 * 16 * 16, out_features=10)

    def forward(self, inp):
        # свёрточный слой
        # слой нормализации
        # слой ReLU
        # слой max-pooling
        out = self.conv1(inp)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)

        # свёрточный слой
        # слой нормализации
        # слой ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # линейный слой
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class GraphicsNetwork:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    def __init__(self, data_dir):
        self.model = Model()
        self.load_data(data_dir, self.BATCH_SIZE)

    def load_data(self, dir, batch_size):
        transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = CIFAR10(
            train=True, transform=transformations, root=dir, download=True
        )
        self.train_data_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        test_set = CIFAR10(
            train=False, transform=transformations, root=dir, download=True
        )
        self.test_data_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

    def training(self, epochs=5, DEBUG=False):
        self.model.train()
        self.loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.LEARNING_RATE, weight_decay=0.0001
        )

        for epoch in range(epochs):
            for i, (images, labels) in enumerate(self.train_data_loader, 0):
                self.optim.zero_grad()
                out = self.model(images)
                error = self.loss(out, labels)
                error.backward()
                self.optim.step()

    def test(self):
        self.model.eval()
        accuracy = 0
        total = 0

        with torch.no_grad():
            for test_data in self.test_data_loader:
                images, labels = test_data

                output = self.model(images)
                predict = torch.max(output.data, 1)[1]
                accuracy += (predict == labels).sum().item()
                total += labels.size(0)

        return 100 * accuracy / total

    def show_predictions(self, num_images=5):
        self.model.eval()

        data_iter = iter(self.test_data_loader)
        images, labels = next(data_iter)

        images = images[:num_images]
        labels = labels[:num_images]

        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)

        images = images / 2 + 0.5

        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        for i in range(num_images):
            ax = axes[i]
            img = np.transpose(images[i].numpy(), (1, 2, 0))
            ax.imshow(img)
            ax.axis("off")

            title = f"Predict: {classes[predicted[i]]}\n Realy:{classes[labels[i]]}"
            ax.set_title(title)

        plt.tight_layout()
        plt.show()


def save_model(network, filename):
    checkpoint = {
        "model_state_dict": network.model.state_dict(),
        "optimizer_state_dict": (
            network.optim.state_dict() if hasattr(network, "optim") else None
        ),
        "batch_size": network.BATCH_SIZE,
        "learning_rate": network.LEARNING_RATE,
        "model_architecture": str(network.model),
    }
    torch.save(checkpoint, filename)


def load_model(filename, data_dir):
    checkpoint = torch.load(filename)

    network = GraphicsNetwork(data_dir)

    network.model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Batch size: {checkpoint.get('batch_size', 'N/A')}")
    print(f"Learning rate: {checkpoint.get('learning_rate', 'N/A')}")

    return network


def test_save():
    net = GraphicsNetwork(ROOT)
    net.training()
    print(net.test())
    save_model(net, MODEL_FILENAME)
    return net


if input("NEW MODEL (N) or MODEL FROM FILE (F)") == "F":
    net = load_model(MODEL_FILENAME, ROOT)
else:
    net = test_save()

print("accuracy: ", net.test())

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

net.show_predictions(num_images=5)
