from tensor import Tensor
from layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, input_count, output_count):
        super().__init__()
        # Xavier инициализация
        weight = np.random.randn(input_count, output_count) * np.sqrt(2.0/input_count) 
        self.weight = Tensor(weight, autograd=True)
        self.bias = Tensor(np.zeros(output_count), autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, inp):
        return inp.dot(self.weight) + self.bias.expand(0, len(inp.data))
