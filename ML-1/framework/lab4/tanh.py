from layer import Layer

class Tanh(Layer):
    def forward(self, inp):
        return inp.tanh()
