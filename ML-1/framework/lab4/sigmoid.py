from layer import Layer

class Sigmoid(Layer):
    def forward(self, inp):
        return inp.sigmoid()
