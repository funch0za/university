from layer import Layer
class Softmax(Layer):
    def forward(self, inp):
        return inp.softmax()
