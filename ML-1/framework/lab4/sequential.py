from layer import Layer

class Sequential(Layer):
    def __init__(self, layers): # принимает список всех слоев
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layerd.append(layer)

    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params
