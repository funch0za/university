class Layer(object):
    def __init__(self):
        self.parameters = [] # параметры класса (наследников класса)

    def get_parameters(self): # геттер
        return self.parameters


