import numpy as np


class Tensor(object):
    def __init__(self, data, creators=None, operation_on_creation=None):
        """
        Конструктор

        :param self: ссылка на текущий объект
        :param data: данные для тензора
        :param creators: родительские тензоры, из которых был создан текущий
        :param operation_on_creation: операция, которая создала этот тензор
        :param grad: градиент тензора
        """
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None

    def __add__(self, other):
        """
        Перегрузка операции +

        :param self: ссылка на текущий объект
        :param other: другой объект тензора
        """
        return Tensor(self.data + other.data, [self, other], "+")

    def __str__(self):
        """
        Удобный вывод

        :param self: ссылка на текущий объект
        """
        return str(self.data.__str__())

    def backward(self, grad):
        """
        Реализация обратного распространения

        :param self: ссылка на текущий объект
        :param grad: градиент тензора
        """

        if grad is None:
            # что-то сделать
            pass

        if self.creators is None:
            # что-то сделать
            pass

        self.grad = grad

        if self.operation_on_creation == "+":
            for i in range(len(self.creators)):
                self.creators[i].backward(grad)
