import numpy as np


class Tensor(object):
    ids = 0

    def __init__(
        self, data, creators=None, operation_on_creation=None, autograd=False, id=None
    ):
        """
        Конструктор

        :param self: ссылка на текущий объект
        :param data: данные для тензора
        :param creators: родительские тензоры, из которых был создан текущий
        :param operation_on_creation: операция, которая создала этот тензор
        :param grad: градиент тензора
        :param autograd:
        :param id: индефикатор для тензора
        """

        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None
        self.autograd = autograd
        self.children = {}

        self.id = id
        if id is None:
            self.id = Tensor.ids
            Tensor.ids += 1

        if creators is not None:
            for creator in creators:
                creator.children[self.id] = (
                    1
                    if self.id not in creator.children
                    else creator.children[self.id] + 1
                )

        self.creators = creators

    def __add__(self, other):
        """
        Перегрузка операции +

        :param self: ссылка на текущий объект
        :param other: другой объект тензора
        """

        return (
            Tensor(self.data + other.data, [self, other], "+", True)
            if self.autograd and other.autograd
            else Tensor(self.data + other.data)
        )

    def __str__(self):
        """
        Удобный вывод

        :param self: ссылка на текущий объект
        """

        return str(self.data.__str__())

    def backward(self, grad=None, grad_origin=None):
        """
        Реализация обратного распространения

        :param self: ссылка на текущий объект
        :param grad: градиент тензора
        :param grad_origin: тензор, созданный благодаря текущему тензору
        """

        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if (grad_origin is not None):
                if self.children[grad_origin.id] > 0:
                    self.children[grad_origin.id] -= 1

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if (self.creators is not None):
            print("OK 1")
            if (self.check_grads_from_children() or grad_origin is None):
                print("OK 2")
                if self.operation_on_creation == "+":
                    print("OK 3")
                    self.creators[0].backward(grad, self)
                    self.creators[1].backward(grad, self)

    def check_grads_from_children(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True
