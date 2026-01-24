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
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

        self.creators = creators

    def __add__(self, other):
        """
        Суммирование тензоров

        :param self: ссылка на текущий объект
        :param other: другой объект тензора
        """

        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __neg__(self):
        """
        Отрицание

        :param self: ссылка на текущий объект
        """

        if self.autograd:
            return Tensor(self.data * (-1), [self], "-1", True)
        return Tensor(self.data * (-1))

    def __mul__(self, other):
        """
        Умножение тензоров

        :param self: ссылка на текущий объект
        :param other: другой объект тензора
        """

        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    def __sub__(self, other):
        """
        Вычитание тензоров

        :param self: ссылка на текущий объект
        :param other: другой объект тензора
        """

        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        return Tensor(self.data - other.data)

    def __sum__(self, axis):
        """
        Суммирование

        :param self: ссылка на текущий объект
        :param axis: это ось, по которой будем проводить суммирование
        """

        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)
        return Tensor(self.data.sum(axis))

    def expand(self, axis, count_copies):
        """
        :param self: ссылка на текущий объект
        :param axis: ось
        :param count_copies:
        """

        # записываем индексы элементов тензора с заданной размерностью
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))

        # определяем размерность будущего тензора после расширения
        expand_shape = list(self.data.shape) + [count_copies]
        # повторяем элементы тензора заданное количество раз 
        # и преобразуем к рассчитанной размерности
        expand_data = self.data.repeat(count_copies).reshape(expand_shape)
        # в случае нулевой оси производим транспонирование в соответствии с кортежем, 
        # иначе не надо его производить.
        expand_data = expand_data.transpose(transpose)

        if self.autograd:
            return Tensor(expand_data, [self], "expand_" + str(axis), True)
        return Tensor(expand_data)

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
            if (grad_origin is not None) and self.children[grad_origin.id] > 0:
                self.children[grad_origin.id] -= 1

        self.grad = grad if self.grad is None else self.grad + grad
        
        operation = self.operation_on_creation
        if (self.operation_on_creation is not None) and '_' in self.operation_on_creation:
            operation, axis = self.operation_on_creation.split('_')
            axis = int(axis)

        if (self.creators is not None) and (
            self.check_grads_from_children() or grad_origin is None
        ):
            if operation == "+":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)
            elif operation == "-1":
                # инверсия градиента
                self.creators[0].backward(self.grad.__neg__(), self)
            elif operation == "*":
                self.creators[0].backward(self.grad * self.creators[1], self)
                self.creators[1].backward(self.grad * self.creators[0], self)
            elif operation == "-":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad.__neg__(), self)
            elif operation == "sum":
                self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)
            elif operation == "expand":
                self.creators[0].backward(self.grad.__sum__(axis), self)
            else:
                # error
                pass

    def check_grads_from_children(self):
        """
        Проверка детей

        :param self: ссылка на текущий объект
        """

        for id in self.children:
            if self.children[id] != 0:
                return False
        return True
