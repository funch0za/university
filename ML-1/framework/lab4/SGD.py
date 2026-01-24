class SGD(object):
    def __init__(self, weights, learning_rate=0.01):
        """
        Конструтктор

        :param self:
        :param weights:
        :param learning_rate:
        """
        self.weights = weights
        self.learning_rate = learning_rate

    def step(self):
        """
        :param self:
        """
        for weight in self.weights:
            weight.data -= self.learning_rate * weight.grad.data
            # обнуляем градиент для того,
            # чтобы он не влиял на вычисления следующей итерации цикла
            weight.grad.data *= 0
