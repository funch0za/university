"""
Добавьте в класс Tensor функцию ReLU.
Проверьте правильность работы данной функции
"""

from tensor import Tensor

x = Tensor([-2, -1, 0, 1, 2], autograd=True)
y = x.relu()
print("x = ", x)
print("y = ", y)

y.backward()
print("x.grad = ", x.grad)

