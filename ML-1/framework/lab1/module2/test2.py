import numpy as np
from ex2.tensor import Tensor

t_1 = Tensor([1, 2, 3], autograd=True)
t_2 = Tensor([1, 2, 3], autograd=True)
t_3 = Tensor([1, 2, 3], autograd=True)

a_add_1 = t_1 + t_2 + t_2 + t_3
#a_add_1.backward(Tensor([2, 3, 4]))
a_add_1.backward()
print(t_2.grad)
