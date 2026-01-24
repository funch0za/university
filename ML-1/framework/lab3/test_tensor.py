from tensor import Tensor

a_1 = Tensor([[1, 2, 3], [4, 5, 6]], autograd=True)
a_2 = Tensor([[2, 3], [2, 3], [2, 3]], autograd=True)

print("-" * 20 + "TEST TRANSPOSE" + "-" * 20)
print(
    "a_1 = ",
    a_1,
    "a_2 = ",
    a_2,
    "a_1_T = ",
    a_1.transpose(),
    "a_2_T = ",
    a_2.transpose(),
    sep="\n",
)

print("-" * 20 + "TEST DOT" + "-" * 20)
a_3 = a_1.dot(a_2)
a_3.backward(Tensor([1, 4]))
print(
    "a_1 = ",
    a_1,
    "a_2 = ",
    a_2,
    "a_3 = a_1 * a_2 =",
    a_3,
    "grad = [1, 4]",
    "a_1.grad = ",
    a_1.grad,
    sep="\n",
)

print("-" * 20 + "TEST SIGMOID" + "-" * 20)
a_1 = Tensor([[1, 2, 3], [4, 5, 6]], autograd=True)
a_3 = a_1.sigmoid()
a_3.backward(Tensor([4, 5, 10]))
print(
    "a_1 = ",
    a_1,
    "a_2 = ",
    a_2,
    "a_3 = a_1.sigmoid() =",
    a_3,
    "grad = [4, 5, 10]",
    "a_1.grad = ",
    a_1.grad,
    sep="\n",
)


print("-" * 20 + "TEST TANH" + "-" * 20)
a_2 = Tensor([[2, 3, 4], [2, 3, 5]], autograd=True)
a_4 = a_2.tanh()
a_4.backward(Tensor([4, 5, 10]))

print(
    "a_2 = ",
    a_2,
    "a_4 = a_2.tanh()",
    a_4,
    "a_3 = ",
    a_3,
    "a_2.grad = ",
    a_2.grad,
    sep="\n",
)
