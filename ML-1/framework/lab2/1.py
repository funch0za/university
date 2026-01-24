import copy  # чтобы передать копии объектов
from tensor import Tensor


def test_neg(a_1, a_2, a_3):
    """
    Добавим в вычисление a_add_1 отрицательный тензор.
    В результате вызова функции обратного распространения для тензора,
    полученного из a_add_1, к тензору a_2 придет нулевой градиент.
    """
    print("-----------TEST NEGATIVE-----------")

    a_add_1 = a_1 + (-a_2)
    a_add_2 = a_2 + a_3

    a_add_1.backward()
    a_add_2.backward()

    print("a_1.grad = ", a_1.grad)
    print("a_2.grad = ", a_2.grad)
    print("a_3.grad = ", a_3.grad)


def test_sub(a_1, a_2, a_3):
    """
    Тоже самое что и в test_neg.
    """
    print("-----------TEST SUBTRACTION-----------")

    a_sub_1 = a_1 - a_2
    a_sub_2 = a_2 - a_3

    a_add = a_sub_1 + a_sub_2
    a_add.backward(Tensor([4, 5, 3]))

    print("a_1.grad = ", a_1.grad)
    print("a_2.grad = ", a_2.grad)
    print("a_3.grad = ", a_3.grad)


def test_mul(a_1, a_2):
    print("-----------TEST MULTIPLICATION-----------")

    a_mult = a_1 * a_2
    a_mult.backward(Tensor([4, 5, 3]))

    print("a_mult.grad = ", a_mult.grad)
    print("a_1.grad = ", a_1.grad)
    print("a_2.grad = ", a_2.grad)


def test_sum(a_1):
    print("-----------TEST SUM-----------")

    a_2 = a_1.__sum__(0)
    a_3 = a_1.__sum__(1)

    print(a_2)
    print(a_3)

    a_2.backward(Tensor([5, 10, 20]))
    print(a_1.grad)


def test_1():
    print("-----------TEST 1-----------")
    a_1 = Tensor([1, 2, 3], autograd=True)
    a_2 = Tensor([4, 5, 6], autograd=True)
    a_3 = Tensor([9, 8, 7], autograd=True)

    test_sub(copy.copy(a_1), copy.copy(a_2), copy.copy(a_3))
    test_neg(copy.copy(a_1), copy.copy(a_2), copy.copy(a_3))
    test_mul(copy.copy(a_1), copy.copy(a_2))

    a_1 = Tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]], autograd=True)

    test_sum(copy.copy(a_1))


def test_2():
    print("-----------TEST 2-----------")
    a_1 = Tensor([-1, 1, 1], autograd=True)
    a_2 = Tensor([1, -1, 1], autograd=True)
    a_3 = Tensor([1, 1, -1], autograd=True)

    test_sub(copy.copy(a_1), copy.copy(a_2), copy.copy(a_3))
    test_neg(copy.copy(a_1), copy.copy(a_2), copy.copy(a_3))
    test_mul(copy.copy(a_1), copy.copy(a_2))

    a_1 = Tensor([[1, -20, 3]], autograd=True)

    test_sum(copy.copy(a_1))


test_1()
print("\n\n")
test_2()
