import numpy as np
import sys

if len(sys.argv) > 1 and sys.argv[1] == "1":
    from ex1.tensor import Tensor

    print("Используется Tensor из упражнения 1.")
elif len(sys.argv) > 1 and sys.argv[1] == "2":
    from ex2.tensor import Tensor

    print("Используется Tensor из упражнения 2.")
else:
    print("Выберите Tensor (1 или 2).")
    exit(0)

print("=" * 60)

x = Tensor([1.0, 2.0, 3.0], autograd=True)
y = Tensor([4.0, 5.0, 6.0], autograd=True)

print(f"Тензоры:")
print(f"  x = {x.data}, x.id = {x.id}")
print(f"  y = {y.data}, y.id = {y.id}")

z = x + y

x = x + x + y

x.backward()

print("grad x")
print(x.grad.data)

print(f"\nz = x + y")
print(f"  z = {z.data}")
print(f"  Ожидаемый результат: [5. 7. 9.]")

print(f"\nСтруктура:")
print(f"  z.creators: {[creator.id for creator in z.creators]}")
print(f"  z.operation_on_creation: '{z.operation_on_creation}'")
print(f"  x.children: {x.children}")
print(f"  y.children: {y.children}")

print(f"\nbackward для z")
z.backward()

print(f"\nГрадиенты после backward():")
print(f"  z.grad: {z.grad.data}")
print(f"  x.grad: {x.grad.data}")
print(f"  y.grad: {y.grad.data}")

expected_grad = np.array([1.0, 1.0, 1.0])
x_grad_correct = (
    np.array_equal(x.grad.data, expected_grad) if x.grad is not None else False
)
y_grad_correct = (
    np.array_equal(y.grad.data, expected_grad) if y.grad is not None else False
)

print(
    "Градиент для x правильный." if x_grad_correct else "Градиент для x неправильный."
)
print(
    "Градиент для y правильный." if y_grad_correct else "Градиент для y неправильный."
)
