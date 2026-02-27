import torch


def task2():
    '''
    Создайте 2 тензора 3x3 со случайными значениями. Выведите их.
    '''
    print("TASK 2")
    t1 = torch.rand(3, 3)
    t2  = torch.rand(3, 3)
    print("tensor 1 = ", t1, "\ntensor 2 = ", t2)
    return t1, t2


def taks3(t1, t2):
    '''
    Сложите созданные тензоры и выведите результат.
    '''
    print("TASK 3")
    print("tensor1 + tenso2 = ", t1 + t2)


def task4(t1, t2):
    '''
    Умножьте первый тензор на второй поэлементно и выведите результат
    '''
    print("TASK 4")
    print("tensor1 * tensor2 = ", t1 * t2)


def task5(t2):
    '''
    Транспонируйте второй тензор и выведите результат.
    '''
    print("TASK 5")
    t2_T = torch.transpose_copy(t2, 0, 1)
    print(t2_T)


def task6(t1, t2):
    '''
    Найдите среднее значение в каждом тензоре и выведите их.
    '''
    print("TASK 6")
    print(torch.mean(t1), torch.mean(t2))
    

def task7(t1, t2):
    '''
    Найдите максимальное значение в каждом тензоре и выведите их.
    '''
    print("TASK 7")
    print(torch.max(t1), torch.max(t2))


def main():
    t1, t2 = task2()
    taks3(t1, t2)
    task4(t1, t2)
    task5(t2)
    task6(t1, t2)
    task7(t1, t2)


main()