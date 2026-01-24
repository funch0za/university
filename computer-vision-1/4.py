"""
Задание #4
сделать сегментацию на основе суперпиксельной сегментации на основе SLIC, SEEDS, LSC
"""

import cv2
import numpy as np


def seeds_segmentation(image, n_segments=100):
    """
    Energy-Driven Sampling (SEEDS) оптимизирует энергию на основе гистограмм цвета и границ, начиная с иерархической сетки блоков и уточняя границы от крупных к мелким. Это быстрый алгоритм реального времени с параметрами num_superpixels, num_levels и histogram_bins; в OpenCV — cv.ximgproc.createSuperpixelSEEDS.
    """

    h, w = image.shape[:2]

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        w,
        h,
        3,  # ширина, высота, каналы
        n_segments,  # количество суперпикселей
        # Алгоритм будет стараться разбить изображение на примерно такое количество областей.
        num_levels=5,  # количество уровней
        # Уровень 0: мелкие сегменты (много маленьких областей)
        # Уровень 1: более крупные сегменты
        # Уровень 2: еще крупнее
        # Уровень 3: самые крупные сегменты
        prior=1,  # априорная вероятность
        # prior=0 - без сглаживания, границы могут быть неровными
        # prior=1 - умеренное сглаживание (обычное значение)
        # prior=2 - сильное сглаживание, границы более гладкие
        histogram_bins=5,  # количество бинов гистограммы
        # каждый канал цвета разбивается на 5 интервалов
    )

    seeds.iterate(image, 10)  # На каждой итерации SEEDS уточняет границы суперпикселей

    labels = seeds.getLabels()
    mask = seeds.getLabelContourMask(thick_line=False)

    return labels, mask


def lsc_segmentation(image, n_segments=100):
    """
    Linear Spectral Clustering (LSC) применяет спектральное кластерирование в 10-мерном взвешенном пространстве (цвет + позиция), семплируя K семян и минимизируя Normalized Cuts. Обеспечивает равномерные суперпики; в OpenCV — cv.ximgproc.createSuperpixelLSC с region_size и ratio
    """
    h, w = image.shape[:2]

    # вычисление среднего размера суперпикселя
    region_size = int(np.sqrt((w * h) / n_segments))

    lsc = cv2.ximgproc.createSuperpixelLSC(image, region_size=region_size, ratio=0.075)

    lsc.iterate(10)

    labels = lsc.getLabels()
    mask = lsc.getLabelContourMask()

    return labels, mask


def display_results(image, labels, mask, algorithm_name):
    colored_mask = np.zeros_like(image)
    for label in range(np.max(labels) + 1):
        colored_mask[labels == label] = np.random.randint(50, 200, 3)

    contours_img = image.copy()
    if len(mask.shape) == 2:
        contours_img[mask > 0] = [255, 255, 255]

    cv2.imshow("original", image)

    cv2.imshow("mask", colored_mask)

    cv2.imshow("countours", contours_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image = cv2.imread("tree.jpg")

    print("1. SEEDS сегментация")
    print("2. LSC сегментация")

    choice = input("Выберите метод (1 или 2): ")

    if choice == "1":
        labels, mask = seeds_segmentation(image)
        display_results(image, labels, mask, "SEEDS")

    elif choice == "2":
        labels, mask = lsc_segmentation(image)
        display_results(image, labels, mask, "LSC")


main()
