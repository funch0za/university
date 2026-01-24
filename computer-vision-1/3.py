"""
Ввод изображения
BGR → HSV (перевод в другую цветовую модель)
Увеличить четкость изображения (например, с помощью фильтра увеличения резкости или CLAHE)
Выделить оттенки HSV (определить диапазон цвета в HSV)
In Range получить маску для цвета (cv2.inRange() для выделения объекта по цвету)
Применить маску к оригинальному изображению (битовая операция cv2.bitwise_and)
Различие по Гауссу (применить Гауссово размытие и/или детектор границ, например cv2.GaussianBlur + cv2.Canny)
Вывести/сохранить результат (показать изображения с помощью cv2.imshow или сохранить cv2.imwrite)
"""

import cv2
import numpy as np


def show_img(img):
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1) ввод изображения
img = cv2.imread("tree.jpg")
show_img(img)

# 2) перевод в другую цветовую модель
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
show_img(img_hsv)

# 3) увеличить четкость изображения
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

img_sharpened = cv2.filter2D(img_hsv, -1, kernel)
show_img(img_sharpened)

# 4) диапазон HSV для зеленого
lower_green = np.array([35, 50, 50])  # светлый зеленый
upper_green = np.array([85, 255, 255])  # темный зеленый

# другие похожие оттенки
lower_other = np.array([25, 30, 30])
upper_other = np.array([35, 255, 255])

# 5) создание маски
mask_green = cv2.inRange(img_sharpened, lower_green, upper_green)
mask_other = cv2.inRange(img_sharpened, lower_other, upper_other)
mask = cv2.bitwise_or(mask_green, mask_other)

show_img(mask)

# 6) применение маски
img_bgr_sharpened = cv2.cvtColor(img_sharpened, cv2.COLOR_HSV2BGR)
masked_result = cv2.bitwise_and(img_bgr_sharpened, img_bgr_sharpened, mask=mask)
show_img(masked_result)

# 7) размытие
denoised = cv2.GaussianBlur(masked_result, (5, 5), 0)

# 8) вывод
show_img(denoised)
