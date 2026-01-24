"""
Задание #1
Обработать любое изображение:
 - Нарисовать горизонтальную линию по центру
 - Повернуть на 75°
 - Уменьшить мастшаб на 90%
 - Написать текст в произвольном месте
"""

import cv2


def show_img():
    cv2.imshow("cat", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("cat.png")
H, W, C = img.shape
CENTER = (W // 2, H // 2)

# вывод изображения
show_img()

# вывод линии
RGB_LINE = (0, 255, 0)
LINE_PX = 5
cv2.line(img, (0, H // 2), (W, H // 2), RGB_LINE, LINE_PX)
show_img()

# поворот
ANGLE = 75
matrix = cv2.getRotationMatrix2D(CENTER, ANGLE, 1.0)
img = cv2.warpAffine(img, matrix, (W, H))
show_img()

# уменьшение
SCALE = 0.9
img = cv2.resize(img, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
show_img()

# вывод текста
RGB_TEXT = (30, 30, 160)
TEXT_PX = 3
cv2.putText(img, "blue cat", CENTER, cv2.FONT_HERSHEY_SIMPLEX, 1, RGB_TEXT, TEXT_PX)
show_img()
