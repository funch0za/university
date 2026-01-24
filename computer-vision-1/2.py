"""
Задание #2
Улучшить качество любого плохого изображения (шумы, размытие...) с помощью фильтров
"""

import cv2
import numpy as np


def show_img(img):
    cv2.imshow("cat", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("cat2.png")
show_img(img)


# удаление шума

denoised = cv2.GaussianBlur(img, (15, 15), 0)
show_img(denoised)

# резкость
gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
img = cv2.addWeighted(denoised, 2.4, gaussian, -1.5, 0)
show_img(img)


# темные области становятся более темными
norm = img.astype(np.float32) / 255.0
dark = np.power(norm, 1.3)
img = np.uint8(255 * dark)  # denorm
show_img(img)


# после затемнения можно понизить яркость пикселей, попадают в оттенки серого
result = img.copy().astype(np.float32)
b, g, r = cv2.split(img)
diff1 = np.abs(g.astype(np.float32) - r.astype(np.float32))
diff2 = np.abs(b.astype(np.float32) - g.astype(np.float32))

# маска серых пикселей (разница < 20)
gray_mask = (diff1 < 20) & (diff2 < 20)

# затемняем серые пиксели
result[gray_mask] *= 0.6  # -40% яркости
img = np.clip(result, 0, 255).astype(np.uint8)
show_img(img)

# увеличение контрста
contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
show_img(img)
