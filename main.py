from tkinter import Button, Tk

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Быстрое преобразование Фурье
def _fft(img_gray, rows, cols):
    rPadded = cv2.getOptimalDFTSize(rows)
    cPadded = cv2.getOptimalDFTSize(cols)
    imgPadded = np.zeros((rPadded, cPadded), dtype=np.float32)
    imgPadded[:rows, :cols] = img_gray
    img_fft = cv2.dft(imgPadded, flags=cv2.DFT_COMPLEX_OUTPUT)
    return img_fft


def _std_fft_image(img_gray, rows, cols):
    fimg = np.copy(img_gray)
    fimg = fimg.astype(np.float32)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2:
                fimg[r][c] = -1 * img_gray[r][c]
    img_fft = _fft(fimg, rows, cols)
    return img_fft


def _grayscale_spectrum(fft_img):
    real_ch = np.power(fft_img[:, :, 0], 2.0)
    imaginary_ch = np.power(fft_img[:, :, 1], 2.0)
    amplitude = np.sqrt(real_ch + imaginary_ch)
    spectrum = np.log(amplitude + 1.0)
    spectrum = cv2.normalize(spectrum, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    spectrum *= 255
    return spectrum


def _build_spectrum_image(main_img):
    # Вычисение кол-ва строк и столбцов в исходном изображении
    rows, cols = main_img.shape[:2]
    # Быстрое преобразование фурье
    img_fft = _std_fft_image(main_img, rows, cols)
    spectre = _grayscale_spectrum(img_fft)
    spectre_img = spectre.astype(np.uint8)
    return spectre_img


def build_plot(src_img, res_img):
    fig = plt.figure(figsize=(10, 7))
    fig.canvas.manager.set_window_title('Построение спектра изображения')
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(src_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Исходное изображение")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(res_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Спектр в градациях серого")
    plt.show()


def build_dot():
    file = cv2.imread('img/dot.jpg', 0)
    result_img = _build_spectrum_image(file)
    build_plot(file, result_img)


def build_checkboard():
    file = cv2.imread('img/checkboard.png', 0)
    result_img = _build_spectrum_image(file)
    build_plot(file, result_img)


def build_circle():
    file = cv2.imread('img/circle.jpg', 0)
    result_img = _build_spectrum_image(file)
    build_plot(file, result_img)


def build_square():
    file = cv2.imread('img/square.png', 0)
    result_img = _build_spectrum_image(file)
    build_plot(file, result_img)


def build_ltl():
    file = cv2.imread('img/line_tilted_left.png', 0)
    result_img = _build_spectrum_image(file)
    build_plot(file, result_img)


def build_lv():
    file = cv2.imread('img/line_vertical.png', 0)
    result_img = _build_spectrum_image(file)
    build_plot(file, result_img)


def build_net():
    file = cv2.imread('img/net.jpg', 0)
    result_img = _build_spectrum_image(file)
    build_plot(file, result_img)


def build_cs():
    file = cv2.imread('img/checkboard_sphere.png', 0)
    result_img = _build_spectrum_image(file)
    build_plot(file, result_img)


if __name__ == '__main__':
    root = Tk()
    root.title("Спектр")

    dot = Button(root, text="Точка", fg="black", command=build_dot)
    dot.grid(row=1, column=1, sticky="ew")

    circle = Button(root, text="Окружность", fg="black", command=build_circle)
    circle.grid(row=2, column=1, sticky="ew")

    square = Button(root, text="Квадрат", fg="black", command=build_square)
    square.grid(row=3, column=1, sticky="ew")

    ltl = Button(root, text="Прямая с наклоном влево", fg="black", command=build_ltl)
    ltl.grid(row=4, column=1, sticky="ew")

    lv = Button(root, text="Вертикальная прямая", fg="black", command=build_lv)
    lv.grid(row=5, column=1, sticky="ew")

    net = Button(root, text="Сетка прямых", fg="black", command=build_net)
    net.grid(row=6, column=1, sticky="ew")

    checkboard = Button(root, text="Шахматная доска", fg="black", command=build_checkboard)
    checkboard.grid(row=7, column=1, sticky="ew")

    cs = Button(root, text="Отражение шахматной доски в сфере", fg="black", command=build_cs)
    cs.grid(row=8, column=1, sticky="ew")

    root.mainloop()
