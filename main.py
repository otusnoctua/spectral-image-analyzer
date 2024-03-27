import numpy as np
import cv2
import matplotlib.pyplot as plt


def _fftImage(img_gray, rows, cols):
    rPadded = cv2.getOptimalDFTSize(rows)
    cPadded = cv2.getOptimalDFTSize(cols)
    imgPadded = np.zeros((rPadded, cPadded), dtype=np.float32)
    imgPadded[:rows, :cols] = img_gray
    img_fft = cv2.dft(imgPadded, flags=cv2.DFT_COMPLEX_OUTPUT)
    return img_fft


def _stdFftImage(img_gray, rows, cols):
    # Преобразование Фурье
    fimg = np.copy(img_gray)
    fimg = fimg.astype(np.float32)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2:
                fimg[r][c] = -1 * img_gray[r][c]
    img_fft = _fftImage(fimg, rows, cols)
    return img_fft


def _graySpectr(fft_img):
    # Выделение спкетра
    real_ch = np.power(fft_img[:, :, 0], 2.0)
    imaginary_ch = np.power(fft_img[:, :, 1], 2.0)
    amplitude = np.sqrt(real_ch + imaginary_ch)
    spectr = np.log(amplitude + 1.0)
    spectr = cv2.normalize(spectr, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    spectr *= 255
    return amplitude, spectr


def _getImage(main_img):
    # Вычисение кол-ва строк и столбцов в исходеном изображении
    rows, cols = main_img.shape[:2]
    # Быстрое преобразование фурье
    img_fft = _stdFftImage(main_img, rows, cols)
    amplitude, spectre = _graySpectr(img_fft)
    minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(
        amplitude)

    spectre_img = spectre.astype(np.uint8)
    return spectre_img


# построение спектра

if __name__ == '__main__':
    file = cv2.imread('img/dot.png', 0)
    result_img = _getImage(file)
    cv2.imwrite('result_dot.png', result_img)

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(file)
    plt.axis('off')
    plt.title("First")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(result_img)
    plt.axis('off')
    plt.title("Second")
    plt.show()
