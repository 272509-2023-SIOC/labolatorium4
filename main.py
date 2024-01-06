# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image


# Definicja jądra Lanczosa
def lanczos_kernel(a):
    def kernel(x):
        if x == 0:
            return 1
        elif -a < x < a:
            x_pi = np.pi * x
            return a * np.sin(x_pi) * np.sin(x_pi / a) / (x_pi ** 2)
        else:
            return 0

    return np.vectorize(kernel)

def generate_lanczos_kernel(a, size):
    """Generuje dwuwymiarowe jądro Lanczosa."""
    lanczos_1d = lanczos_kernel(a)
    kernel_1d = np.array([lanczos_1d(i) for i in np.linspace(-a, a, size)])
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()

def calculate_mse(image1, image2):
    """Oblicza średni błąd kwadratowy (MSE) między dwoma obrazami."""
    return np.mean((image1 - image2) ** 2)

# Funkcja do zmniejszania obrazu
def downscale_image(image, kernel):
    return convolve2d(image, kernel, mode='same')


# Funkcja do interpolacji obrazu
def interpolate_image(image, multiplier, kernel):
    new_size = (int(image.shape[0] * multiplier), int(image.shape[1] * multiplier))
    interpolated_image = np.zeros(new_size)

    kernel_size = kernel.shape[0]
    pad_width = kernel_size // 2

    # Dodanie paddingu do obrazu
    padded_image = np.pad(image, pad_width, mode='edge')

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            # Wyliczenie odpowiednich indeksów w oryginalnym obrazie
            x, y = int(i / multiplier), int(j / multiplier)

            # Wybór odpowiedniej części obrazu i zastosowanie jądra
            region = padded_image[x:x+kernel_size, y:y+kernel_size]
            interpolated_value = np.sum(region * kernel)

            interpolated_image[i, j] = interpolated_value

    return interpolated_image


# Parametry i wczytanie obrazu
image = np.array(Image.open('C:\\Users\\Szymon Nowicki\\Downloads\\grey monkey.jpg').convert('L'))
kernel_avg = np.ones((3, 3)) / 9
a = 3
lanczos = lanczos_kernel(a)

# Wygenerowanie jądra Lanczosa
lanczos_kernel_array = generate_lanczos_kernel(a, 7)

# Zmniejszenie obrazu
downscaled_image = downscale_image(image, kernel_avg)

# Powiększanie obrazu
multiplier = 2
kernel_linear = np.ones((3, 3))
interpolated_image_linear = interpolate_image(downscaled_image, multiplier, kernel_linear)

# Powiększanie obrazu z użyciem jądra Lanczosa
multiplier = 2
interpolated_image_lanczos = interpolate_image(downscaled_image, multiplier, lanczos_kernel_array)

# Obliczanie MSE
mse_lanczos = calculate_mse(downscaled_image, interpolated_image_lanczos)
mse_linear = calculate_mse(downscaled_image, interpolated_image_linear)

# Wyświetlenie obrazów
plt.figure(figsize=(24, 6))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 4, 2)
plt.imshow(downscaled_image, cmap='gray')
plt.title('Downscaled Image')
plt.subplot(1, 4, 3)
plt.imshow(interpolated_image_lanczos, cmap='gray')
plt.title('Lanczos Kernel Interpolation')
plt.subplot(1, 4, 4)
plt.imshow(interpolated_image_linear, cmap='gray')
plt.title('Linear Kernel Interpolation')
plt.show()