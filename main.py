# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.signal import convolve

def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(x - 1)

def f3(x):
    return np.sign(np.sin(8 * x))

def function(x, function_id):
    if function_id == 1:
        return f1(x)
    elif function_id == 2:
        return f2(x)
    elif function_id == 3:
        return f3(x)
    else:
        raise ValueError("Niepoprawny identyfikator funkcji")

x = np.linspace(-np.pi, np.pi, 100)

N = len(x)

kernels = [np.ones(3), np.array([1, 2, 1]), np.array([1, 0, -1]), np.array([-1, 0, 1]), np.array([-1, 2, -1])]

point_counts = [N, 2 * N, 4 * N, 10 * N]

for function_id in [1, 2, 3]:
    print(f"Function: {function.__name__}")
    for kernel in kernels:
        print(f"Kernel: {kernel}")
        for point_count in point_counts:
            x_interpolated = np.linspace(-np.pi, np.pi, point_count)
            y_original = function(x_interpolated, function_id)
            y_interpolated = convolve(y_original, kernel, mode='same')
            kernel_sum = np.sum(kernel)
            if kernel_sum != 0:
                y_interpolated /= kernel_sum
            mse = np.mean((y_original - y_interpolated) ** 2)
            print(f"Function: {function_id}, Point count: {point_count}, MSE: {mse}")

def reduce_image(image, scale_factor):
    kernel = np.ones((scale_factor, scale_factor)) / (scale_factor ** 2)
    return convolve(image, kernel, mode='same')

def enlarge_image(image, scale_factor, interpolation_function):
    new_height = image.shape[0] * scale_factor
    new_width = image.shape[1] * scale_factor

    enlarged_image = np.zeros((new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            x = i / scale_factor
            y = j / scale_factor
            if 0 <= x < image.shape[0] - 1 and 0 <= y < image.shape[1] - 1:
                enlarged_image[i, j] = interpolation_function(image, x, y)
    return enlarged_image

def nearest_neighbor_interpolation(image, x, y):
    x_nearest = int(round(x))
    y_nearest = int(round(y))
    return image[x_nearest, y_nearest]

image = Image.open("C:\\Users\\01szy\\OneDrive\\Obrazy\\mandril.bmp").convert("L")
image = np.array(image)
scale_factor = 2

reduced_image = reduce_image(image, scale_factor)

enlarged_image = enlarge_image(reduced_image, scale_factor, nearest_neighbor_interpolation)

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(reduced_image, cmap='gray')
plt.title('Reduced Image')

plt.subplot(133)
plt.imshow(enlarged_image, cmap='gray')
plt.title('Enlarged Image')

plt.show()

mse = np.mean((image - enlarged_image) ** 2)
print(f'Mean Squared Error (MSE) between original and enlarged image: {mse}')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
