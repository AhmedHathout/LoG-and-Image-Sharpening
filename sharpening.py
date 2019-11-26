from PIL import Image
import numpy as np
import math

kernel = np.array([[-1, -1, -1],
                   [-1, +8, -1],
                   [-1, -1, -1]])

intensity_scaling_factor = 0.1

def convolute(numpy_image, kernel):
    half_kernel_size = len(kernel) // 2

    convoluted_image = np.zeros(numpy_image.shape)
    for i_image in range(half_kernel_size, len(numpy_image) - half_kernel_size):
        for j_image in range(half_kernel_size, len(numpy_image) - half_kernel_size):

            new_pixel = 0
            for i_kernel in range(len(kernel)):
                i_window = i_image + i_kernel - half_kernel_size
                for j_kernel in range(len(kernel)):
                    j_window = j_image + j_kernel - half_kernel_size
                    new_pixel += kernel[i_kernel][j_kernel] * numpy_image[i_window][j_window]

            convoluted_image[i_image][j_image] = new_pixel

    return convoluted_image

def scale_intensities(numpy_image, intensity_scaling_factor):
    new_intensities = numpy_image * intensity_scaling_factor

    for i in range(len(new_intensities)):
        for j in range(len(new_intensities[i])):
            if new_intensities[i][j] > 50:
                new_intensities[i][j] = 50
            if new_intensities[i][j] < -50:
                new_intensities[i][j] = -50

    return new_intensities

def add_2_images(first_image, second_image):
    error_message = lambda first_length, second_length:   "Different dimensions: " \
                                                          + str(len(first_image)) \
                                                          + ", " + str(len(second_image))

    new_image = np.zeros(shape=first_image.shape)
    if len(first_image) != len(second_image):
        raise AssertionError(error_message(len(first_image), len(second_image)))
    for i in range(len(first_image)):
        if len(first_image[i]) != len(second_image[i]):
            raise AssertionError(len(first_image[i]), len(second_image[i]))

        for j in range(len(first_image[i])):
            new_image[i][j] = first_image[i][j] + second_image[i][j]
            if new_image[i][j] > 255:
                new_image[i][j] = 255

    return new_image

def sharpen(numpy_image, kernel, intensity_scaling_factor):
    convoluted_image = convolute(numpy_image, kernel)
    scaled_image = scale_intensities(convoluted_image, intensity_scaling_factor)
    sharpened_image = add_2_images(numpy_image, scaled_image)
    Image.fromarray(add_2_images(numpy_image, convoluted_image)).convert("L").save("Sharpened2.jpg")

    return Image.fromarray(sharpened_image).convert("L")

def main():
    image = Image.open("./Cameraman.tif")

    numpy_image = np.array(image)

    sharpened_image = sharpen(numpy_image, kernel, intensity_scaling_factor)
    sharpened_image.save("Sharpened.jpg")

if __name__ == '__main__':
    main()