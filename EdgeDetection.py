import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image

# 𝐿𝑜𝐺(𝑥,𝑦)= −1/(𝜋𝜎^4)(1−(𝑥^2 + 𝑦^2)/(2𝜎^2))𝑒^(−(𝑥^2 + 𝑦^2)/(2𝜎^2))
def LoG1(x, y, sigma):
    exPow = -(x*x+y*y)/(2*sigma*sigma)
    t1 = -1/(math.pi * sigma**4)
    t2 = 1 + exPow
    t3 = math.exp(exPow)
    return t1 * t2 * t3

def LoG(sigma):
    s = 2 * math.ceil(3 * sigma) + 1
    kernel = [[0 for x in range(s)] for y in range(s)]
    for x in range(-s//2, s//2):
        for y in range(-s//2, s//2):
            kernel[x+s//2][y+s//2] = LoG1(x, y, sigma)
    return kernel

def convolute(numpy_image, kernel):
    (n, m) = numpy_image.shape
    half_kernel_size = len(kernel)//2
    convoluted_image = np.zeros(numpy_image.shape)
    for i in range(half_kernel_size, n-half_kernel_size):
        for j in range(half_kernel_size, m-half_kernel_size):
            new_val = 0
            for i_kernel in range(len(kernel)):
                for j_kernel in range(len(kernel)):
                    i_org = i+i_kernel-half_kernel_size
                    j_org = j+j_kernel-half_kernel_size
                    new_val+=kernel[i_kernel][j_kernel] * numpy_image[i_org][j_org]
            convoluted_image[i][j] = new_val
    return convoluted_image

def zeroCrossing(second_derivative, first_derivative):
    zero_crossings = np.zeros(second_derivative.shape)
    (n, m) = second_derivative.shape
    r = [1, 1, 1, 0, 0, -1, -1, -1]
    c = [-1, 0, 1, 1, -1, -1, 0, 1]
    for i in range(n):
        for j in range(m):
            pos = False
            neg = False
            for itr in range(8):
                newI = i+r[itr]
                newJ = j+r[itr]
                if(newI < 0 or newI >= n or newJ < 0 or newJ>=m):
                    continue
                if(second_derivative[newI][newJ]>0):
                    pos = True
                if(second_derivative[newI][newJ]<0):
                    neg = True
            if(pos and neg and first_derivative[i][j]==1):
                zero_crossings[i][j] = 1
    return zero_crossings

def normalize(numpy_image):
    (n, m) = numpy_image.shape
    normalized_image = np.zeros(numpy_image.shape)
    for i in range(n):
        for j in range(m):
            normalized_image[i][j] = numpy_image[i][j]/255
    return normalized_image

def prewitt(numpy_image, threshold):
    h1 = [[1,1,1],[0,0,0],[-1,-1,1]]
    h2 = [[-1,0,1],[-1,0,1],[-1,0,1]]
    numpy_image = normalize(numpy_image)
    h1_image = convolute(numpy_image, h1)
    h2_image = convolute(numpy_image, h2)
    edge_image = np.zeros(numpy_image.shape)
    for i in range(h1_image.shape[0]):
        for j in range(h1_image.shape[1]):
            edge_image[i][j] = math.sqrt(h1_image[i][j]**2 + h2_image[i][j]**2)
    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            if(edge_image[i][j] > threshold):
                edge_image[i][j] = 1
            else:
                edge_image[i][j] = 0
    return edge_image   

def LoG_edge_detection(image, sigma, threshold):
    numpy_image = np.array(image)
    log_mask = LoG(sigma)
    second_derivative = convolute(numpy_image, log_mask)
    first_derivative = prewitt(numpy_image, threshold)
    return zeroCrossing(second_derivative, first_derivative) * 255

def perform_log_and_save_as(image, sigma, threshold, name):
    final = LoG_edge_detection(image, sigma, threshold)
    final_image = Image.fromarray(final).convert("L")
    final_image.save(name)
                
def main():
    image = Image.open("./Cameraman.tif")
    perform_log_and_save_as(image, 2, 0.1,  "LoG_2.jpg")
    perform_log_and_save_as(image, 3, 0.1,  "LoG_3.jpg")
    perform_log_and_save_as(image, 4, 0.1,  "LoG_4.jpg")


if __name__ == '__main__':
    main()

