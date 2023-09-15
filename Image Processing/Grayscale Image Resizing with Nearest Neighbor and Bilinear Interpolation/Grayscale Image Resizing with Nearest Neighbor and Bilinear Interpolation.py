import sys
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
import cv2
from skimage import color
import math

inputImage = cv2.imread('C:/users/yagna/OneDrive/Desktop/yoda.png')
grayImage = color.rgb2gray(inputImage)
width, height = grayImage.shape[:2];

def grayscale_resize(img, param):
    x_new = int(param[0]);
    y_new = int(param[1]);

    resizedImage = np.zeros([x_new, y_new]);

    for i in range(x_new - 1):
        for j in range(y_new - 1):
            resizedImage[i + 1, j + 1] = img[1 + round(i / (x_new / (width - 1))), 1 + round(j / (y_new / (height - 1)))]

    resizedImage = (resizedImage * 255).round().astype(np.uint8)
    print("Nearest-Neighbor Resized Image Values:" + "\n" + str(resizedImage) + "\n")
    return resizedImage

def grayscale_resize_bilinear(img, param):
    x_new = int(param[0]);
    y_new = int(param[1]);

    resizedImageBilinear = np.zeros([x_new, y_new]);

    for i in range(x_new - 1):
        for j in range(y_new - 1):
            # scale values are included in formula
            new_width = -(((i / (x_new / (width - 1))) - math.floor(i / (x_new / (width - 1)))) - 1)
            new_height = -(((j / (y_new / (height - 1))) - math.floor(j / (y_new / (height - 1)))) - 1)
            # for 4 nearest neighbors
            neighbor1_1 = img[1 + math.floor(i / (x_new / (width - 1))), 1 + math.floor(j / (y_new / (height - 1)))]
            neighbor1_2 = img[1 + math.ceil(i / (x_new / (width - 1))), 1 + math.floor(j / (y_new / (height - 1)))]
            neighbor2_1 = img[1 + math.floor(i / (x_new / (width - 1))), 1 + math.ceil(j / (y_new / (height - 1)))]
            neighbor2_2 = img[1 + math.ceil(i / (x_new / (width - 1))), 1 + math.ceil(j / (y_new / (height - 1)))]
            resizedImageBilinear[i + 1, j + 1] = (1 - new_width) * (1 - new_height) * neighbor2_2 + (new_width) * (1 - new_height) * neighbor2_1 + (1 - new_width) * (new_height) * neighbor1_2 + (new_width) * (new_height) * neighbor1_1

    resizedImageBilinear = (resizedImageBilinear * 255).round().astype(np.uint8)
    print("Bilinear Resized Image Values:" + "\n" + str(resizedImageBilinear) + "\n")
    return resizedImageBilinear

# 1(a) Resizing Grayscale Images - Nearest Neighbor
resizedImage = grayscale_resize(grayImage, (800, 700))
cv2.imshow('Nearest-Neighbor Resized Grayscale Image', resizedImage)

# 1(b) Bonus - Resizing Grayscale Images - Bilinear
resizedImageBilinear = grayscale_resize_bilinear(grayImage, (500, 400))
cv2.imshow('Bilinear Resized Grayscale Image', resizedImageBilinear)

cv2.waitKey(0)