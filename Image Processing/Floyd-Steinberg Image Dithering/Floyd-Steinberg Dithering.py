import sys
import numpy as np
from math import sqrt
# np.set_printoptions(threshold=sys.maxsize)
import cv2

def dithering_gray(img, threshold):
    height = img.shape[0]
    width = img.shape[1]

    for y in range(0, height - 1):
        for x in range(1, width - 1):

            old_pixel_val = img[y, x]
            new_pixel_val = np.round(threshold * old_pixel_val / 255) * (255 / threshold)
            img[y, x] = new_pixel_val

            error = old_pixel_val - new_pixel_val

            if x < width - 1:
                img[y, x + 1] += np.clip(error * 7 / 16, 0, 255)
            if y < height - 1:
                if x > 0:
                    img[y + 1, x - 1] += np.clip(error * 3 / 16, 0, 255)
                img[y + 1, x] += np.clip(error * 5 / 16, 0, 255)
                if x < width - 1:
                    img[y + 1, x + 1] += np.clip(error * 1 / 16, 0, 255)

    print("Dithered Image:" + "\n" + str(img) + "\n")
    return img

def find_closest(levelsArray, valueToBeCompared):
    closestValueIndex = np.abs(levelsArray - valueToBeCompared).argmin()
    return levelsArray[closestValueIndex]

def grayscale_dither_multilevel(img, threshold):
    height = img.shape[0]
    width = img.shape[1]

    for y in range(0, height - 1):
        for x in range(1, width - 1):

            old_pixel_val = img[y, x]
            new_pixel_val = find_closest(threshold, old_pixel_val)
            img[y, x] = new_pixel_val

            error = old_pixel_val - new_pixel_val

            if x < width - 1:
                img[y, x + 1] = np.clip(img[y, x + 1] + error * 7 / 16, 0, 255)
            if y < height - 1:
                if x > 0:
                    img[y + 1, x - 1] = np.clip(img[y + 1, x - 1] + error * 3 / 16, 0, 255)
                img[y + 1, x] = np.clip(img[y + 1, x] + error * 5 / 16, 0, 255)
                if x < width - 1:
                    img[y + 1, x + 1] = np.clip(img[y + 1, x + 1] + error * 1 / 16, 0, 255)

    print("Dithered Multi-Level Image:" + "\n" + str(img) + "\n")
    return img

def color_dither_multilevel(img, colors):

    height = img.shape[0]
    weight = img.shape[1]

    for y in range(0, height - 1):
        for x in range(1, weight - 1):

            old_pixel_val_b = img[y, x, 0]
            old_pixel_val_g = img[y, x, 1]
            old_pixel_val_r = img[y, x, 2]

            threshold = findClosestColor((old_pixel_val_b, old_pixel_val_g, old_pixel_val_r), colors)

            img[y, x, 0] = threshold[0]
            img[y, x, 1] = threshold[1]
            img[y, x, 2] = threshold[2]

            error_b = old_pixel_val_b - threshold[0]
            error_g = old_pixel_val_g - threshold[1]
            error_r = old_pixel_val_r - threshold[2]

            img[y, x + 1, 0] = np.clip(img[y, x + 1, 0] + error_b * 7 / 16, 0, 255)
            img[y, x + 1, 1] = np.clip(img[y, x + 1, 1] + error_g * 7 / 16, 0, 255)
            img[y, x + 1, 2] = np.clip(img[y, x + 1, 2] + error_r * 7 / 16, 0, 255)

            img[y + 1, x - 1, 0] = np.clip(img[y + 1, x - 1, 0] + error_b * 3 / 16, 0, 255)
            img[y + 1, x - 1, 1] = np.clip(img[y + 1, x - 1, 1] + error_g * 3 / 16, 0, 255)
            img[y + 1, x - 1, 2] = np.clip(img[y + 1, x - 1, 2] + error_r * 3 / 16, 0, 255)

            img[y + 1, x, 0] = np.clip(img[y + 1, x, 0] + error_b * 5 / 16, 0, 255)
            img[y + 1, x, 1] = np.clip(img[y + 1, x, 1] + error_g * 5 / 16, 0, 255)
            img[y + 1, x, 2] = np.clip(img[y + 1, x, 2] + error_r * 5 / 16, 0, 255)

            img[y + 1, x + 1, 0] = np.clip(img[y + 1, x + 1, 0] + error_b * 1 / 16, 0, 255)
            img[y + 1, x + 1, 1] = np.clip(img[y + 1, x + 1, 1] + error_g * 1 / 16, 0, 255)
            img[y + 1, x + 1, 2] = np.clip(img[y + 1, x + 1, 2] + error_r * 1 / 16, 0, 255)

    print("Dithered Color Multi-Level Image:" + "\n" + str(img) + "\n")

    return img

def findClosestColor(colorTuple, colors):
    closest = (0, 0, 0)
    min_difference = 1000
    for columns in colors:
        difference = sqrt((columns[0] - colorTuple[0]) ** 2 + (columns[1] - colorTuple[1]) ** 2 + (columns[2] - colorTuple[2]) ** 2)
        if(difference < min_difference):
            min_difference = difference
            closest = columns
    return closest

inputImage = cv2.imread('C:/users/yagna/OneDrive/Desktop/yoda.png')

# 2(a) Floyd-Steinberg Dithering
grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
ditheredImage = dithering_gray(grayImage, 20)
cv2.imshow('Dithered Grayscale Image', ditheredImage)

# 2(b) Floyd-Steinberg Dithering levels
grayscale_dither_multileveledImage = grayscale_dither_multilevel(grayImage, [0, 85, 170, 255])
cv2.imshow('Dithered Multi-Level Grayscale Image', grayscale_dither_multileveledImage)

# 2(bonus) Floyd-Steinberg Dithering color levels
dithered_color_multilevel = color_dither_multilevel(inputImage.copy(), [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
cv2.imshow('Dithered Color Image', dithered_color_multilevel)

cv2.waitKey(0)