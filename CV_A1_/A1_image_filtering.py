'''
Computer vision assignment 1 by Yoseob Kim
A1_image_filtering.py
Image filtering using Gaussian kernel and cross-correlation.
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A1_
'''

import cv2
import numpy as np
import math
import time
import os
from filtering_by_yoseob import *

## ** initial settings to make result directory.
## Reference - https://www.geeksforgeeks.org/python-os-makedirs-method/?ref=lbp
try:
    os.makedirs('result', exist_ok=True)
except OSError as error:
    print("[NOTICE!] '/result' directory cannot be created.")
    print("Please CREATE the DIRECTORY MANUALLY to save created images.")

###
# 1-1. Image Filtering by Cross-Correlation

# functions are implemented in filtering_by_yoseob.py
###


###
# 1.2 The Gaussian Filter

## a, b)
# functions are implemented in filtering_by_yoseob.py
###

## c) Print results of gaussian filter functions
kernel_1d_51 = get_gaussian_filter_1d(5, 1)
kernel_2d_51 = get_gaussian_filter_2d(5, 1)
print('Computer Vision A#1 // Yoseob Kim')
print('Part #1. Image Filtering\n')

print('1-2. c) 1-d gaussian kernel (size=5, sigma=1)')
print(kernel_1d_51, end='\n\n')

print('1-2. c) 2-d gaussian kernel (size=5, sigma=1)')
print(kernel_2d_51, end='\n\n')

## d) Perform 9 different Gaussian filtering
print('1-2. d) perform 9 different gaussian filtering to lenna and shapes.')
k_size = [5, 11, 17]
sigmas = [1, 6, 11]
font = cv2.FONT_HERSHEY_SIMPLEX

# for "lenna.png"
print(' * filtering for "lenna.png" initiate.')
img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
final_filtered_lenna = np.zeros((img_lenna.shape[0]*3, img_lenna.shape[1]*3))

for i in range(3):
    for j in range(3):
        k = k_size[i]
        s = sigmas[j]
        #kernel = get_gaussian_filter_2d(k, s)
        kernel_h = get_gaussian_filter_1d(k, s)
        kernel_v = np.array([kernel_h]).transpose()

        print('filtering... kernel size: {0}, sigma: {1}'.format(k, s), end='')
        #filtered_img_lenna = cross_correlation_2d(img_lenna, kernel).astype(dtype='uint8')
        filtered_img_lenna = cross_correlation_1d(img_lenna, kernel_h)
        filtered_img_lenna = cross_correlation_1d(filtered_img_lenna, kernel_v).astype(dtype='uint8')
        cv2.putText(filtered_img_lenna, '{0}x{0}, s={1}'.format(k, s), (30, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        for x in range(img_lenna.shape[0] * i, img_lenna.shape[0] * (i+1)):
            u = x - img_lenna.shape[0] * i
            for y in range(img_lenna.shape[1] * j, img_lenna.shape[1] * (j+1)):
                v = y - img_lenna.shape[1] * j
                final_filtered_lenna[x][y] = filtered_img_lenna[u][v] / 255
        print(' ---> done.')

#print(final_filtered_lenna)
#print(final_filtered_lenna.shape)
final_filtered_lenna = cv2.normalize(final_filtered_lenna, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_1_gaussian_filtered_lenna.png', final_filtered_lenna)
print(' * filtered image of "lenna.png" saved to ./result/ directory.')
cv2.imshow('Gaussian filtering of "Lenna.png" by Yoseob Kim', final_filtered_lenna)
print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.waitKey(0)

# for "shapes.png"
print(' * filtering for "shapes.png" initiate.')
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)
final_filtered_shapes = np.zeros((img_shapes.shape[0] * 3, img_shapes.shape[1] * 3))

for i in range(3):
    for j in range(3):
        k = k_size[i]
        s = sigmas[j]
        #kernel = get_gaussian_filter_2d(k, s)
        kernel_h = get_gaussian_filter_1d(k, s)
        kernel_v = np.array([kernel_h]).transpose()

        print('filtering... kernel size: {0}, sigma: {1}'.format(k, s), end='')
        #filtered_img_shapes = cross_correlation_2d(img_shapes, kernel).astype(dtype='uint8')
        filtered_img_shapes = cross_correlation_1d(img_shapes, kernel_h)
        filtered_img_shapes = cross_correlation_1d(filtered_img_shapes, kernel_v).astype(dtype='uint8')
        cv2.putText(filtered_img_shapes, '{0}x{0}, s={1}'.format(k, s), (30, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        for x in range(img_shapes.shape[0] * i, img_shapes.shape[0] * (i + 1)):
            u = x - img_shapes.shape[0] * i
            for y in range(img_shapes.shape[1] * j, img_shapes.shape[1] * (j + 1)):
                v = y - img_shapes.shape[1] * j
                final_filtered_shapes[x][y] = filtered_img_shapes[u][v] / 255
        print(' ---> done.')

#print(final_filtered_shapes)
#print(final_filtered_shapes.shape)
final_filtered_shapes = cv2.normalize(final_filtered_shapes, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_1_gaussian_filtered_shapes.png', final_filtered_shapes)
print(' * filtered image of "shapes.png" saved to ./result/ directory.')

cv2.imshow('Gaussian filtering of "Shapes.png" by Yoseob Kim', final_filtered_shapes)
print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.waitKey(0)


## e) comparison btw 1-d kernel correlation superpositioning and 2-d kernel filtering.

e_k_size = 17
e_sigma = 6
kernel_1d_h = get_gaussian_filter_1d(e_k_size, e_sigma)
kernel_2d = get_gaussian_filter_2d(e_k_size, e_sigma)
kernel_1d_v = np.array([kernel_1d_h]).transpose()

print('1-2. e) comparison btw 1-d kernel correlation superpositioning and 2-d kernel filtering.')
print(' *** kernel size = {0}, sigma = {1}'.format(e_k_size, e_sigma))

# for "lenna.png"
print(' * for "lenna.png"')
print('filtering... (1-d horizontal kernel)', end='')

start_time = time.process_time()
filtered_img_1d_lenna = cross_correlation_1d(img_lenna, kernel_1d_h)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

print('filtering... (1-d vertical kernel)', end='')
stqrt_time = time.process_time()
filtered_img_1d_lenna = cross_correlation_1d(filtered_img_1d_lenna, kernel_1d_v)
elapsed_time_ = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time_)
elapsed_time_ += elapsed_time
print(' == whole 1-d kernel filtering elapsed time:', elapsed_time_)

#cv2.imshow('1-d', filtered_img_1d_lenna.astype(dtype='uint8'))
#cv2.waitKey(0)

print('filtering... (2-d kernel), ** time consuming alert (about 62.16sec) **') # , end='')
print(' [about 20 dots will be shown to be done]')
start_time = time.process_time()
filtered_img_2d_lenna = cross_correlation_2d(img_lenna, kernel_2d)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)
print(' == comparison of elapsed time (1-d time) - (2-d time):', elapsed_time_-elapsed_time)

#cv2.imshow('2-d', filtered_img_2d_lenna.astype(dtype='uint8'))
#cv2.waitKey(0)

subtracted_img = np.zeros((img_lenna.shape[0], img_lenna.shape[1]))
_sum = 0
for i in range(img_lenna.shape[0]):
    for j in range(img_lenna.shape[1]):
        subtracted_img[i][j] = filtered_img_1d_lenna[i][j] - filtered_img_2d_lenna[i][j]
        _sum += abs(subtracted_img[i][j])

cv2.imshow("pixel-wise difference map of lenna", subtracted_img)
print(' *** sum of absolute intensity difference:', _sum) # , end='\n\n')
print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.waitKey(0)

# for "shapes.png"
print(' * for "shapes.png"')
print('filtering... (1-d horizontal kernel)', end='')

start_time = time.process_time()
filtered_img_1d_shapes = cross_correlation_1d(img_shapes, kernel_1d_h)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

print('filtering... (1-d vertical kernel)', end='')
stqrt_time = time.process_time()
filtered_img_1d_shapes = cross_correlation_1d(filtered_img_1d_shapes, kernel_1d_v)
elapsed_time_ = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time_)
elapsed_time_ += elapsed_time
print(' == whole 1-d kernel filtering elapsed time:', elapsed_time_)

#cv2.imshow('1-d', filtered_img_1d_shapes.astype(dtype='uint8'))
#cv2.waitKey(0)

print('filtering... (2-d kernel), ** time consuming alert (about 50.01sec) **') # , end='')
print(' [about 20 dots will be shown to be done]')
start_time = time.process_time()
filtered_img_2d_shapes = cross_correlation_2d(img_shapes, kernel_2d)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)
print(' == comparison of elapsed time (1-d time) - (2-d time):', elapsed_time_-elapsed_time)

#cv2.imshow('2-d', filtered_img_2d_shapes.astype(dtype='uint8'))
#cv2.waitKey(0)

subtracted_img = np.zeros((img_shapes.shape[0], img_shapes.shape[1]))
_sum = 0
for i in range(img_shapes.shape[0]):
    for j in range(img_shapes.shape[1]):
        subtracted_img[i][j] = filtered_img_1d_shapes[i][j] - filtered_img_2d_shapes[i][j]
        _sum += abs(subtracted_img[i][j])

cv2.imshow("pixel-wise difference map of shapes", subtracted_img)
print(' *** sum of absolute intensity difference:', _sum) # , end='\n\n')
print(' ## notice: P#1 done. press any key (on image window) to finish.\n')


cv2.waitKey(0)
cv2.destroyAllWindows()

'''
size = 8
img_a = np.arange(-(size//2), (size//2) + 1)
img = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        img[i][j] = img_a[i] * img_a[j]

print('original img')
print(img)
kernel = get_gaussian_filter_1d(5, 1)
print('padded img - horizontal')
cross_correlation_1d(img, kernel)

print('padded img - vertical')
kernel = np.array([kernel]).transpose()
cross_correlation_1d(img, kernel)
#cross_correlation_2d(img, kernel)
'''