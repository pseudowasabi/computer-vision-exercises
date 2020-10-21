'''
Computer vision assignment 1 by Yoseob Kim
A1_corner_detection.py
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A1_
'''

import cv2
import numpy as np
import math
import time
import operator
from filtering_by_yoseob import *

# ** initial settings


###
# 3-2. Corner response (define function - a, b, c and d)
###

def compute_corner_response(img):
    ## a) apply Sobel filters
    # Sobel filtering function is implemented in filtering_by_yoseob.py
    # sobel_img_x, sobel_img_y is derivatives along x and y direction respectively.
    sobel_img_x, sobel_img_y = my_sobel_filtering(img)

    # padded with 0 value to operate corner response
    sobel_img_x = image_padding_2d(sobel_img_x, 2, 1)   # (img, padd_width, type=1)
    sobel_img_y = image_padding_2d(sobel_img_y, 2, 1)   # type=1 means just padd with zeros.

    ## b) compute second moment matrix M
    uni_window = np.ones((5, 5))
    patch_img_x = np.zeros((5, 5))
    patch_img_y = np.zeros((5, 5))
    size0 = img.shape[0]
    size1 = img.shape[1]

    ## c) variables for computing corner responses
    R = np.zeros((size0, size1))
    _k = 0.04
    _max_val = 0

    for x in range(size0):
        for y in range(size1):
            # i. subtract mean of each image patch
            _sum_x = 0.
            _sum_y = 0.
            for i in range(x, x+5):
                for j in range(y, y+5):
                    patch_img_x[i-x][j-y] = sobel_img_x[i][j]
                    _sum_x = operator.__add__(_sum_x, sobel_img_x[i][j])
                    patch_img_y[i-x][j-y] = sobel_img_y[i][j]
                    _sum_y = operator.__add__(_sum_y, sobel_img_y[i][j])
            _avg_x = operator.__truediv__(_sum_x, 25)
            _avg_y = operator.__truediv__(_sum_y, 25)

            sum_of_ix_ix = 0.
            sum_of_ix_iy = 0.
            sum_of_iy_iy = 0.
            for i in range(5):
                for j in range(5):
                    patch_img_x[i][j] = operator.__sub__(patch_img_x[i][j], _avg_x)
                    patch_img_y[i][j] = operator.__sub__(patch_img_y[i][j], _avg_y)

                    sum_of_ix_ix = operator.__add__(sum_of_ix_ix, operator.__mul__(patch_img_x[i][j], patch_img_x[i][j]))
                    sum_of_ix_iy = operator.__add__(sum_of_ix_iy, operator.__mul__(patch_img_x[i][j], patch_img_y[i][j]))
                    sum_of_iy_iy = operator.__add__(sum_of_iy_iy, operator.__mul__(patch_img_y[i][j], patch_img_y[i][j]))

            # ii. get second moment matrix
            # since we use uniform window, just calculated the summation of ix_ix, ix_iy, iy_iy respectively (above).
            M = np.array([[sum_of_ix_ix, sum_of_ix_iy], [sum_of_ix_iy, sum_of_iy_iy]])
            eigenvalues, _ = np.linalg.eig(M)
            #print(eigen_values)

            e1 = eigenvalues[0]
            e2 = eigenvalues[1]
            R[x][y] = e1 * e2 - _k * ((e1 + e2) ** 2)

            ## d) normalize responses
            # i. negative values to 0, otherwise normalize to range [0, 1]
            if R[x][y] < 0:
                R[x][y] = 0
            if R[x][y] > _max_val:
                _max_val = R[x][y]

    normalizer = 1.
    if _max_val != 0:
        normalizer = 1 / _max_val
    for x in range(size0):
        for y in range(size1):
            R[x][y] = operator.__mul__(R[x][y], normalizer)
    return R


def non_maximum_suppression_win(R, winSize):
    pass

    ## c) compute local maximas by NMS

    ## d) ...


###
# 3-1. Gaussian filtering
###

img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

filtered_img_shapes = my_gaussian_filtering(img_shapes, 7, 1.5)
filtered_img_lenna = my_gaussian_filtering(img_lenna, 7, 1.5)


###
# 3-2. Corner response (execute requirements - ...)
###


R_shapes = compute_corner_response(filtered_img_shapes)
cv2.imshow("corner response of shapes", R_shapes)
cv2.waitKey(0)
R_lenna = compute_corner_response(filtered_img_lenna)
cv2.imshow("corner response of lenna", R_lenna)
cv2.waitKey(0)



###
# 3-3. Thresholding, Non-maximum suppression (execute requirements)
###

## a) change corner response to green

## b) show image and save

## c) nms

_winSize = 11
suppressed_R_shapes = non_maximum_suppression_win(R_shapes, _winSize)
suppressed_R_lenna = non_maximum_suppression_win(R_lenna, _winSize)

## d) show image and save