'''
Computer vision assignment 1 by Yoseob Kim
A1_corner_detection.py
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A1_
'''

import cv2
import numpy as np
import math
import time
from filtering_by_yoseob import *

# ** initial settings
# ** set Sobel filter kernels as global variables
derivative_kernel = [-1, 0, +1]  # correlation ; [1, 0, -1] is for convolution.
blurring_filter = [1, 2, 1]
# for faster operation, use superposition of 1-d kernel filtering method.
sobel_x_0 = np.array(derivative_kernel)
sobel_x_1 = np.array([blurring_filter]).transpose()
sobel_y_0 = np.array(blurring_filter)
sobel_y_1 = np.array([derivative_kernel]).transpose()
# below is 2-d kernel for sobel cross-correlation kernel.
# sobel_x = sobel_x_1.dot(np.array([sobel_x_0]))
# sobel_y = sobel_y_1.dot(np.array([sobel_y_0]))
# print(sobel_x)
# print(sobel_y)


def compute_corner_response(img):
    ## a) apply Sobel filters

    # derivatives along x direction
    sobel_img_x = cross_correlation_1d(img, sobel_x_0)
    sobel_img_x = cross_correlation_1d(sobel_img_x, sobel_x_1)

    # derivatives along y direction
    sobel_img_y = cross_correlation_1d(img, sobel_y_0)
    sobel_img_y = cross_correlation_1d(sobel_img_y, sobel_y_1)

    ## b) compute second moment matrix M

    ## c) get corner response

    ## d) normalize responses


def non_maximum_suppression_win(R, winSize):
    pass

    ## c) compute local maximas by NMS

    ## d) ...


###
# 3-1. Gaussian filtering
###

img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

kernel_h = get_gaussian_filter_1d(7, 1.5)
kernel_v = np.array([kernel_h]).transpose()

filtered_img_lenna = cross_correlation_1d(img_lenna, kernel_h)
filtered_img_lenna = cross_correlation_1d(filtered_img_lenna, kernel_v)

filtered_img_shapes = cross_correlation_1d(img_shapes, kernel_h)
filtered_img_shapes = cross_correlation_1d(filtered_img_shapes, kernel_v)


###
# 3-2. Corner response
###




###
# 3-3. Thresholding, Non-maximum suppression
###

