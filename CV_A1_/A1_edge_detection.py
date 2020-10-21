'''
Computer vision assignment 1 by Yoseob Kim
A1_edge_detection.py
Implementation of Canny edge detection.
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A1_
'''

import cv2
import numpy as np
import math
import time
from filtering_by_yoseob import *


###
# 2-2. Image gradient (define function - a, b and c)
###

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

def compute_image_gradient(img):
    ## a) apply Sobel filters

    # derivatives along x direction
    sobel_img_x = cross_correlation_1d(img, sobel_x_0)
    sobel_img_x = cross_correlation_1d(sobel_img_x, sobel_x_1)

    # derivatives along y direction
    sobel_img_y = cross_correlation_1d(img, sobel_y_0)
    sobel_img_y = cross_correlation_1d(sobel_img_y, sobel_y_1)


    ## b) compute magnitude and direction of gradient

    mag = np.zeros((img.shape[0], img.shape[1]))
    dir = np.zeros((img.shape[0], img.shape[1]))
    j_range = range(img.shape[1])

    mag_max = 0
    for i in range(img.shape[0]):
        for j in j_range:
            # magnitude
            mag[i][j] = math.sqrt((sobel_img_x[i][j] ** 2) + (sobel_img_y[i][j] ** 2))
            mag_max = mag[i][j] if mag[i][j] > mag_max else mag_max

            # direction - adjust range to [0, 2 * pi].
            dir[i][j] = math.atan2(sobel_img_y[i][j], sobel_img_x[i][j])
            #if dir[i][j] < 0.0:
            #    dir[i][j] += (2 * math.pi)

    # normalize magnitude to range [0, 1]
    # reference to the idea of normalize (when mag_max value goes above 255).
    # https://stackoverflow.com/questions/52913722/image-processing-the-result-of-sobel-filter-turns-out-to-be-gray-instead-of-bla#comment92744720_52916385
    if mag_max != 0:
        normalizer = 1 / mag_max  # (255 / mag_max) / 255
    else:
        normalizer = 1

    for i in range(img.shape[0]):
        for j in j_range:
            mag[i][j] = mag[i][j] * normalizer

    return mag, dir


#dx = [0, -1, -1, -1, 0, 1, 1, 1]
#dy = [1, 1, 0, -1, -1, -1, 0, 1]
dx = [0, 1, 1, 1, 0, -1, -1, -1]
dy = [-1, -1, 0, 1, 1, 1, 0, -1]
d45 = math.pi / 4   # 45 degree
bins = [-3*d45, -2*d45, -d45, 0.0, d45, 2*d45, 3*d45, math.pi]

def non_maximum_suppression_dir(mag, dir):
    suppressed_mag = np.zeros((mag.shape[0], mag.shape[1]))
    j_range = range(mag.shape[1])

    # thresholding first (only check for TH; high threshold)
    # reference to select threshold value (take max threshold)
    # https://stackoverflow.com/questions/24862374/canny-edge-detector-threshold-values-gives-different-result
    mag_avg = np.average(mag)
    threshold_value = 1.33 * mag_avg
    for i in range(mag.shape[0]):
        for j in j_range:
            if mag[i][j] < threshold_value:
                mag[i][j] = 0

    for i in range(mag.shape[0]):
        for j in j_range:
            # get direction of each pixels.
            bin_num = -1
            #if dir[i][j] > bins[7]:
            #    dir[i][j] -= (2 * math.pi)
            for k in range(8):
                if dir[i][j] <= bins[k]:
                    bin_num = k
                    break
            #print(dir[i][j], ", ", bin_num)
            if bin_num == -1:
                print("index error occurs...")
                continue

            # check is direction or opposite direction pixels are in range.
            dir0 = (i+dx[bin_num], j+dy[bin_num])
            if 0 <= dir0[0] < mag.shape[0] and 0 <= dir0[1] < mag.shape[1]:
                dir0_valid = True
            else:
                dir0_valid = False

            dir1 = (i+dx[(bin_num+4)%8], j+dy[(bin_num+4)%8])
            if 0 <= dir1[0] < mag.shape[0] and 0 <= dir1[1] < mag.shape[1]:
                dir1_valid = True
            else:
                dir1_valid = False

            # execute non-maximum suppression.
            if dir0_valid:
                if dir1_valid:
                    if mag[i][j] >= mag[dir0[0]][dir0[1]] and mag[i][j] >= mag[dir1[0]][dir1[1]]:  # alive
                        suppressed_mag[i][j] = mag[i][j]
                else:   # dir0 is valid only
                    if mag[i][j] >= mag[dir0[0]][dir0[1]]:
                        suppressed_mag[i][j] = mag[i][j]
            else:
                if dir1_valid:
                    if mag[i][j] >= mag[dir1[0]][dir1[1]]:
                        suppressed_mag[i][j] = mag[i][j]
    return suppressed_mag


###
# 2-1. Gaussian filtering
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
# 2-2. Image gradient (execute requirements - d and e)
###

## d) print computational times, call imshow function to show magnitude maps, store image files

mag_lenna, dir_lenna = compute_image_gradient(filtered_img_lenna)
cv2.imshow("magnitude map of lenna", mag_lenna)
cv2.waitKey(0)

mag_shapes, dir_shapes = compute_image_gradient(filtered_img_shapes)
cv2.imshow("magnitude map of shapes", mag_shapes)
cv2.waitKey(0)
#cv2.destroyAllWindows()

###
# 2-3. Non-maximum suppression
###

suppressed_mag_lenna = non_maximum_suppression_dir(mag_lenna, dir_lenna)
suppressed_mag_shapes = non_maximum_suppression_dir(mag_shapes, dir_shapes)

cv2.imshow("magnitude map of lenna, non-maximum suppressed", suppressed_mag_lenna)
cv2.waitKey(0)

cv2.imshow("magnitude map of shapes, non-maximum suppressed", suppressed_mag_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()