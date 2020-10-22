'''
Computer vision assignment 1 by Yoseob Kim
A1_corner_detection.py
Implementation of Harris corner detector.
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A1_
'''

import cv2
import numpy as np
import math
import time
import operator
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

    elapsed_ = list(range(0, size0, size0 // 20))
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

        if x in elapsed_:
            print('.', end='')

    #print("_max_val:", _max_val)
    #normalizer = 1.
    #if _max_val != 0:
    #    normalizer = 1 / _max_val
    #normalizer = 1 / 255

    #normalizer = 1 / np.linalg.norm(R)
    for x in range(size0):
        for y in range(size1):
            R[x][y] = operator.__truediv__(R[x][y], _max_val)
    return R


def non_maximum_suppression_win(R, winSize):
    ## c) compute local maximas by NMS
    #* input argument R is already thresholded in corner_response_embedding func.
    x_bound = R.shape[0] - winSize + 1
    y_bound = R.shape[1] - winSize + 1

    # elapsed_ = list(range(0, R.shape[0], R.shape[1] // 20))
    for x in range(0, x_bound, winSize // 2):
        for y in range(0, y_bound, winSize // 2):
            local_maxima = R[x][y]
            lm_x = x
            lm_y = y
            for i in range(x, x + winSize):
                for j in range(y, y + winSize):
                    if R[i][j] > local_maxima:
                        local_maxima = R[i][j]
                        lm_x = i
                        lm_y = j
                    R[i][j] = 0
            # print(x, y, local_maxima, lm_x, lm_y)
            R[lm_x][lm_y] = local_maxima
        #if x in elapsed_:
        #    print('.', end='')
    return R


###
# 3-1. Gaussian filtering
###

img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

print('Computer Vision A#1 // Yoseob Kim')
print('Part #3. Corner Detection\n')

print('3-1. apply gaussian filtering for shapes and lenna. (size=7, sigma=1.5)')

print('filtering... (for "shapes.png")', end='')
filtered_img_shapes = my_gaussian_filtering(img_shapes, 7, 1.5)
print(' ---> done.')

print('filtering... (for "lenna.png")', end='')
filtered_img_lenna = my_gaussian_filtering(img_lenna, 7, 1.5)
print(' ---> done.')
print()
print()

###
# 3-2. Corner response (execute requirements - ...)
###
print('3-2. compute corner responses for shapes and lenna respectively.')

# for "shapes.png"
print(' ** corner responses for "shapes.png" initiate.')
print(' [about 20 dots will be shown to be done]')

start_time = time.process_time()
R_shapes = compute_corner_response(filtered_img_shapes)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

#print(np.min(R_shapes), np.max(R_shapes))
R_shapes = cv2.normalize(R_shapes, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_3_corner_raw_shapes.png', R_shapes)
R_shapes = cv2.normalize(R_shapes, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print(' * corner responses of "shapes.png" saved to ./result/ directory.')

print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.imshow("corner responses of shapes", R_shapes)
cv2.waitKey(0)

# for "lenna.png"
print(' ** corner responses for "lenna.png" initiate.')
print(' [about 20 dots will be shown to be done]')

start_time = time.process_time()
R_lenna = compute_corner_response(filtered_img_lenna)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

#print(np.min(R_lenna), np.max(R_lenna))
R_lenna = cv2.normalize(R_lenna, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_3_corner_raw_lenna.png', R_lenna)
R_lenna = cv2.normalize(R_lenna, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print(' * corner responses of "lenna.png" saved to ./result/ directory.')

print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.imshow("corner responses of lenna", R_lenna)
cv2.waitKey(0)
print()


###
# 3-3. Thresholding, Non-maximum suppression (execute requirements)
###
print('3-3. thresholding and apply NMS to corner responses for each images.')

## a) change corner response to green
## b) show image and save

def corner_response_embedding(R, img):
    # convert gray scale to rgb channel
    normalized_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    rgb_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)

    # thresholding (greater than 0.1)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i][j] > 0.1:
                cv2.circle(rgb_img, (j, i), 5, (0, 1.0, 0), 1)
            else:
                R[i][j] = 0

    return R, rgb_img

print('a, b) thresholding and corner response embedding to original images')

# for shapes.png
#print(np.min(img_shapes), np.max(img_shapes))
thresholded_R_shapes, rgb_img_shapes = corner_response_embedding(R_shapes, img_shapes)
#print(np.min(rgb_img_shapes), np.max(rgb_img_shapes))
rgb_img_shapes = cv2.normalize(rgb_img_shapes, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_3_corner_bin_shapes.png', rgb_img_shapes)
print(' ** corner response embedding of "shapes.png" saved to ./result/ directory.')

print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.imshow("corner response > 0.1 :: shapes.png", rgb_img_shapes)
cv2.waitKey(0)

# for lenna.png
#print(np.min(img_lenna), np.max(img_lenna))
thresholded_R_lenna, rgb_img_lenna = corner_response_embedding(R_lenna, img_lenna)
#print(np.min(rgb_img_lenna), np.max(rgb_img_lenna))
rgb_img_lenna = cv2.normalize(rgb_img_lenna, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_3_corner_bin_lenna.png', rgb_img_lenna)
print(' ** corner response embedding of "lenna.png" saved to ./result/ directory.')

print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.imshow("corner response > 0.1 :: lenna.png", rgb_img_lenna)
cv2.waitKey(0)
#print()


## c) nms
## d) show image and save

print("c, d) apply non-maximum suppression to corner responses")
_winSize = 11

# for "shapes.png"
print('...applying NMS to R (shapes.png)', end='')
start_time = time.process_time()
suppressed_R_shapes = non_maximum_suppression_win(thresholded_R_shapes, _winSize)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

_, rgb_img_shapes2 = corner_response_embedding(suppressed_R_shapes, img_shapes)
#print(np.min(rgb_img_shapes2), np.max(rgb_img_shapes2))
rgb_img_shapes2 = cv2.normalize(rgb_img_shapes2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_3_corner_sup_shapes.png', rgb_img_shapes2)
print(' ** NMS applied R of "shapes.png" saved to ./result/ directory.')

print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.imshow("non-maximum suppressed :: shapes.png", rgb_img_shapes2)
cv2.waitKey(0)

# for "lenna.png"
print('...applying NMS to R (lenna.png)', end='')
start_time = time.process_time()
suppressed_R_lenna = non_maximum_suppression_win(thresholded_R_lenna, _winSize)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

_, rgb_img_lenna2 = corner_response_embedding(suppressed_R_lenna, img_lenna)
#print(np.min(rgb_img_lenna2), np.max(rgb_img_lenna2))
rgb_img_lenna2 = cv2.normalize(rgb_img_lenna2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_3_corner_sup_lenna.png', rgb_img_lenna2)
print(' ** NMS applied R of "lenna.png" saved to ./result/ directory.')

print(' ## notice: P#3 done. press any key (on image window) to finish.\n')
cv2.imshow("non-maximum suppressed :: lenna.png", rgb_img_lenna2)
cv2.waitKey(0)


cv2.destroyAllWindows()
