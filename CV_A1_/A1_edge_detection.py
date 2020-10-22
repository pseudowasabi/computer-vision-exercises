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


def compute_image_gradient(img):
    ## a) apply Sobel filters

    # Sobel filtering function is implemented in filtering_by_yoseob.py
    # sobel_img_x, sobel_img_y is derivatives along x and y direction respectively.
    sobel_img_x, sobel_img_y = my_sobel_filtering(img)

    ## b) compute magnitude and direction of gradient

    mag = np.zeros((img.shape[0], img.shape[1]))
    dir = np.zeros((img.shape[0], img.shape[1]))
    j_range = range(img.shape[1])

    mag_max = 0
    elapsed_ = list(range(0, img.shape[0], img.shape[0]//10))
    for i in range(img.shape[0]):
        for j in j_range:
            # magnitude
            mag[i][j] = math.sqrt((sobel_img_x[i][j] ** 2) + (sobel_img_y[i][j] ** 2))
            mag_max = mag[i][j] if mag[i][j] > mag_max else mag_max

            # direction - adjust range to [0, 2 * pi].
            dir[i][j] = math.atan2(sobel_img_y[i][j], sobel_img_x[i][j])
            #if dir[i][j] < 0.0:
            #    dir[i][j] += (2 * math.pi)
        if i in elapsed_:
            print('.', end='')

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
        if i in elapsed_:
            print('.', end='')

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

    elapsed_ = list(range(0, mag.shape[0], mag.shape[0]//10))
    for i in range(mag.shape[0]):
        for j in j_range:
            if mag[i][j] < threshold_value:
                mag[i][j] = 0
        if i in elapsed_:
            print('.', end='')

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
        if i in elapsed_:
            print('.', end='')

    return suppressed_mag


###
# 2-1. Gaussian filtering
###

img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

print('Computer Vision A#1 // Yoseob Kim')
print('Part #2. Edge Detection\n')

print('2-1. apply gaussian filtering for shapes and lenna. (size=7, sigma=1.5)')

print('filtering... (for "shapes.png")', end='')
filtered_img_shapes = my_gaussian_filtering(img_shapes, 7, 1.5)
print(' ---> done.')

print('filtering... (for "lenna.png")', end='')
filtered_img_lenna = my_gaussian_filtering(img_lenna, 7, 1.5)
print(' ---> done.')
print()
print()

###
# 2-2. Image gradient (execute requirements - d and e)
###

## d) print computational times, call imshow function to show magnitude maps, store image files
print('2-2. compute image gradients for shapes and lenna respectively.')

# for "shapes.png"
print(' ** image gradients for "shapes.png" initiate.')
print(' [about 20 dots will be shown to be done]')

start_time = time.process_time()
mag_shapes, dir_shapes = compute_image_gradient(filtered_img_shapes)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

#print(mag_shapes)
mag_shapes = cv2.normalize(mag_shapes, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_2_edge_raw_shapes.png', mag_shapes)
print(' * image gradients of "shapes.png" saved to ./result/ directory.')

print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.imshow("magnitude map of shapes", mag_shapes)
cv2.waitKey(0)

# for "lenna.png"
print(' ** image gradients for "lenna.png" initiate.')
print(' [about 20 dots will be shown to be done]')

start_time = time.process_time()
mag_lenna, dir_lenna = compute_image_gradient(filtered_img_lenna)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

#print(mag_lenna)
mag_lenna = cv2.normalize(mag_lenna, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_2_edge_raw_lenna.png', mag_lenna)
print(' * image gradients of "lenna.png" saved to ./result/ directory.')

print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.imshow("magnitude map of lenna", mag_lenna)
cv2.waitKey(0)
print()

###
# 2-3. Non-maximum suppression
###

print('2-3. apply non-maximum suppression for image gradients of each image respectively.')

# for "shapes.png"
print(' ** non-maximum suppression for image gradients of "shapes.png" initiate.')
print(' [about 20 dots will be shown to be done]')

start_time = time.process_time()
suppressed_mag_shapes = non_maximum_suppression_dir(mag_shapes, dir_shapes)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

#print(suppressed_mag_shapes)
suppressed_mag_shapes = cv2.normalize(suppressed_mag_shapes, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_2_edge_sup_shapes.png', suppressed_mag_shapes)
print(' * non-maximum suppression result of shapes saved to ./result/ directory.')

print(' ## notice: press any key (on image window) to continue. !!do not close window!!\n')
cv2.imshow("magnitude map of shapes, non-maximum suppressed", suppressed_mag_shapes)
cv2.waitKey(0)

# for "lenna.png"
print(' ** non-maximum suppression for image gradients of "lenna.png" initiate.')
print(' [about 20 dots will be shown to be done]')

start_time = time.process_time()
suppressed_mag_lenna = non_maximum_suppression_dir(mag_lenna, dir_lenna)
elapsed_time = time.process_time() - start_time
print(' ---> done. /elapsed time:', elapsed_time)

#print(suppressed_mag_lenna)
suppressed_mag_lenna = cv2.normalize(suppressed_mag_lenna, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./result/part_2_edge_sup_lenna.png', suppressed_mag_lenna)
print(' * non-maximum suppression result of lenna saved to ./result/ directory.')

print(' ## notice: P#2 done. press any key (on image window) to finish.\n')
cv2.imshow("magnitude map of lenna, non-maximum suppressed", suppressed_mag_lenna)
cv2.waitKey(0)

cv2.destroyAllWindows()