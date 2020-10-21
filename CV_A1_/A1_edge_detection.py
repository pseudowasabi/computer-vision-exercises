'''
Computer vision assignment 1 by Yoseob Kim
A1_edge_detection.py
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A1_
'''

import cv2
import numpy as np
import math
import time
from filtering_by_yoseob import *


def compute_image_gradient(img):
    pass

def non_maximum_suppression_dir(mag, dir):
    pass



###
# 2-1. Gaussian filtering
###

img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

kernel_h = get_gaussian_filter_1d(7, 1.5)
kernel_v = np.array([kernel_h]).transpose()

filtered_img_lenna = cross_correlation_1d(img_lenna, kernel_h)
filtered_img_lenna = cross_correlation_1d(filtered_img_lenna, kernel_v).astype(dtype='uint8')

filtered_img_shapes = cross_correlation_1d(img_shapes, kernel_h)
filtered_img_shapes = cross_correlation_1d(filtered_img_shapes, kernel_v).astype(dtype='uint8')

'''
cv2.imshow("filtered lenna", filtered_img_lenna)
cv2.waitKey(0)

cv2.imshow('filtered shapes', filtered_img_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


###
# 2-2. Image gradient
###

## a) apply Sobel filters



## b) compute magnitude and direction of gradient

## c) ...
## d) print computational times, call imshow function to show magnitude maps, store image files



###
# 2-3. Non-maximum suppression
###

