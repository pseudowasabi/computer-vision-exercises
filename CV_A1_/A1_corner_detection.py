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



def compute_corner_response(img):
    ## a) apply Sobel filters
    # Sobel filtering function is implemented in filtering_by_yoseob.py
    # sobel_img_x, sobel_img_y is derivatives along x and y direction respectively.
    sobel_img_x, sobel_img_y = my_sobel_filtering(img)

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

filtered_img_lenna = my_gaussian_filtering(img_lenna, 7, 1.5)
filtered_img_shapes = my_gaussian_filtering(img_shapes, 7, 1.5)


###
# 3-2. Corner response
###




###
# 3-3. Thresholding, Non-maximum suppression
###

