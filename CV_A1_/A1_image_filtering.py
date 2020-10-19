'''
Computer vision assignment 1 by Yoseob Kim
A1_image_filtering.py
'''

import cv2

'''
Step 1. 
define functions - cross_correlation_1d, cross_correlation_2d, get_gaussian_filter_1d, get_gaussian_filter_2d
'''

def cross_correlation_1d(img, kernel):
    pass

def cross_correlation_2d(img, kernel):
    pass

def get_gaussian_filter_1d(size, sigma):
    pass

def get_gaussian_filter_2d(size, sigma):
    pass


'''
Step 2. 
load images in grayscale
'''

img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Lenna in grayscale", img_lenna)
# cv2.imshow("Shapes", img_shapes)


'''
Step 3.
implementation of requirements
'''





#cv2.waitKey(0)
#cv2.destroyAllWindows()