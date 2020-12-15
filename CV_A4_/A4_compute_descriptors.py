'''
Computer vision assignment 4 by Yoseob Kim
A4_compute_descriptors.py
Compute similarity-reflected image descriptors with L1, L2 norm distances by using SIFT descriptors.

* Status:       (working on it)
* GitHub Link:  https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A4_
'''

import cv2
import numpy as np
import math
import time
import operator
import random

img = cv2.imread('ukbench00000.jpg', cv2.IMREAD_GRAYSCALE)

'''
my_min = np.inf
my_max = 0'''
for i in range(1000):
    offset = '00' if i < 10 else '0' if i < 100 else ''
    offset += str(i)
    #print(offset)

    f = open('./sift/sift100'+offset, 'rb')

    # reference - https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html
    sift_des = np.frombuffer(f.read(), dtype=np.uint8)
    #print(sift_des.shape)
    #print(sift_des)

    '''
    if sift_des.shape[0] % 128 != 0:
        print('divide error')
    '''
    sift_des_reshaped = np.reshape(sift_des, (sift_des.shape[0] // 128, 128))
    #print(sift_des_reshaped.shape)

    '''
    if sift_des_reshaped.shape[0] < my_min:
        my_min = sift_des_reshaped.shape[0]
    if sift_des_reshaped.shape[0] > my_max:
        my_max = sift_des_reshaped.shape[0]'''

    f.close()


#print(my_min, my_max)
# N size
# min = 73, max = 2388




