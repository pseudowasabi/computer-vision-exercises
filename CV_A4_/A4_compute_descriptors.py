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
f = open('./sift/sift100000', 'rb')
line1 = f.readline()
line2 = f.readline()
line3 = f.readline()

print(line1)
print(len(line1))
print(line2)
print(len(line2))
print(line3)
print(len(line3))


