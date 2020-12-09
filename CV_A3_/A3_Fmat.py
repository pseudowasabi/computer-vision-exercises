'''
Computer vision assignment 3 by Yoseob Kim
A3_Fmat.py
Implementation of fundamental matrix, epipolar lines.
'''

import cv2
import numpy as np
import math
import time
import operator
import random
import compute_avg_reproj_error as care

def compute_F_raw(M):
    _M = M.shape[0]
    A = np.zeros((_M, 9))

    #print(M)
    #print(M.shape)

    for i in range(_M):
        x, y, x_, y_ = M[i][0], M[i][1], M[i][2], M[i][3]

        A[i][0] = x * x_
        A[i][1] = x * y_
        A[i][2] = x
        A[i][3] = y * x_
        A[i][4] = y * y_
        A[i][5] = y
        A[i][6] = x_
        A[i][7] = y_
        A[i][8] = 1

    # singular value decomposition of A
    # reference - https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    #print(s)
    #print(np.min(s))

    F_raw = np.reshape(vh[-1], (3, 3))
    return F_raw


def compute_F_norm(M):
    pass

def compute_F_mine(M):
    pass

## 1-1. Fundamental matrix computation

print('# 1-1. Fundamental matrix computation')
print()

img_temple1 = cv2.imread('temple1.png', cv2.IMREAD_GRAYSCALE)
img_temple2 = cv2.imread('temple2.png', cv2.IMREAD_GRAYSCALE)
img_library1 = cv2.imread('house1.png', cv2.IMREAD_GRAYSCALE)
img_library2 = cv2.imread('house2.png', cv2.IMREAD_GRAYSCALE)
img_library1 = cv2.imread('library1.png', cv2.IMREAD_GRAYSCALE)
img_library2 = cv2.imread('library2.png', cv2.IMREAD_GRAYSCALE)

M_temple = np.loadtxt('temple_matches.txt')
M_house = np.loadtxt('house_matches.txt')
M_library = np.loadtxt('library_matches.txt')

print('Average Reprojection Errors (temple1.png and temple2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_temple, compute_F_raw(M_temple)))

'''
print('\tNorm =', care.compute_avg_reproj_error(M_temple, compute_F_norm(M_temple)))
print('\tMine =', care.compute_avg_reproj_error(M_temple, compute_F_mine(M_temple)))

print('Average Reprojection Errors (house1.png and house2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_house, compute_F_raw(M_house)))
print('\tNorm =', care.compute_avg_reproj_error(M_house, compute_F_norm(M_house)))
print('\tMine =', care.compute_avg_reproj_error(M_house, compute_F_mine(M_house)))

print('Average Reprojection Errors (library1.png and library2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_library, compute_F_raw(M_library)))
print('\tNorm =', care.compute_avg_reproj_error(M_library, compute_F_norm(M_library)))
print('\tMine =', care.compute_avg_reproj_error(M_library, compute_F_mine(M_library)))

print()

## 1-2. Visualization of epipolar lines
print('# 1-2. Visualization of epipolar lines')
print()

'''