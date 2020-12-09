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
    M_size = M.shape[0]
    A = np.zeros((M_size, 9))

    # print(M)
    # print(M.shape)

    # for i in range(8):
    for i in range(M_size):
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

img1_size = ()
img2_size = ()
def compute_F_norm(M):
    # translate center to origin
    diff_h1, diff_w1 = img1_size[0] / 2, img1_size[1] / 2
    diff_h2, diff_w2 = img2_size[0] / 2, img2_size[1] / 2

    _M = np.copy(M)

    normalizer = np.array([diff_w1, diff_h1, diff_w2, diff_h2])
    M = M - normalizer     # numpy broadcasting

    # scale into unit square
    M = M / normalizer     # numpy broadcasting

    # generate fundamental matrix
    _F_norm = compute_F_raw(M)

    M_translate1 = np.array([[1., 0., -diff_w1], [0., 1., -diff_h1], [0., 0., 1.]])
    M_scale1 = np.array([[1. / diff_w1, 0., 0.], [0., 1. / diff_h1, 0.], [0., 0., 1.]])
    M_transform1 = M_scale1.dot(M_translate1)

    M_translate2 = np.array([[1., 0., -diff_w2], [0., 1., -diff_h2], [0., 0., 1.]])
    M_scale2 = np.array([[1. / diff_w2, 0., 0.], [0., 1. / diff_h2, 0.], [0., 0., 1.]])
    M_transform2 = M_scale2.dot(M_translate2)

    F_norm = np.transpose(M_transform2).dot(_F_norm).dot(M_transform1)    # reference - slide P73

    M = np.copy(_M)
    return F_norm

def compute_F_mine(M):
    pass

## 1-1. Fundamental matrix computation

print('# 1-1. Fundamental matrix computation')
print()

img_temple1 = cv2.imread('temple1.png', cv2.IMREAD_GRAYSCALE)
img_temple2 = cv2.imread('temple2.png', cv2.IMREAD_GRAYSCALE)
img_house1 = cv2.imread('house1.jpg', cv2.IMREAD_GRAYSCALE)
img_house2 = cv2.imread('house2.jpg', cv2.IMREAD_GRAYSCALE)
img_library1 = cv2.imread('library1.jpg', cv2.IMREAD_GRAYSCALE)
img_library2 = cv2.imread('library2.jpg', cv2.IMREAD_GRAYSCALE)


'''
# picture 1, 2 has same shape
print(img_temple1.shape)
print(img_temple2.shape)
print(img_house1.shape)
print(img_house2.shape)
print(img_library1.shape)
print(img_library2.shape)
'''


M_temple = np.loadtxt('temple_matches.txt')
M_house = np.loadtxt('house_matches.txt')
M_library = np.loadtxt('library_matches.txt')

print('Average Reprojection Errors (temple1.png and temple2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_temple, compute_F_raw(M_temple)))
img1_size = img_temple1.shape
img2_size = img_temple2.shape
print('\tNorm =', care.compute_avg_reproj_error(M_temple, compute_F_norm(M_temple)))
#print('\tMine =', care.compute_avg_reproj_error(M_temple, compute_F_mine(M_temple)))

print('Average Reprojection Errors (house1.png and house2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_house, compute_F_raw(M_house)))
img1_size = img_house1.shape
img2_size = img_house2.shape
print('\tNorm =', care.compute_avg_reproj_error(M_house, compute_F_norm(M_house)))
#print('\tMine =', care.compute_avg_reproj_error(M_house, compute_F_mine(M_house)))

print('Average Reprojection Errors (library1.png and library2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_library, compute_F_raw(M_library)))
img1_size = img_library1.shape
img2_size = img_library2.shape
print('\tNorm =', care.compute_avg_reproj_error(M_library, compute_F_norm(M_library)))
#print('\tMine =', care.compute_avg_reproj_error(M_library, compute_F_mine(M_library)))

print()

## 1-2. Visualization of epipolar lines
print('# 1-2. Visualization of epipolar lines')
print()
