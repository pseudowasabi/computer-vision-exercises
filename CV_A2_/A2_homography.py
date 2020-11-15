'''
Computer vision assignment 2 by Yoseob Kim
A2_homography.py
Implementation of feature matching, homography, RANSAC, image warping.
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A2_
'''

import cv2
import numpy as np
import math
import time
import operator
import os

def compute_homography(srcP, destP):
    H = None

    N_for_homography = len(srcP)

    ## normalize srcP, destP
    # get average of x, y
    avg_x_src = 0.
    avg_y_src = 0.
    avg_x_dest = 0.
    avg_y_dest = 0.
    for i in range(N_for_homography):
        avg_x_src = operator.__add__(avg_x_src, srcP[i][0])
        avg_y_src = operator.__add__(avg_y_src, srcP[i][1])
        avg_x_dest = operator.__add__(avg_x_dest, destP[i][0])
        avg_y_dest = operator.__add__(avg_y_dest, destP[i][1])

    avg_x_src = operator.__truediv__(avg_x_src, N_for_homography)
    avg_y_src = operator.__truediv__(avg_y_src, N_for_homography)
    avg_x_dest = operator.__truediv__(avg_x_dest, N_for_homography)
    avg_y_dest = operator.__truediv__(avg_y_dest, N_for_homography)

    # subtract mean - need to change using matrix product
    src_max = 0.
    dest_max = 0.
    for i in range(N_for_homography):
        srcP[i][0] = operator.__sub__(srcP[i][0], avg_x_src)
        srcP[i][1] = operator.__sub__(srcP[i][1], avg_y_src)
        destP[i][0] = operator.__sub__(destP[i][0], avg_x_dest)
        destP[i][1] = operator.__sub__(destP[i][1], avg_y_dest)

        current_src = operator.__add__(operator.__pow__(srcP[i][0], 2.), operator.__pow__(srcP[i][1], 2.))
        if current_src > src_max:
            src_max = current_src
        current_dest = operator.__add__(operator.__pow__(destP[i][0], 2.), operator.__pow__(destP[i][1], 2.))
        if current_dest > dest_max:
            dest_max = current_dest

    # scaling - need to change using matrix product
    scaling_factor_src = math.sqrt(operator.__truediv__(2., src_max))
    scaling_factor_dest = math.sqrt(operator.__truediv__(2., dest_max))

    for i in range(N_for_homography):
        srcP[i][0] = operator.__mul__(srcP[i][0], scaling_factor_src)
        srcP[i][1] = operator.__mul__(srcP[i][1], scaling_factor_src)
        destP[i][0] = operator.__mul__(destP[i][0], scaling_factor_dest)
        destP[i][1] = operator.__mul__(destP[i][1], scaling_factor_dest)


    ## compute homography

    # stacking A matrices
    A = np.zeros((2 * N_for_homography, 9))
    for i in range(N_for_homography):
        x = srcP[i][0]
        y = srcP[i][1]
        x_ = destP[i][0]
        y_ = destP[i][1]

        A[i * 2] = [-x, -y, -1, 0, 0, 0, x * x_, y * x_, x_]
        A[i * 2 + 1] = [0, 0, 0, -x, -y, -1, x * y_, y * y_, y_]
    #print(A)

    # singular value decomposition of A
    # reference - https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    print(vh)
    print(vh.shape)
    # which vh to select?, how to select row/column?

    return vh



    #return H

def compute_homography_ransac(srcP, destP, th):
    H = None

    return H

# hamming distance reference - https://www.geeksforgeeks.org/hamming-distance-between-two-integers/
def getPreCalculatedHammingDistSet():
    HD = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            xor_val = operator.__xor__(i, j)
            cnt = 0
            while xor_val:
                cnt = operator.__add__(cnt, operator.__xor__(xor_val, 1))
                xor_val = operator.__rshift__(xor_val, 1)
            HD[i][j] = cnt
    return HD

precalcHD = getPreCalculatedHammingDistSet()

def getHammingDist(int_list1, int_list2):
    if len(int_list1) != len(int_list2):
        print('size of descriptor NOT MATCHING!')
        print('CANNOT CALCULATE hamming distance!')
        return -1

    dist_sum = 0
    for i in range(len(int_list1)):
        dist_sum = operator.__add__(dist_sum, precalcHD[int_list1[i]][int_list2[i]])

    return dist_sum


## 2-1. feature matching
# reference1 - https://076923.github.io/posts/Python-opencv-38/
# reference2 - https://m.blog.naver.com/samsjang/220657424078
# reference3 - https://python.hotexamples.com/examples/cv2/-/drawMatches/python-drawmatches-function-examples.html

cv_desk = cv2.imread('./cv_desk.png', cv2.IMREAD_GRAYSCALE)
cv_cover = cv2.imread('./cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

kp_desk = orb.detect(cv_desk, None)
kp_desk, des_desk = orb.compute(cv_desk, kp_desk)

kp_cover = orb.detect(cv_cover, None)
kp_cover, des_cover = orb.compute(cv_cover, kp_cover)

'''
class my_dmatch:
    queryIdx = 0
    trainIdx = 0
    imgIdx = 0
    distance = 10

    def __init__(self, qidx, tidx):
        self.queryIdx = qidx
        self.trainIdx = tidx

test1 = my_dmatch(desk_idx, cover_idx)
test2 = cv2.DMatch(_distance=10, _queryIdx=desk_idx, _trainIdx=cover_idx, _imgIdx=0)
'''
matches = []

elapsed_ = list(range(0, len(kp_desk), len(kp_desk) // 20))

# below is complete code of feature matching
for i in range(len(kp_desk)):
    for j in range(len(kp_cover)):
        dist = getHammingDist(des_desk[i], des_cover[j])
        matches.append(cv2.DMatch(_distance=float(dist), _queryIdx=i, _trainIdx=j, _imgIdx=0))
    if i in elapsed_:
        print('.', end='')
print()

matches = sorted(matches, key=lambda x: x.distance)

match_res = None
match_res = cv2.drawMatches(cv_desk, kp_desk, cv_cover, kp_cover, matches[:10], match_res, flags=2)
cv2.imshow("feature matching using ORB", match_res)
cv2.waitKey(0)


## 2-2~4. homography with... normalization vs RANSAC

N_for_homography = 15
srcP = np.zeros((N_for_homography, 2))
destP = np.zeros((N_for_homography, 2))

for i in range(N_for_homography):
    srcP[i][0] = kp_desk[matches[i].queryIdx].pt[0]
    srcP[i][1] = kp_desk[matches[i].queryIdx].pt[1]
    destP[i][0] = kp_cover[matches[i].trainIdx].pt[0]
    destP[i][1] = kp_cover[matches[i].trainIdx].pt[1]

#print(srcP)
#print(destP)

H_norm = compute_homography(srcP, destP)


#th = 0.
#H_ransac = compute_homography_ransac(srcP, destP, th)

## 2-5. stiching images
