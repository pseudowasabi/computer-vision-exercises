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

    return H

# hamming distance reference - https://www.geeksforgeeks.org/hamming-distance-between-two-integers/
def getPreCalculatedHammingDistSet():
    HD = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            xor = i ^ j
            cnt = 0
            while xor:
                cnt += xor & 1
                xor >>= 1
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
        dist_sum += precalcHD[int_list1[i]][int_list2[i]]

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
