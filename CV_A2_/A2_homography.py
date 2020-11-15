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
    H = np.zeros((3, 3))

    N_for_homography = len(srcP)
    T_S = np.identity(3)
    T_D = np.identity(3)

    _srcP = srcP.copy()
    _destP = destP.copy()

    ## normalize srcP, destP
    # get average of x, y
    avg_x_src = 0.
    avg_y_src = 0.
    avg_x_dest = 0.
    avg_y_dest = 0.
    for i in range(N_for_homography):
        avg_x_src += _srcP[i][0]
        avg_y_src += _srcP[i][1]
        avg_x_dest += _destP[i][0]
        avg_y_dest += _destP[i][1]

    avg_x_src /= N_for_homography
    avg_y_src /= N_for_homography
    T_S[0][2] = -avg_x_src
    T_S[1][2] = -avg_y_src

    avg_x_dest /= N_for_homography
    avg_y_dest /= N_for_homography
    T_D[0][2] = -avg_x_dest
    T_D[1][2] = -avg_y_dest

    # subtract mean
    src_max = 0.
    dest_max = 0.

    for i in range(N_for_homography):
        _x = T_S[0][0] * _srcP[i][0] + T_S[0][1] * _srcP[i][1] + T_S[0][2] * 1
        _y = T_S[1][0] * _srcP[i][0] + T_S[1][1] * _srcP[i][1] + T_S[1][2] * 1
        _srcP[i][0] = _x
        _srcP[i][1] = _y

        _x = T_D[0][0] * _destP[i][0] + T_D[0][1] * _destP[i][1] + T_D[0][2] * 1
        _y = T_D[1][0] * _destP[i][0] + T_D[1][1] * _destP[i][1] + T_D[1][2] * 1
        _destP[i][0] = _x
        _destP[i][1] = _y

        current_src = _srcP[i][0] ** 2 + _srcP[i][1] ** 2
        if current_src > src_max:
            src_max = current_src

        current_dest = _destP[i][0] ** 2 + _destP[i][1] ** 2
        if current_dest > dest_max:
            dest_max = current_dest

    # scaling (longest distance to sqrt(2))
    scaling_factor_src = math.sqrt(2 / src_max)
    _t_s = np.identity(3)
    _t_s[0][0] = scaling_factor_src
    _t_s[1][1] = scaling_factor_src

    scaling_factor_dest = math.sqrt(2 / dest_max)
    _t_d = np.identity(3)
    _t_d[0][0] = scaling_factor_dest
    _t_d[1][1] = scaling_factor_dest

    T_S = np.matmul(_t_s, T_S)
    T_D = np.matmul(_t_d, T_D)

    _srcP = srcP.copy()
    _destP = destP.copy()
    for i in range(N_for_homography):
        _x = T_S[0][0] * _srcP[i][0] + T_S[0][1] * _srcP[i][1] + T_S[0][2] * 1
        _y = T_S[1][0] * _srcP[i][0] + T_S[1][1] * _srcP[i][1] + T_S[1][2] * 1
        _srcP[i][0] = _x
        _srcP[i][1] = _y

        _x = T_D[0][0] * _destP[i][0] + T_D[0][1] * _destP[i][1] + T_D[0][2] * 1
        _y = T_D[1][0] * _destP[i][0] + T_D[1][1] * _destP[i][1] + T_D[1][2] * 1
        _destP[i][0] = _x
        _destP[i][1] = _y

    ## compute homography

    # stacking A matrices
    A = np.zeros((2 * N_for_homography, 9))
    for i in range(N_for_homography):
        x = _srcP[i][0]
        y = _srcP[i][1]
        x_ = _destP[i][0]
        y_ = _destP[i][1]

        A[i * 2] = [-x, -y, -1, 0, 0, 0, x * x_, y * x_, x_]
        A[i * 2 + 1] = [0, 0, 0, -x, -y, -1, x * y_, y * y_, y_]

    # singular value decomposition of A
    # reference - https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    # get singular vector of smallest singular value
    s_i = 0
    for i in range(9):
        if s[i] < s[s_i]:
            s_i = i

    for i in range(9):
        H[i // 3][i % 3] = vh[s_i][i]

    # we have to get un-normalized H
    H = np.matmul(np.linalg.inv(T_D), np.matmul(H, T_S))

    return H

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


## 2-1. feature matching
# reference1 - https://076923.github.io/posts/Python-opencv-38/
# reference2 - https://m.blog.naver.com/samsjang/220657424078
# reference3 - https://python.hotexamples.com/examples/cv2/-/drawMatches/python-drawmatches-function-examples.html

cv_desk = cv2.imread('./cv_desk.png', cv2.IMREAD_GRAYSCALE)
cv_cover = cv2.imread('./cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)

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
_matches = [0 for i in range(len(kp_desk) * len(kp_cover))]
elapsed_ = list(range(0, len(kp_desk), len(kp_desk) // 20))

# below is complete code of feature matching
m = 0
for i in range(len(kp_desk)):
    for j in range(len(kp_cover)):
        dist = 0
        for k in range(len(des_desk[i])):
            dist = operator.__add__(dist, precalcHD[des_desk[i][k]][des_cover[j][k]])

        #matches[m] = cv2.DMatch(_distance=float(dist), _queryIdx=i, _trainIdx=j, _imgIdx=0)
        _matches[m] = [float(dist), i, j]
        # dist 기준으로 저장해두고 정렬한 다음에 상위 값만 DMatch 만드는 방식으로 하면 성능 향상 가능할듯...?
        m += 1
    if i in elapsed_:
        print('.', end='')
print()

#matches = sorted(matches, key=lambda x: x.distance)
_matches.sort(key=lambda x: x[0])
matches = []
for i in range(30):     # 상위 30개 정도만 처리
    matches.append(cv2.DMatch(_distance=_matches[i][0], _queryIdx=_matches[i][1], _trainIdx=_matches[i][2], _imgIdx=0))

match_res = None
match_res = cv2.drawMatches(cv_desk, kp_desk, cv_cover, kp_cover, matches[:10], match_res, flags=2)
cv2.imshow("feature matching using ORB", match_res)
cv2.waitKey(0)


## 2-2~4. homography with... normalization vs RANSAC

N_for_homography = 15
srcP = np.zeros((N_for_homography, 2))
destP = np.zeros((N_for_homography, 2))

for i in range(N_for_homography):
    srcP[i][0] = kp_cover[matches[i].trainIdx].pt[0]
    srcP[i][1] = kp_cover[matches[i].trainIdx].pt[1]
    destP[i][0] = kp_desk[matches[i].queryIdx].pt[0]
    destP[i][1] = kp_desk[matches[i].queryIdx].pt[1]

#print(srcP)
#print(destP)


H_norm = compute_homography(srcP, destP)
print("H_norm")
print(H_norm)

homography_applied = np.zeros(cv_desk.shape)

for y in range(cv_cover.shape[0]):
    for x in range(cv_cover.shape[1]):
        w = H_norm[2][0] * x + H_norm[2][1] * y + H_norm[2][2] * 1
        x_ = (H_norm[0][0] * x + H_norm[0][1] * y + H_norm[0][2] * 1) / w
        y_ = (H_norm[1][0] * x + H_norm[1][1] * y + H_norm[1][2] * 1) / w

        if int(y_) in range(cv_desk.shape[0]) and int(x_) in range(cv_desk.shape[1]):
            homography_applied[int(y_)][int(x_)] = cv_cover[y][x]
'''
homography_applied = cv2.normalize(homography_applied, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("after homography with normalize", homography_applied)
cv2.waitKey(0)
'''
img_warp_0 = cv2.warpPerspective(cv_cover, H_norm, (cv_desk.shape[1], cv_desk.shape[0]))
cv2.imshow("image warp perspective", img_warp_0)
cv2.waitKey(0)

homography_applied_overlay = cv_desk.copy()

for y in range(cv_desk.shape[0]):
    for x in range(cv_desk.shape[0]):
        if img_warp_0[y][x] > 0:
            homography_applied_overlay[y][x] = img_warp_0[y][x]

cv2.imshow("overlay homography", homography_applied_overlay)
cv2.waitKey(0)


th = 0.8
H_ransac = compute_homography_ransac(srcP, destP, th)

## 2-5. stiching images
