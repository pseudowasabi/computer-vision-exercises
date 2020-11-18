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
import random

def compute_homography(srcP, destP):
    H = np.zeros((3, 3))

    N_for_homography = len(srcP)
    T_S = np.identity(3)    # transformation matrices for srcP (T_S), destP (T_D)
    T_D = np.identity(3)

    _srcP = srcP.copy()
    _destP = destP.copy()

    ## step 1. normalize srcP, destP

    # step 1-1. get average of x, y
    avg_x_src, avg_y_src, avg_x_dest, avg_y_dest = 0., 0., 0., 0.

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

    # step 1-2. subtract mean
    src_max, dest_max = 0., 0.
    #src_max_idx, dest_max_idx = 0, 0

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
        #if current_src > (_srcP[src_max_idx][0] ** 2 + _srcP[src_max_idx][1] ** 2):
        #    src_max_idx = i

        current_dest = _destP[i][0] ** 2 + _destP[i][1] ** 2
        if current_dest > dest_max:
            dest_max = current_dest
        #if current_dest > (_destP[dest_max_idx][0] ** 2 + _destP[dest_max_idx][1] ** 2):
        #    dest_max_idx = i

    # scaling (longest distance to sqrt(2))
    scaling_factor_src = math.sqrt(2 / src_max)
    _t_s = np.identity(3)
    _t_s[0][0] = scaling_factor_src
    _t_s[1][1] = scaling_factor_src
    #t_s[0][0] = 1. / _srcP[src_max_idx][0]
    #t_s[1][1] = 1. / _srcP[src_max_idx][1]

    scaling_factor_dest = math.sqrt(2 / dest_max)
    _t_d = np.identity(3)
    _t_d[0][0] = scaling_factor_dest
    _t_d[1][1] = scaling_factor_dest
    #_t_d[0][0] = 1. / _destP[dest_max_idx][0]
    #_t_d[1][1] = 1. / _destP[dest_max_idx][1]

    T_S = np.matmul(_t_s, T_S)  # translation first, scaling next
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

        A[i * 2] = [-x, -y, -1, 0., 0., 0., x * x_, y * x_, x_]
        A[i * 2 + 1] = [0., 0., 0., -x, -y, -1, x * y_, y * y_, y_]

    # singular value decomposition of A
    # reference - https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    # get singular vector of smallest singular value
    for i in range(9):
        H[i // 3][i % 3] = vh[len(s) - 1][i]    # last singular value is the smallest value.

    # we have to get un-normalized H
    H = np.matmul(np.linalg.inv(T_D), np.matmul(H, T_S))

    return H

def compute_homography_ransac(srcP, destP, th):
    H = None

    select_list = [i for i in range(len(srcP))]

    start_time = time.process_time()
    while True:
        elapsed_time_ = time.process_time() - start_time
        if elapsed_time_ > 2.9:
            break

        select = random.sample(select_list, 4)

        srcP_randomly_selected = []
        destP_randomly_selected = []

        for i in select:
            srcP_randomly_selected.append(list(srcP[i]))
            destP_randomly_selected.append(list(destP[i]))

        print(srcP_randomly_selected)
        print(destP_randomly_selected)

        _H = compute_homography(srcP_randomly_selected, destP_randomly_selected)
        print(_H)

        # reprojection error < threshold인 경우를 어떻게 찾아야 할까?
        # 해당 부분 강의 다시 들어보기 ...

        # hypothesis 설정...

        break

    return H

# hamming distance reference - https://www.geeksforgeeks.org/hamming-distance-between-two-integers/
def getPreCalculatedHammingDistSet():
    HD = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            xor_val = i ^ j
            cnt = 0
            while xor_val > 0:
                cnt += (xor_val & 1)
                xor_val >>= 1
            HD[i][j] = cnt
    return HD

precalcHD = getPreCalculatedHammingDistSet()


## 2-1. feature matching
# reference1 - https://076923.github.io/posts/Python-opencv-38/
# reference2 - https://m.blog.naver.com/samsjang/220657424078
# reference3 - https://python.hotexamples.com/examples/cv2/-/drawMatches/python-drawmatches-function-examples.html

cv_desk = cv2.imread('./cv_desk.png', cv2.IMREAD_GRAYSCALE)
cv_cover = cv2.imread('./cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

#orb = cv2.ORB_create(nfeatures=900)
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
_matches_desk_to_cover = [[255, 0, 0] for i in range(len(kp_desk))]
_matches_cover_to_desk = [[255, 0, 0] for i in range(len(kp_cover))]
elapsed_desk = list(range(0, len(kp_desk), len(kp_desk) // 20))

kp_desk_len = len(kp_desk)
kp_cover_len = len(kp_cover)
des_len = len(des_desk[0])

# matching process (processing time is about 21 seconds when nfeatures=900)
start_time = time.process_time()
for i in range(kp_desk_len):
    _matches_desk_to_cover[i][1] = i
    desk_i = des_desk[i]
    for j in range(kp_cover_len):
        dist = 0
        _matches_cover_to_desk[j][1] = j
        cover_j = des_cover[j]

        for k in range(des_len):
            dist = operator.__add__(dist, precalcHD[desk_i[k]][cover_j[k]])

        if dist < _matches_desk_to_cover[i][0]:
            _matches_desk_to_cover[i][0] = dist
            _matches_desk_to_cover[i][2] = j

        if dist < _matches_cover_to_desk[j][0]:
            _matches_cover_to_desk[j][0] = dist
            _matches_cover_to_desk[j][2] = i

    if i in elapsed_desk:
        print('.', end='')

print()
print("elapsed time:", time.process_time() - start_time)

_matches_desk_to_cover.sort(key=lambda x: x[0])
_matches_cover_to_desk.sort(key=lambda x: x[0])
'''
max_nfeatures = max(kp_desk_len, kp_cover_len)
cross_check = np.zeros((max_nfeatures, max_nfeatures))

for x in range(kp_desk_len):
    cross_check[_matches_desk_to_cover[x][1]][_matches_desk_to_cover[x][2]] += 1.
for x in range(kp_cover_len):
    cross_check[_matches_cover_to_desk[x][2]][_matches_cover_to_desk[x][1]] += 1.

matches = []
for m_i in range(kp_desk_len):
    if cross_check[_matches_desk_to_cover[m_i][1]][_matches_desk_to_cover[m_i][2]] == 2.:
        matches.append(cv2.DMatch(_distance=float(_matches_desk_to_cover[m_i][0]), _queryIdx=_matches_desk_to_cover[m_i][1], _trainIdx=_matches_desk_to_cover[m_i][2], _imgIdx=0))
'''

'''
# check if my matching process produces same result as BFMatcher.
bf = cv2.BFMatcher(cv2.NORM_HAMMING)    # , crossCheck=True) ## --> default value of crossCheck is False.

# after RANSAC part, use bf.match temporarily for implementing & debugging (no time consuming).
#matches = bf.match(des_desk, des_cover)
matches = bf.match(des_cover, des_desk)
matches.sort(key=lambda x: x.distance)

print('bf matcher', len(matches))
for x in matches:
    print(x.distance, x.queryIdx, x.trainIdx, end=' // ')
print()

print('my match', len(matches))
for x in matches:
    print(x.distance, x.queryIdx, x.trainIdx, end=' // ')
print()
'''

matches_d2c = []
for m_i in range(kp_desk_len):
    matches_d2c.append(cv2.DMatch(_distance=float(_matches_desk_to_cover[m_i][0]), _queryIdx=_matches_desk_to_cover[m_i][1], _trainIdx=_matches_desk_to_cover[m_i][2], _imgIdx=0))

'''
match_res = None
match_res = cv2.drawMatches(cv_desk, kp_desk, cv_cover, kp_cover, matches_d2c[:10], match_res, flags=2)
cv2.imshow("feature matching using ORB", match_res)
cv2.waitKey(0)
'''

## 2-2~4. homography with... normalization vs RANSAC

matches_c2d = []
for m_i in range(kp_cover_len):
    matches_c2d.append(cv2.DMatch(_distance=float(_matches_cover_to_desk[m_i][0]), _queryIdx=_matches_cover_to_desk[m_i][1], _trainIdx=_matches_cover_to_desk[m_i][2], _imgIdx=0))

N_for_homography = 15
print("N for homography:", N_for_homography)
srcP = np.zeros((N_for_homography, 2))
destP = np.zeros((N_for_homography, 2))

#for x in matches_c2d[:28]:
#    print(x.distance, x.queryIdx, x.trainIdx)

match_res = None
#match_res = cv2.drawMatches(cv_cover, kp_cover, cv_desk, kp_desk, matches_c2d[:N_for_homography], match_res, flags=2)
#cv2.imshow("feature matching using ORB", match_res)
#cv2.waitKey(0)

#orig_list = [i for i in range(40)]
#point_list = random.sample(orig_list, 15)
# below is candidates (sampled by above 2 lines)
# [22, 13, 11, 9, 19, 1, 33, 39, 34, 6, 32, 29, 17, 25, 5]
# [36, 28, 2, 11, 39, 14, 6, 1, 17, 10, 4, 8, 24, 25, 22]
# [20, 18, 8, 13, 19, 25, 9, 1, 31, 21, 11, 36, 0, 4, 14]       # select this one.

point_list = [20, 18, 8, 13, 19, 25, 9, 1, 31, 21, 11, 36, 0, 4, 14]
i = 0
for x in point_list:
    srcP[i] = list(kp_cover[matches_c2d[x].queryIdx].pt)
    destP[i] = list(kp_desk[matches_c2d[x].trainIdx].pt)
    i += 1

H_norm = compute_homography(srcP, destP)
'''
# similar result as cv2.warpPerspective (multiply homography matrix to origin[book cover] coordinates.)
homography_applied = np.zeros(cv_desk.shape)

for y in range(cv_cover.shape[0]):
    for x in range(cv_cover.shape[1]):
        w = H_norm[2][0] * x + H_norm[2][1] * y + H_norm[2][2] * 1
        x_ = (H_norm[0][0] * x + H_norm[0][1] * y + H_norm[0][2] * 1) / w
        y_ = (H_norm[1][0] * x + H_norm[1][1] * y + H_norm[1][2] * 1) / w

        if int(y_) in range(cv_desk.shape[0]) and int(x_) in range(cv_desk.shape[1]):
            homography_applied[int(y_)][int(x_)] = cv_cover[y][x]

homography_applied = cv2.normalize(homography_applied, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("after homography with normalize", homography_applied)
cv2.waitKey(0)
'''


img_warp_0 = cv2.warpPerspective(cv_cover, H_norm, (cv_desk.shape[1], cv_desk.shape[0]))
#cv2.imshow("homography with normalization", img_warp_0)
#cv2.waitKey(0)

homography_applied_overlay = cv_desk.copy()

for y in range(cv_desk.shape[0]):
    for x in range(cv_desk.shape[1]):
        if img_warp_0[y][x] > 0:
            homography_applied_overlay[y][x] = img_warp_0[y][x]

#cv2.imshow("homography with normalization (overlay)", homography_applied_overlay)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


th = 0.8

srcP_ransac = []
destP_ransac = []

for x in range(kp_cover_len):
    srcP_ransac.append(list(kp_cover[matches_c2d[x].queryIdx].pt))
    destP_ransac.append(list(kp_desk[matches_c2d[x].trainIdx].pt))

H_ransac = compute_homography_ransac(srcP_ransac, destP_ransac, th)
'''
img_warp_1 = cv2.warpPerspective(cv_cover, H_ransac, (cv_desk.shape[1], cv_desk.shape[0]))
cv2.imshow("homography with RANSAC", img_warp_1)
cv2.waitKey(0)

ransac_applied_overlay = cv_desk.copy()

for y in range(cv_desk.shape[0]):
    for x in range(cv_desk.shape[0]):
        if img_warp_1[y][x] > 0:
            ransac_applied_overlay[y][x] = img_warp_0[y][x]

cv2.imshow("homography with RANSAC (overlay)", ransac_applied_overlay)
cv2.waitKey(0)
'''

## 2-5. stiching images
