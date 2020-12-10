'''
Computer vision assignment 3 by Yoseob Kim
A3_Fmat.py
Implementation of fundamental matrix, epipolar lines.
* Status:       Epipolar line visualization remains.
* GitHub Link:  https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A3_
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

num_of_points = 8    # bigger number than 8
def compute_F_mine(M):
    # basic idea from RANSAC

    F_mine = compute_F_norm(M)
    care_min_val = care.compute_avg_reproj_error(M, F_mine)

    select_list = [i for i in range(M.shape[0])]
    start_time = time.process_time()
    while True:
        elapsed_time_ = time.process_time() - start_time
        if elapsed_time_ >= 3.:  # get RANSAC within 3 seconds.
            break

        select = random.sample(select_list, num_of_points)

        M_randomly_selected = M[select]
        _F_mine = compute_F_norm(M_randomly_selected)

        #care_val = care.compute_avg_reproj_error(M_randomly_selected, _F_mine)
        care_val = care.compute_avg_reproj_error(M, _F_mine)
        if care_val < care_min_val:
            care_min_val = care_val
            F_mine = np.copy(_F_mine)

    return F_mine



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

'''
# find best num of points (like cross validation)
best_num_of_points, min_error = 8, math.inf
for i in range(8, 50):
    print('num_of_points =', i)
    num_of_points = i
    # find best number of points (in RANSAC)
    img1_size, img2_size = img_temple1.shape, img_temple2.shape
    t_error = care.compute_avg_reproj_error(M_temple, compute_F_mine(M_temple))
    print('Temple =', t_error)

    img1_size, img2_size = img_house1.shape, img_house2.shape
    h_error = care.compute_avg_reproj_error(M_house, compute_F_mine(M_house))
    print('House =', h_error)

    img1_size, img2_size = img_library1.shape, img_library2.shape
    l_error = care.compute_avg_reproj_error(M_library, compute_F_mine(M_library))
    print('Library =', l_error)

    tot_error = t_error + h_error + l_error
    if tot_error < min_error:
        min_error = tot_error
        best_num_of_points = num_of_points
    print('total error =', tot_error, ', minimum error =', min_error)
    print('best_num_of_points =', best_num_of_points)
    print()
    
'''

# best number of points = 8
num_of_points = 8

print('Average Reprojection Errors (temple1.png and temple2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_temple, compute_F_raw(M_temple)))
img1_size = img_temple1.shape
img2_size = img_temple2.shape
print('\tNorm =', care.compute_avg_reproj_error(M_temple, compute_F_norm(M_temple)))
print('\tMine =', care.compute_avg_reproj_error(M_temple, compute_F_mine(M_temple)))
print()

print('Average Reprojection Errors (house1.png and house2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_house, compute_F_raw(M_house)))
img1_size = img_house1.shape
img2_size = img_house2.shape
print('\tNorm =', care.compute_avg_reproj_error(M_house, compute_F_norm(M_house)))
print('\tMine =', care.compute_avg_reproj_error(M_house, compute_F_mine(M_house)))
print()

print('Average Reprojection Errors (library1.png and library2.png)')
print('\tRaw =', care.compute_avg_reproj_error(M_library, compute_F_raw(M_library)))
img1_size = img_library1.shape
img2_size = img_library2.shape
print('\tNorm =', care.compute_avg_reproj_error(M_library, compute_F_norm(M_library)))
print('\tMine =', care.compute_avg_reproj_error(M_library, compute_F_mine(M_library)))

print()

## 1-2. Visualization of epipolar lines
print('# 1-2. Visualization of epipolar lines')
print()

img_temple1_clr = cv2.imread('temple1.png', cv2.IMREAD_COLOR)
img_temple2_clr = cv2.imread('temple2.png', cv2.IMREAD_COLOR)
img_house1_clr = cv2.imread('house1.jpg', cv2.IMREAD_COLOR)
img_house2_clr = cv2.imread('house2.jpg', cv2.IMREAD_COLOR)
img_library1_clr = cv2.imread('library1.jpg', cv2.IMREAD_COLOR)
img_library2_clr = cv2.imread('library2.jpg', cv2.IMREAD_COLOR)

def visualize_epipolar_lines(img1, img2, M, title):
    global img1_size, img2_size

    img1_size = img1.shape
    img2_size = img2.shape

    select_list = [i for i in range(M.shape[0])]
    F_mine = compute_F_mine(M)

    print(title)

    end = False
    while not end:
        select = random.sample(select_list, 3)

        _img1 = np.copy(img1)
        _img2 = np.copy(img2)

        for _i in range(3):
            x, y, x_, y_ = M[select[_i]][0], M[select[_i]][1], M[select[_i]][2], M[select[_i]][3]

            color_info = (255 if _i == 0 else 0, 255 if _i == 1 else 0, 255 if _i == 2 else 0)
            cv2.circle(_img1, (int(x), int(y)), 6, color_info, 2)
            cv2.circle(_img2, (int(x_), int(y_)), 6, color_info, 2)

            # drawing epipolar line using fundamental matrix / reference - slide P61

            homogeneous_X  = np.array([x,  y,  1])
            homogeneous_X_ = np.array([x_, y_, 1])

            line_img1 = F_mine.dot(np.transpose(homogeneous_X))
            line_img2 = np.transpose(F_mine).dot(np.transpose(homogeneous_X_))

            # line_img1[0] * x + line_img1[1] * y + line_img1[2] = 0
            cv2.line(_img1, (0, int(-line_img1[2] / line_img1[1])), (_img1.shape[1] - 1, int(-(line_img1[0] * (_img1.shape[1] - 1) + line_img1[2]) / line_img1[1])), color_info, 1)
            cv2.line(_img2, (0, int(-line_img2[2] / line_img2[1])), (_img2.shape[1] - 1, int(-(line_img2[0] * (_img2.shape[1] - 1) + line_img2[2]) / line_img2[1])), color_info, 1)

            print('Red   - ' if _i == 0 else 'Green - ' if _i == 1 else 'Blue  - ', end='')
            print('[L] point: (%.1f, %.1f), line: %.6f * x + %.6f * y + %.2f'%(x, y, line_img1[0], line_img1[1], line_img1[2]), end='\t')
            print('[R] point: (%.1f, %.1f), line: %.6f * x + %.6f * y + %.2f'%(x_, y_, line_img2[0], line_img2[1], line_img2[2]))

        print()

        # reference - https://copycoding.tistory.com/159
        img_with_epipolar_line = cv2.hconcat([_img1, _img2])
        cv2.imshow(title, img_with_epipolar_line)

        key = cv2.waitKey(0)
        if key in [ord('q'), ord('Q')]:
            end = True



visualize_epipolar_lines(img_temple1_clr, img_temple2_clr, M_temple, 'Epipolar lines of Temple')
visualize_epipolar_lines(img_house1_clr, img_house2_clr, M_house, 'Epipolar lines of House')
visualize_epipolar_lines(img_library1_clr, img_library2_clr, M_library, 'Epipolar lines of Library')

cv2.destroyAllWindows()