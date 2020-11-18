'''
Computer vision assignment 2 by Yoseob Kim
A2_2d_transformation.py
Implementation of various 2-d transformation.
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A2_
'''

import cv2
import numpy as np
import math
import time
import operator
import os

def get_transformed_image(img, M):
    # initialize plane
    plane = np.empty((801, 801))
    for i in range(801):
        for j in range(801):
            plane[i][j] = 255

    '''
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0
    (3, 5) shape image
    
    center is (400, 400)
    left top is (399, 398) correspond to (center.x - shape[0] // 2, center.y - shape[1] // 2)
    right bottom is (401, 402) correspond to (center.x + shape[0] // 2, center.y + shape[1] // 2)
    
    - under condition that assumes all input image should be odd size in both height and width.
    
    [solution]
    1. make homogenous coordinate first for original image.
    2. calculate transformed coordinate and set intensity in each transformed position.
    
    '''

    origin_y_coord = np.zeros((img.shape[0], img.shape[1]))
    origin_x_coord = np.zeros((img.shape[0], img.shape[1]))
    # do not use zeros_like (data type would be uint8, however desired data type is at least uint16)

    y_range = range(400 - img.shape[0] // 2, 400 + img.shape[0] // 2 + 1)
    x_range = range(400 - img.shape[1] // 2, 400 + img.shape[1] // 2 + 1)

    # (i, 0 -> y), (j, 1 -> x)

    i = 0
    for y in y_range:
        j = 0
        for x in x_range:
            origin_y_coord[i][j] = y
            origin_x_coord[i][j] = x
            j += 1
        i += 1

    #print(origin_y_coord)
    #print(origin_x_coord)

    for i in range(img.shape[0]):       # y range
        for j in range(img.shape[1]):   # x range
            x_prime = M[0][0] * origin_x_coord[i][j] + M[0][1] * origin_y_coord[i][j] + M[0][2] * 1
            y_prime = M[1][0] * origin_x_coord[i][j] + M[1][1] * origin_y_coord[i][j] + M[1][2] * 1

            if (int(y_prime) in range(801)) and (int(x_prime) in range(801)):
                plane[int(y_prime)][int(x_prime)] = img[i][j]
            else:
                print("out of range", i, j, origin_y_coord[i][j], origin_x_coord[i][j], y_prime, x_prime)

    # plus - reducing artifact (not implemented yet)
    # check x-dir and y-dir respectively, then fill the blanks along each directions

    cv2.arrowedLine(plane, (400, 800), (400, 0), (0, 0, 0), thickness=2, tipLength=0.01)
    cv2.arrowedLine(plane, (0, 400), (800, 400), (0, 0, 0), thickness=2, tipLength=0.01)

    return plane


img_smile = cv2.imread('./smile.png', cv2.IMREAD_GRAYSCALE)

current_M = np.identity(3)
plane = get_transformed_image(img_smile, current_M)

_M = {}     # save all transformation matrix into hash map

degree_5 = math.pi / 36.
cos_5 = math.cos(degree_5)
sin_5 = math.sin(degree_5)

_M[ord('a')] = np.array([[1., 0., -5.], [0., 1., 0.], [0., 0., 1.]])                # move left by 5 pixels
_M[ord('d')] = np.array([[1., 0., +5.], [0., 1., 0.], [0., 0., 1.]])                 # move right by 5 pixels
_M[ord('w')] = np.array([[1., 0., 0.], [0., 1., -5.], [0., 0., 1.]])                # move upward by 5 pixels
_M[ord('s')] = np.array([[1., 0., 0.], [0., 1., +5.], [0., 0., 1.]])                 # move downward by 5 pixels
_M[ord('R')] = np.array([[cos_5, -sin_5, 0.], [sin_5, cos_5, 0.], [0., 0., 1.]])    # rotate CW by 5 degrees
_M[ord('r')] = np.array([[cos_5, sin_5, 0.], [-sin_5, cos_5, 0.], [0., 0., 1.]])    # rotate CCW by 5 degrees
_M[ord('F')] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])                # flip across x axis
_M[ord('f')] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])                # flip across y axis
_M[ord('x')] = np.array([[0.95, 0., 0.], [0., 1., 0.], [0., 0., 1.]])               # shrink size by 5% along x dir
_M[ord('X')] = np.array([[1.05, 0., 0.], [0., 1., 0.], [0., 0., 1.]])               # enlarge size by 5% along x dir
_M[ord('y')] = np.array([[1., 0., 0.], [0., 0.95, 0.], [0., 0., 1.]])               # shrink size by 5% along y dir
_M[ord('Y')] = np.array([[1., 0., 0.], [0., 1.05, 0.], [0., 0., 1.]])               # enlarge size by 5% along y dir

translate_minus_400 = np.array([[1., 0., -400.], [0., 1., -400.], [0., 0., 1.]])
translate_plus_400 = np.array([[1., 0., +400.], [0., 1., +400.], [0., 0., 1.]])

for mat in _M:
    _M[mat] = np.matmul(translate_plus_400, np.matmul(_M[mat], translate_minus_400))


# reference for opencv keyboard input (below link)
# https://subscription.packtpub.com/book/application_development/9781788474443/1/ch01lvl1sec18/handling-user-input-from-a-keyboard
end = False
while not end:
    print(current_M)
    plane = get_transformed_image(img_smile, current_M)
    cv2.imshow("various 2-d transformations", plane)
    key = cv2.waitKey(0)
    print(key)

    # upper case input not working on Macintosh environment. (execute on Windows!)
    if key == ord('H'):
        current_M = np.identity(3)
    elif key == ord('Q'):
        end = True
    else:
        if key in _M:
            current_M = np.matmul(_M[key], current_M)

cv2.destroyAllWindows()