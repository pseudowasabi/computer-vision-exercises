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

    x_range = range(400 - img.shape[0] // 2, 400 + img.shape[0] // 2)
    y_range = range(400 - img.shape[1] // 2, 400 + img.shape[1] // 2)

    i = 0
    for x in x_range:
        j = 0
        for y in y_range:
            plane[x][y] = img[i][j]
            j += 1
        i += 1

    cv2.arrowedLine(plane, (400, 800), (400, 0), (0, 0, 0), thickness=2, tipLength=0.01)
    cv2.arrowedLine(plane, (0, 400), (800, 400), (0, 0, 0), thickness=2, tipLength=0.01)

    return plane


img_smile = cv2.imread('./smile.png', cv2.IMREAD_GRAYSCALE)

M = np.identity(3)
plane = get_transformed_image(img_smile, M)

_M = {}     # save all transformation matrix into hash map

_M['translate_left'] = np.array([[1., 0., -5.], [0., 1., 0.], [0., 0., 1.]])
_M['translate_right'] = np.array([[1., 0., 5.], [0., 1., 0.], [0., 0., 1.]])
_M['translate_up'] = np.array([[1., 0., 0.], [0., 1., -5.], [0., 0., 1.]])
_M['translate_down'] = np.array([[1., 0., 0.], [0., 1., 5.], [0., 0., 1.]])

degree_5 = math.pi / 36.
cos_5 = math.cos(degree_5)
sin_5 = math.sin(degree_5)
_M['rotate_ccw'] = np.array([[cos_5, -sin_5, 0.], [sin_5, cos_5, 0.], [0., 0., 1.]])
_M['rotate_cw'] = np.array([[cos_5, sin_5, 0.], [-sin_5, cos_5, 0.], [0., 0., 1.]])
_M['flip_y_axis'] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
_M['flip_x_axis'] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
_M['identity'] = np.identity(3)
_M['shrink_x_dir'] = np.array([[0.95, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
_M['enlarge_x_dir'] = np.array([[1.05, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
_M['shrink_y_dir'] = np.array([[1., 0., 0.], [0., 0.95, 0.], [0., 0., 1.]])
_M['enlarge_y_dir'] = np.array([[1., 0., 0.], [0., 1.05, 0.], [0., 0., 1.]])

# reference for opencv keyboard input (below link)
# https://subscription.packtpub.com/book/application_development/9781788474443/1/ch01lvl1sec18/handling-user-input-from-a-keyboard
end = False
while not end:
    cv2.imshow("various 2-d transformations", plane)
    key = cv2.waitKey(0)
    if key == ord('a'):     # move left by 5 pixels
        plane = get_transformed_image(img_smile, _M['translate_left'])
    elif key == ord('d'):   # move right by 5 pixels
        pass
    elif key == ord('w'):   # move upward by 5 pixels
        pass
    elif key == ord('s'):   # move downward by 5 pixels
        pass
    elif key == ord('r'):   # rotate CCW by 5 degrees
        pass
    elif key == ord('R'):   # rotate CW by 5 degrees
        pass
    elif key == ord('f'):   # flip across y axis
        pass
    elif key == ord('F'):   # flip across x axis
        pass
    elif key == ord('x'):   # shrink size by 5% along x dir
        pass
    elif key == ord('X'):   # enlarge size by 5% along x dir
        pass
    elif key == ord('y'):   # shrink size by 5% along y dir
        pass
    elif key == ord('Y'):   # enlarge size by 5% along y dir
        pass
    elif key == ord('H'):   # set to initial state
        pass
    elif key == ord('Q'):   # quit
        end = True