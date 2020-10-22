'''
Computer vision assignment 1 by Yoseob Kim
GitHub Link: https://github.com/pseudowasabi/computer-vision-exercises/tree/master/CV_A1_
'''


import numpy as np
import math
import operator

###
# 1-1. Image Filtering by Cross-Correlation
###

def cross_correlation_1d(img, kernel):
    # 1. check direction of kernel and flatten
    dir = 0     # dir (direction) is 0 when kernel is horizontal, 1 when vertical.
    if len(kernel.shape) != 1: # if the shape is not an 1-d ndarray,
        if kernel.shape[1] == 1: # vertical kernel
            dir = 1
        kernel = kernel.flatten() # flatten to 1-d ndarray
    size0 = img.shape[0]
    size1 = img.shape[1]
    k_size = kernel.shape[0]
    diff = k_size // 2

    # 2. padding original image (img)
    if dir == 0:
        padded_size = size1 + k_size - 1
        padded_img = np.zeros((size0, padded_size))

        j_range0 = range(diff)
        j_range1 = range(size1)
        j_range2 = range(size1+diff, size1+2*diff)

        for i in range(size0):
            for j in j_range0:
                padded_img[i][j] = img[i][0]
            for j in j_range1:
                padded_img[i][j+diff] = img[i][j]
            for j in j_range2:
                padded_img[i][j] = img[i][size1-1]
    else: # dir == 1 (vertical cases)
        padded_size = size0 + k_size - 1
        padded_img = np.zeros((padded_size, size1))

        j_range0 = range(diff)
        j_range1 = range(size0)
        j_range2 = range(size0+diff, size0+2*diff)

        for j in range(size1):
            for i in j_range0:
                padded_img[i][j] = img[0][j]
            for i in j_range1:
                padded_img[i+diff][j] = img[i][j]
            for i in j_range2:
                padded_img[i][j] = img[size0-1][j]

    #print(padded_img)
    #print()

    # 3. apply cross correlation using iteration
    filtered_img = np.zeros((size0, size1))
    for x in range(size0):
        for y in range(size1):
            if dir == 0:    # horizontal kernel
                _sum = 0
                for j in range(k_size):
                    _sum = operator.__add__(_sum, operator.__mul__(padded_img[x][y+j], kernel[j]))
                filtered_img[x][y] = _sum
            elif dir == 1:  # vertical kernel
                _sum = 0
                for i in range(k_size):
                    _sum = operator.__add__(_sum, operator.__mul__(padded_img[x+i][y], kernel[i]))
                filtered_img[x][y] = _sum
            #print('%.04f'%filtered_img[i][j], end=' ')
        #print()
    #print(filtered_img)

    return filtered_img

def image_padding_2d(img, padd_width, type=0):
    # type == 0 ; padd image with the nearest pixel inside the image.
    # type != 0 ; padd image with zeros.
    size0 = img.shape[0]
    size1 = img.shape[1]

    p_size0 = size0 + 2 * padd_width
    p_size1 = size1 + 2 * padd_width
    padded_img = np.zeros((p_size0, p_size1))
    diff = padd_width

    j_range0 = range(diff)
    j_range1 = range(size1)
    j_range2 = range(size1 + diff, size1 + 2 * diff)
    for i in range(size0):
        i_prime = i + diff
        for j in j_range1:
            # center
            padded_img[i_prime][j + diff] = img[i][j]
        if type == 0:
            for j in j_range0:
                # west
                padded_img[i_prime][j] = img[i][0]
            for j in j_range2:
                # east
                padded_img[i_prime][j] = img[i][size1 - 1]
    if type == 0:
        for i in range(diff):  # upper
            for j in j_range0:
                padded_img[i][j] = img[0][0]
            for j in j_range1:
                padded_img[i][j + diff] = img[0][j]
            for j in j_range2:
                padded_img[i][j] = img[0][size1 - 1]
        for i in range(size0 + diff, size0 + 2 * diff):  # lower
            for j in j_range0:
                padded_img[i][j] = img[size0 - 1][0]
            for j in j_range1:
                padded_img[i][j + diff] = img[size0 - 1][j]
            for j in j_range2:
                padded_img[i][j] = img[size0 - 1][size1 - 1]
    # print(padded_img)

    return padded_img

def cross_correlation_2d(img, kernel):
    # 1. padding image - about O(N^2).
    size0 = img.shape[0]
    size1 = img.shape[1]
    k_size = kernel.shape[0]
    padded_img = image_padding_2d(img, k_size // 2)

    # 2. apply cross correlation using iteration - O(N^2 * K^2).
    filtered_img = np.zeros((size0, size1))

    y_range = range(size1)
    elapsed_ = list(range(0, size0, size0//20))
    for x in range(size0):
        for y in y_range:
            # at each (x, y) point, cross correlation using kernel is calculated.
            _sum = 0
            for i in range(x, x + k_size):
                for j in range(y, y + k_size):
                    _sum = operator.__add__(_sum, operator.__mul__(padded_img[i][j], kernel[i - x][j - y]))
            filtered_img[x][y] = _sum
            #print('%.04f'%(filtered_img[x][y]), end=' ')
        #print()
        if x in elapsed_:
            print('.', end='')
    #print(filtered_img)

    #filtered_img = filtered_img.astype('uint8')
    return filtered_img



###
# 1.2 The Gaussian Filter
###

def get_gaussian_filter_1d(size, sigma):
    # return gaussian filter for horizontal 1-d
    x = np.linspace(-size//2+1, size//2, size)
    kernel = np.zeros((size))

    coeff1 = 1 / (math.sqrt(2 * math.pi) * sigma)
    coeff2 = -(1 / (2 * (sigma ** 2)))
    _sum = 0
    for i in range(size):
        kernel[i] = coeff1 * math.exp(coeff2 * (x[i] ** 2))
        #print('%.04f'%kernel[i], end=' ')
        _sum += kernel[i]

    coeff_normalize = 1 / _sum
    for i in range(size): # normalize
        kernel[i] *= coeff_normalize
    return np.array(kernel)

def get_gaussian_filter_2d(size, sigma):
    # return gaussian filter for 2-d
    x = np.linspace(-size//2+1, size//2, size)
    kernel = np.zeros((size, size))

    coeff1 = 1 / (2 * math.pi * (sigma ** 2))
    coeff2 = -(1 / (2 * (sigma ** 2)))
    for i in range(size):
        x[i] = coeff2 * (x[i] ** 2)

    _sum = 0
    for i in range(size):
        for j in range(size):
            kernel[i][j] = coeff1 * math.exp(x[i] + x[j])
            #print('%.04f'%kernel[i][j], end=' ')
            _sum += kernel[i][j]
        #print()

    coeff_normalize = 1 / _sum
    for i in range(size):
        for j in range(size):
            kernel[i][j] *= coeff_normalize
    return kernel


### ***
# for Part #2, #3
###

##
## 1. Gaussian filtering

def my_gaussian_filtering(img, kernel_size, sigma):
    kernel_h = get_gaussian_filter_1d(kernel_size, sigma)
    kernel_v = np.array([kernel_h]).transpose()

    filtered_img = cross_correlation_1d(img, kernel_h)
    filtered_img = cross_correlation_1d(filtered_img, kernel_v)

    return filtered_img


##
## 2. Sobel filtering

# ** set Sobel filter kernels as global variables
derivative_kernel = [-1, 0, +1]  # correlation ; [1, 0, -1] is for convolution.
blurring_filter = [1, 2, 1]
# for faster operation, use superposition of 1-d kernel filtering method.
sobel_x_0 = np.array(derivative_kernel)
sobel_x_1 = np.array([blurring_filter]).transpose()
sobel_y_0 = np.array(blurring_filter)
sobel_y_1 = np.array([derivative_kernel]).transpose()
# below is 2-d kernel for sobel cross-correlation kernel.
# sobel_x = sobel_x_1.dot(np.array([sobel_x_0]))
# sobel_y = sobel_y_1.dot(np.array([sobel_y_0]))
# print(sobel_x)
# print(sobel_y)

def my_sobel_filtering(img):
    # derivatives along x direction
    sobel_img_x = cross_correlation_1d(img, sobel_x_0)
    sobel_img_x = cross_correlation_1d(sobel_img_x, sobel_x_1)

    # derivatives along y direction
    sobel_img_y = cross_correlation_1d(img, sobel_y_0)
    sobel_img_y = cross_correlation_1d(sobel_img_y, sobel_y_1)

    return sobel_img_x, sobel_img_y