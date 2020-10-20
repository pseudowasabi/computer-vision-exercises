'''
Computer vision assignment 1 by Yoseob Kim
A1_image_filtering.py
'''

import cv2
import numpy as np
import math

###
# 1-1. Image Filtering by Cross-Correlation
###

def cross_correlation_1d(img, kernel):
    # filtering using 1d kernel

    # 1. check direction of kernel and flatten
    dir = 0     # dir (direction) is 0 when kernel is horizontal, 1 when vertical.
    if len(kernel.shape) != 1: # if the shape is not an 1-d ndarray,
        if kernel.shape[1] == 1: # vertical kernel
            dir = 1
        kernel = kernel.flatten() # flatten to 1-d ndarray
    #img_size_0 = img.shape[0]
    #img_size_1 = img.shape[1]
    kernel_size = kernel.shape[0]

    # 2. padding original image (img)
    #padded_size = img_size_0 + kernel_size - 1
    if dir == 0:
        padded_size = img.shape[1] + kernel_size - 1
        padded_img = np.zeros((img.shape[0], padded_size))

        for i in range(img.shape[0]):
            for j in range(padded_size):
                j_prime = j - (kernel_size // 2)
                if 0 <= j_prime < img.shape[1]:
                    padded_img[i][j] = img[i][j_prime]
                elif j_prime < 0:
                    padded_img[i][j] = img[i][0]
                else:
                    padded_img[i][j] = img[i][img.shape[1]-1]
    elif dir == 1:
        padded_size = img.shape[0] + kernel_size - 1
        padded_img = np.zeros((padded_size, img.shape[1]))

        for j in range(img.shape[1]):
            for i in range(padded_size):
                i_prime = i - (kernel_size // 2)
                if 0 <= i_prime < img.shape[0]:
                    padded_img[i][j] = img[i_prime][j]
                elif i_prime < 0:
                    padded_img[i][j] = img[0][j]
                else:
                    padded_img[i][j] = img[img.shape[0]-1][j]

    print(padded_img)
    print()

    # 3. apply cross correlation using iteration
    filtered_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if dir == 0:
                filtered_img[i][j] = sum(np.multiply(padded_img[i][j:j+kernel_size], kernel))
            elif dir == 1:
                #pass
                k_size_patch_from_padded_img = np.zeros(kernel_size)
                for x in range(i, i+kernel_size):
                    k_size_patch_from_padded_img[x-i] = padded_img[x][j]
                #print(k_size_patch_from_padded_img)
                filtered_img[i][j] = sum(np.multiply(k_size_patch_from_padded_img, kernel))
            #print('%.04f'%filtered_img[i][j], end=' ')
        #print()
    #print(filtered_img)

    filtered_img = filtered_img.astype('uint8')
    return filtered_img # need to be checked


def cross_correlation_2d(img, kernel):
    # 1. padding image - about O(N^2).
    size0 = img.shape[0]
    size1 = img.shape[1]
    k_size = kernel.shape[0]
    p_size0 = size0 + k_size - 1
    p_size1 = size1 + k_size - 1
    padded_img = np.zeros((p_size0, p_size1))
    diff = k_size // 2

    j_range0 = range(diff)
    j_range1 = range(size1)
    j_range2 = range(size1+diff, size1+2*diff)
    for i in range(size0):
        i_prime = i + diff
        for j in j_range0:
            # west
            padded_img[i_prime][j] = img[i][0]
        for j in j_range1:
            # center
            padded_img[i_prime][j+diff] = img[i][j]
        for j in j_range2:
            # east
            padded_img[i_prime][j] = img[i][size1-1]
    for i in range(diff): # upper
        for j in j_range0:
            padded_img[i][j] = img[0][0]
        for j in j_range1:
            padded_img[i][j+diff] = img[0][j]
        for j in j_range2:
            padded_img[i][j] = img[0][size1-1]
    for i in range(size0+diff, size0+2*diff): # lower
        for j in j_range0:
            padded_img[i][j] = img[size0-1][0]
        for j in j_range1:
            padded_img[i][j+diff] = img[size0-1][j]
        for j in j_range2:
            padded_img[i][j] = img[size0-1][size1-1]
    #print(padded_img)

    # 2. apply cross correlation using iteration - O(N^2 * K^2).
    filtered_img = np.zeros((size0, size1))

    y_range = range(size1)
    for x in range(size0):
        for y in y_range:
            # at each (x, y) point, cross correlation using kernel is calculated.
            _sum = 0
            for i in range(x, x + k_size):
                for j in range(y, y + k_size):
                    _sum += (padded_img[i][j] * kernel[i - x][j - y])
            filtered_img[x][y] = _sum
            #print('%.04f'%(filtered_img[x][y]), end=' ')
        #print()
    #print(filtered_img)

    #filtered_img = filtered_img.astype('uint8')
    return filtered_img


###
# 1.2 The Gaussian Filter
###

## a, b)
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
    return kernel

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


## c) Print results of gaussian filter functions
kernel_1d_51 = get_gaussian_filter_1d(5, 1)
kernel_2d_51 = get_gaussian_filter_2d(5, 1)
print('Computer Vision A#1 // Yoseob Kim')
print('Part #1. Image Filtering\n')

print('1-2. c) 1-d gaussian kernel (size=5, sigma=1)')
print(kernel_1d_51, end='\n\n')

print('1-2. c) 2-d gaussian kernel (size=5, sigma=1)')
print(kernel_2d_51, end='\n\n')

## d) Perform 9 different Gaussian filtering
print('1-2. d) perform 9 different gaussian filtering to "lenna.png"')
k_size = [5, 11, 17]
sigmas = [1, 6, 11]
font = cv2.FONT_HERSHEY_SIMPLEX

# for "lenna.png"
print('filtering for "lenna.png" initiate.')
img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
final_filtered_lenna = np.zeros((img_lenna.shape[0]*3, img_lenna.shape[1]*3))

for i in range(3):
    for j in range(3):
        k = k_size[i]
        s = sigmas[j]
        kernel = get_gaussian_filter_2d(k, s)

        print('filtering... kernel size: {0}, sigma: {1}'.format(k, s), end='')
        filtered_img_lenna = cross_correlation_2d(img_lenna, kernel).astype(dtype='uint8')
        cv2.putText(filtered_img_lenna, '{0}x{0}, s={1}'.format(k, s), (30, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        for x in range(img_lenna.shape[0] * i, img_lenna.shape[0] * (i+1)):
            u = x - img_lenna.shape[0] * i
            for y in range(img_lenna.shape[1] * j, img_lenna.shape[1] * (j+1)):
                v = y - img_lenna.shape[1] * j
                final_filtered_lenna[x][y] = filtered_img_lenna[u][v]
        print(' ---> done.')

cv2.imwrite('./result/part_1_gaussian_filtered_lenna.png', final_filtered_lenna)
print('filtered image of "lenna.png" saved to ./result/ directory.\n')
cv2.imshow('Gaussian filtering of "Lenna.png" by Yoseob Kim', final_filtered_lenna)
cv2.waitKey(0)

# for "shapes.png"
print('filtering for "shapes.png" initiate.')
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)
final_filtered_shapes = np.zeros((img_shapes.shape[0] * 3, img_shapes.shape[1] * 3))

for i in range(3):
    for j in range(3):
        k = k_size[i]
        s = sigmas[j]
        kernel = get_gaussian_filter_2d(k, s)

        print('filtering... kernel size: {0}, sigma: {1}'.format(k, s), end='')
        filtered_img_shapes = cross_correlation_2d(img_shapes, kernel).astype(dtype='uint8')
        cv2.putText(filtered_img_shapes, '{0}x{0}, s={1}'.format(k, s), (30, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        for x in range(img_shapes.shape[0] * i, img_shapes.shape[0] * (i + 1)):
            u = x - img_shapes.shape[0] * i
            for y in range(img_shapes.shape[1] * j, img_shapes.shape[1] * (j + 1)):
                v = y - img_shapes.shape[1] * j
                final_filtered_shapes[x][y] = filtered_img_shapes[u][v]
        print(' ---> done.')

cv2.imwrite('./result/part_1_gaussian_filtered_shapes.png', final_filtered_shapes)
print('filtered image of "shapes.png" saved to ./result/ directory.\n')
cv2.imshow('Gaussian filtering of "Shapes.png" by Yoseob Kim', final_filtered_shapes)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
size = 8
img_a = np.arange(-(size//2), (size//2) + 1)
img = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        img[i][j] = img_a[i] * img_a[j]

print('original img')
print(img)
kernel = get_gaussian_filter_2d(5, 1)
print('padded img')
cross_correlation_2d(img, kernel)
'''