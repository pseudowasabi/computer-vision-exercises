'''
Computer vision assignment 1 by Yoseob Kim
A1_image_filtering.py
'''

import cv2
import numpy as np
import math
from scipy.ndimage import gaussian_filter

'''
Step 1. 
define functions - cross_correlation_1d, cross_correlation_2d, get_gaussian_filter_1d, get_gaussian_filter_2d
'''

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
    # 1. padding image
    img_size = [img.shape[0], img.shape[1]]
    kernel_size = kernel.shape[0]
    padded_size = [img_size[0] + kernel_size - 1, img_size[1] + kernel_size - 1]
    padded_img = np.zeros((padded_size[0], padded_size[1]))

    for i in range(padded_size[0]):
        i_prime = i - (kernel_size // 2)
        for j in range(padded_size[1]):
            j_prime = j - (kernel_size // 2)
            if 0 <= i_prime < img_size[0]:
                if 0 <= j_prime < img_size[1]:
                    #padded_img[i][j] = -img[i_prime][j_prime]
                    padded_img[i][j] = img[i_prime][j_prime]
                elif j_prime < 0: # West
                    padded_img[i][j] = img[i_prime][0]
                elif j_prime >= img_size[1]: # East
                    padded_img[i][j] = img[i_prime][img_size[1] - 1]
            elif i_prime < 0:
                if 0 <= j_prime < img_size[1]: # North
                    padded_img[i][j] = img[0][j_prime]
                elif j_prime < 0: # NW
                    padded_img[i][j] = img[0][0]
                elif j_prime >= img_size[1]: # NE
                    padded_img[i][j] = img[0][img_size[1] - 1]
            elif i_prime >= img_size[0]:
                if 0 <= j_prime < img_size[1]: # South
                    padded_img[i][j] = img[img_size[0] - 1][j_prime]
                elif j_prime < 0: # SW
                    padded_img[i][j] = img[img_size[0] - 1][0]
                elif j_prime >= img_size[1]: # SE
                    padded_img[i][j] = img[img_size[0] - 1][img_size[1] - 1]
    #print(padded_img)

    # 2. apply cross correlation using iteration
    filtered_img = np.zeros((img_size[0], img_size[1]))
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            # at each (x, y) point, cross correlation using kernel is calculated.
            for i in range(x, x + kernel_size):
                for j in range(y, y + kernel_size):
                    filtered_img[x][y] += (padded_img[i][j] * kernel[i - x][j - y])
            #print('%.04f'%(filtered_img[x][y]), end=' ')
        #print()
    #print(filtered_img)

    filtered_img = filtered_img.astype('uint8')
    return filtered_img # need to be checked

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

'''
Step 2. 
load images in grayscale
'''

img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)

k_size = [5, 11, 17]
sigmas = [1, 6, 11]
font = cv2.FONT_HERSHEY_SIMPLEX
final_filtered_lenna = np.zeros((img_lenna.shape[0]*3, img_lenna.shape[1]*3))
final_filtered_shapes = np.zeros((img_shapes.shape[0]*3, img_shapes.shape[1]*3))

for i in range(1):
    for j in range(1):
        #k = k_size[i]
        #s = sigmas[j]
        k = 17
        s = 11
        kernel = get_gaussian_filter_2d(k, s)

        filtered_img_lenna = cross_correlation_2d(img_lenna, kernel)
        #filtered_img_shapes = cross_correlation_2d(img_shapes, kernel)

        cv2.putText(filtered_img_lenna, '{0}x{0}, s={1}'.format(k, s), (30, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        #cv2.putText(filtered_img_shapes, '{0}x{0}, s={1}'.format(k, s), (30, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        #for x in range(img_lenna.shape[0]*i, img_lenna.shape[0]*(i+1)):
        #    for y in range(img_lenna.shape[1]*i, img_lenna.shape[1]*(i+1)):
        #        final_filtered_lenna[x][y] = filtered_img_lenna[x-img_lenna.shape[0]*i][y-img_lenna.shape[1]*i]
        print('{0}x{0}, s={1}'.format(k, s))
        cv2.imshow("aa", filtered_img_lenna)
        #cv2.waitKey(0)
        #for x in range(img_shapes.shape[0]*i, img_shapes.shape[0]*(i+1)):
        #    for y in range(img_shapes.shape[1]*i, img_shapes.shape[1]*(i+1)):
        #        final_filtered_shapes[x][y] = filtered_img_shapes[x-img_shapes.shape[0]*i][y-img_shapes.shape[1]*i]

#cv2.imshow("Gaussian filtering of Lenna by Yoseob Kim", final_filtered_lenna)
#cv2.imshow("Gaussian filtering of Shapes by Yoseob Kim", final_filtered_shapes)

blur = cv2.GaussianBlur(img_lenna, (5, 5), 11)
#cv2.imshow("by cv2", blur)


cv2.waitKey(0)
cv2.destroyAllWindows()