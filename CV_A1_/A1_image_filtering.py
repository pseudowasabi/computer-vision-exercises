'''
Computer vision assignment 1 by Yoseob Kim
A1_image_filtering.py
'''

import cv2
import numpy as np
# not allowed to use any assignment-related numpy built-in functions

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
    img_size = img.shape[0]
    kernel_size = kernel.shape[0]

    # 2. padding original image (img)
    padded_size = img_size + kernel_size - 1
    if dir == 0:
        padded_img = np.zeros((img_size, padded_size))
    elif dir == 1:
        padded_img = np.zeros((padded_size, img_size))

    for i in range(img_size):
        for j in range(padded_size):
            j_prime = j - (kernel_size // 2)
            if 0 <= j_prime < img_size:
                if dir == 0:
                    padded_img[i][j] = img[i][j_prime]
                elif dir == 1:
                    padded_img[j][i] = img[j_prime][i]
            elif j_prime < 0:
                if dir == 0:
                   padded_img[i][j] = img[i][0]
                elif dir == 1:
                    padded_img[j][i] = img[0][i]
            elif j_prime >= img_size:
                if dir == 0:
                    padded_img[i][j] = img[i][img_size - 1]
                elif dir == 1:
                    padded_img[j][i] = img[img_size - 1][i]
    print(padded_img)
    print()
    # 3. apply cross correlation using iteration
    filtered_img = np.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
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


def cross_correlation_2d(img, kernel):
    # 1. padding image

    # 2. apply cross correlation using iteration

    pass

# if np.arange, sqrt, pi, exp, multiply functions are not allowed to use,
# below code should be modified !!!
def get_gaussian_filter_1d(size, sigma):
    # return gaussian filter for horizontal 1D
    kernel = np.arange(-(size//2), size//2+1)
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp((-1) * np.multiply(kernel, kernel) / (2 * (sigma ** 2)))
    '''
    for i in range(size):
        print('%.04f'%(kernel[i]))
    '''
    return kernel

def get_gaussian_filter_2d(size, sigma):
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    kernel = np.zeros((size, size))
    #print(kernel.shape)
    for i in range(size):
        for j in range(size):
            kernel[i][j] = kernel_1d[i] * kernel_1d[j]
            #print('%.04f'%kernel[i][j], end = ' ')
        #print()
    return kernel

'''
Step 2. 
load images in grayscale
'''

img_lenna = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
img_shapes = cv2.imread('./shapes.png', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Lenna in grayscale", img_lenna)
# cv2.imshow("Shapes", img_shapes)
print(img_lenna.shape)


'''
Step 3.
implementation of requirements
'''

#get_gaussian_filter_2d(9, 1)

size = 8
img_a = np.arange(-(size//2), (size//2) + 1)
img = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        img[i][j] = img_a[i] * img_a[j]
#print(img.shape)
print("original img")
print(img)

'''
# 1-d kernel cross correlationing
kernel = get_gaussian_filter_1d(5, 1)
kernel = np.array([kernel]).transpose()
print(kernel)
cross_correlation_1d(img, kernel)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()