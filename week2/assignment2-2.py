from matplotlib.pyplot import imshow, show, colorbar
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.transform import resize
from skimage.util import random_noise
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage import filters
from scipy.signal import medfilt
from pylab import ginput
import numpy as np
from skimage import exposure, img_as_float
import cv2
import timeit
import time

# -----------------------------------------------------------------------
#                             Task 2.3
#------------------------------------------------------------------------

img = imread("images/eight.tif")
imgST = random_noise(img, mode='s&p')
imgGaussian = random_noise(img, mode='gaussian')
def task_2_3():


    #imshow(imgGaussian,  cmap='gray')
    #show()
    #imshow(imgST, cmap='gray')
    #show()

    matrix = np.zeros((100,13))
    k = 0
    while (k < 100):
        counter = 0
        for i in range(5):
            for j in range (5):
                # We only want to work with kernel of odd size
                if ((i+j) % 2 == 0):
                    now = time.time()
                    blur = cv2.blur(imgST, (i+1, j+1))
                    end = time.time()
                    matrix[k][counter] = end - now
                    counter += 1

                    # in case you want to plot the graph
                    #imshow(blur, cmap='gray')
                    #plt.title("kernel size" + "(" + str(i+1) + "," + str(1+j) + ")")
                    #show()
                else:
                    continue
        k += 1

    matrix_mean = np.mean(matrix,axis=0)
    plt.plot([1,3,5,7,9,11,13,15,17,19,21,23,25],matrix_mean)
    plt.title('Computation time of average filter')
    plt.xlabel('Kernel size')
    plt.ylabel('Mean time in seconds')
    plt.show()


    a = np.zeros((5, 13))
    j = 0
    while (j < 5):
        index = 0
        for i in range(1,27, 2):
            now = time.time()
            newimg = medfilt(imgGaussian,i)
            end = time.time()
            a[j][index] = end - now
            index += 1
            #imshow(newimg, cmap='gray')
            #plt.title("Kernel size:" + str(i))
            #show()
        j = j + 1

    mean = np.mean(a,axis=0)
    print (mean)
    plt.plot([1,3,5,7,9,11,13,15,17,19,21,23,25],mean)
    plt.xlabel('Size of kernel')
    plt.ylabel('Mean time in seconds')
    plt.title('Plot over kernel size vs computation time')
    plt.show()
    return 42

# -----------------------------------------------------------------------
#                             Task 2.4
#------------------------------------------------------------------------

def task_2_4():

    # Gaus kernels
    gausBlurGaus1 = cv2.GaussianBlur(imgGaussian, (1, 1), 10)
    gausBlurGaus3 = cv2.GaussianBlur(imgGaussian, (3, 3), 10)
    gausBlurGaus5 = cv2.GaussianBlur(imgGaussian, (5, 5), 10)
    gausBlurGaus7 = cv2.GaussianBlur(imgGaussian, (7, 7), 10)

    # Gaus blur
    plt.subplot(131), plt.imshow(imgGaussian, cmap='gray'), plt.title('Image with Gaussian noise')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(gausBlurGaus1, cmap='gray'), plt.title('Kernel size = 1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(gausBlurGaus3, cmap='gray'), plt.title('Kernel size = 9')
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.subplot(121), plt.imshow(gausBlurGaus5, cmap='gray'), plt.title('Kernel size = 25')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gausBlurGaus7, cmap='gray'), plt.title('Kernel size = 49')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # Gaus kernels
    gausBlurST1 = cv2.GaussianBlur(imgST, (1, 1), 5)
    gausBlurST3 = cv2.GaussianBlur(imgST, (3, 3), 5)
    gausBlurST5 = cv2.GaussianBlur(imgST, (7, 7), 5)
    gausBlurST7 = cv2.GaussianBlur(imgST, (7, 7), 5)

    # Salt and peper
    plt.subplot(131), plt.imshow(imgST, cmap='gray'), plt.title('Image with S%T noice')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(gausBlurST1, cmap='gray'), plt.title('Kernel size = 1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(gausBlurST3, cmap='gray'), plt.title('Kernel size = 9')
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.subplot(121), plt.imshow(gausBlurST5, cmap='gray'), plt.title('Kernel size = 25')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gausBlurST7, cmap='gray'), plt.title('Kernel size = 49')
    plt.xticks([]), plt.yticks([])
    plt.show()

# -----------------------------------------------------------------------
#                             Task 2.5
#------------------------------------------------------------------------

def task_2_5():
    pass


# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------
task_2_3()
#task_2_4()
#task_2_5()
