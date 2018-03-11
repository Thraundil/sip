import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imshow, show
from skimage.io import imread
from skimage import feature
from skimage.transform import resize
from scipy.stats import norm
from skimage import exposure, img_as_float, filters
import math
from scipy.ndimage import filters
import cv2
from scipy import fftpack
from scipy.fftpack import fftfreq
import copy



# -----------------------------------------------------------------------
#                             Task 3.1
#------------------------------------------------------------------------
def task_3_1(a,b,c):
  #img = imread('images/basic_shapes.png')
  #imshow(img, cmap='gray')
  #plt.title('orignal image')
  #show()

  my_image = np.zeros((20,20))
  my_image[10,10] = 1
  #my_image[7:-7, 7:-7] = 1
  imshow(my_image,cmap='gray',interpolation='nearest')
  plt.title('centered white square')
  show()
  if (c == 0):
    # x -> right, y -> up
    arr = copy.copy(my_image)
    arr[a:, b:] = arr[:-a, :-b]
    arr[0:a, 0:] = 0
    arr[0:, 0:b] = 0
    imshow(arr, cmap='gray',interpolation='nearest')
    plt.title('translated')
    show()

    # x -> left, y -> down
    arr1 = imread('images/basic_shapes.png')
    arr1[:-a, :-b] = arr1[a:, b:]
    arr1[-a:, 0:] = 0
    arr1[0:, -b:] = 0
    #imshow(arr1, cmap='gray')
    #show()
  else:
    new_img = np.zeros((my_image.shape[0],my_image.shape[1]))
    for i in range(my_image.shape[0]):
      for j in range(my_image.shape[1]):
        if ((i - a) < 0) or ((j-b) < 0):
          new_img[i][j] = 0
        else:
          new_img[i][j] = my_image[np.around(i-a)][np.around(j-b)]

    #imshow(new_img,cmap='gray')
    #show()

  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(my_image, cmap='gray', interpolation='nearest')
  ax[0].set_title('Orignal image')
  ax[1].imshow(arr, cmap='gray', interpolation='nearest')
  ax[1].set_title('Translated with (2,2)')
  plt.show()

# -----------------------------------------------------------------------
#                             Task 3.2
#------------------------------------------------------------------------


def task_3_1_3(input_array,shift):
    xs,ys = np.meshgrid(fftfreq(input_array.shape[1]),fftfreq(input_array.shape[0]))
    print (xs.shape)
    # The j is pythons convention for complex numbers
    fourier_shift = np.exp(1j*2*np.pi*((-shift[0]*ys)+(-shift[1]*xs)))
    output_array = np.real(np.fft.ifft2(np.fft.fft2(input_array)*fourier_shift))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(input_array,cmap='gray', interpolation='nearest')
    ax[0].set_title('Orignal image')
    ax[1].imshow(output_array,cmap='gray', interpolation='nearest')
    ax[1].set_title('Fourier method')
    plt.show()



# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------
# Number of pixel to move in the x and y direction
# C indicates whether or not we move by integers or real values.
img = imread('images/basic_shapes.png')
#task_3_1(2,2,0)
image = np.zeros((20, 20))
image[10, 10] = 1
task_3_1_3(img,[15.6,14.2])
