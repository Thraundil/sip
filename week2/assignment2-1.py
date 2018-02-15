import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from scipy.stats import norm
from skimage import exposure, img_as_float


# -----------------------------------------------------------------------
#                         General Functions
#------------------------------------------------------------------------

def cumulative_histogram(hist, N, M):
  # Compute cumulative histogram and normalize with respect to number of pixels
  cumsum = np.cumsum(hist)   
  return cumsum / (N*M)


def pseudo_inverse_cdf(cdf, l):
  # Get subset of pixel values where f(s) >= l and get minimum of that
  subset  = np.take(cdf, np.where(cdf >= l))
  return np.min(subset) 


# -----------------------------------------------------------------------
#                             Task 1.3
#------------------------------------------------------------------------

def task_1_3():
  # Read pout.tif and compute histogram
  img = imread('images/pout.tif')
  hist, bins = np.histogram(img.ravel(),256,[0,256])
  
  # Compute its cumulative histogram
  cumsum_hist = cumulative_histogram(hist, img.shape[0], img.shape[1])
  
  # Display the cumulative histogram
  plt.plot(cumsum_hist)
  plt.title('Cumulative histogram for \'pout.tif\'')
  plt.xlabel('Pixel range')
  plt.ylabel('Cumulative sum')
  plt.xlim([0,256])
  plt.show()

  
# -----------------------------------------------------------------------
#                             Task 1.4
#------------------------------------------------------------------------

def task_1_4():
  # Read pout.tif(I) and compute histogram and CumSum(C)
  I = imread('images/pout.tif')
  hist, bins  = np.histogram(I.ravel(),256,[0,256])
  C           = np.cumsum(hist)
  CNormalized = C*256/C.max()

  # Prints the original image 'pout.tif'
  plt.imshow(I, cmap='gray')
  plt.show()

  # Shows the normalized CDF (to 256)
  plt.plot(CNormalized)
  plt.show()

  # Computes the floating-point image C(I), such that the Intensity at each (x,y) is C(I(x,y))
  newI  = I
  (A,B) = I.shape
  for x in range(0,A):
    for y in range(0,B):
      newI[x][y] = CNormalized[(I[x][y])]

  # Prints the new Image, based on 'pout.tif'
  plt.imshow(newI, cmap='gray')
  plt.show()



# -----------------------------------------------------------------------
#                             Task 1.5
#------------------------------------------------------------------------

def task_1_5():
  #l = 0.5
  #inverse = pseudo_inverse_cdf(cdf, l)
  pass
  

# -----------------------------------------------------------------------
#                             Task 1.6
#------------------------------------------------------------------------

def task_1_6():
  pass

# -----------------------------------------------------------------------
#                             Task 1.7
#------------------------------------------------------------------------

def task_1_7():
  pass

# -----------------------------------------------------------------------
#                             Task 1.9
#------------------------------------------------------------------------

def task_1_9():
  pass

# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_1_3()
#task_1_4()
#task_1_5()
#task_1_6()
#task_1_7()
#task_1_9()