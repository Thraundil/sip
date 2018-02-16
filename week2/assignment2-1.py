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
  subset  = np.take(range(256), np.where(cdf >= l))
  return np.min(subset) 

  
def histogram_matching(original, target):
  # Compute histogram and then the cdf for the two images
  hist_1, _ = np.histogram(original.ravel(),256,[0,256])
  hist_2, _ = np.histogram(target.ravel(),256,[0,256])
  cdf_1 = cumulative_histogram(hist_1, original.shape[0], original.shape[1])
  cdf_2 = cumulative_histogram(hist_2, target.shape[0], target.shape[1])
  
  # Perform histogram matching f = C_2^{-1} * (C_1( img_1 ))
  l_values = np.linspace(0, 1, num=256)
  cdf_2_inverse = []
  for l in l_values:
    cdf_2_inverse.append(pseudo_inverse_cdf(cdf_2, l))
  
  hist_matching = cdf_2_inverse * cdf_1
    
  # Plot cumulative histograms for original, target, and histogram matching
  plt.plot(cdf_1, color='r', label='Original image')
  plt.plot(cdf_2, color='g', label='Target image')
  plt.plot(hist_matching, l_values, '-', label='Histogram matching')
  
  # Define plot properties
  plt.title('Cumulative histograms')
  plt.legend(bbox_to_anchor=(1.2,0.1), loc='lower right', ncol=1)
  plt.xlabel('Pixel range')
  plt.ylabel('Cumulative sum')
  plt.xlim([0,256])
  plt.show()
  

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
  #see function pseudo_inverse_cdf()
  pass
  

# -----------------------------------------------------------------------
#                             Task 1.6
#------------------------------------------------------------------------

def task_1_6():
  # Define original and target image and perform histogram matching
  img_1 = imread('images/pout.tif')
  img_2 = imread('images/cameraman.tif')
  histogram_matching(img_1, img_2)

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