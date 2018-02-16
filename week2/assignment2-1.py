import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
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

  
def histogram_matching(original_cdf, target_cdf):  
  # Perform histogram matching f = C_2^{-1} * (C_1( img_1 ))
  l_values = np.linspace(0, 1, num=256)
  cdf_target_inverse = []
  for l in l_values:
    cdf_target_inverse.append(pseudo_inverse_cdf(target_cdf, l))
  
  hist_matching = cdf_target_inverse * original_cdf
  return hist_matching  

  

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
  # Define original and target image
  original = imread('images/pout.tif')
  target = imread('images/cameraman.tif')
  
  # Compute histogram and then the cdf for the two images
  hist_1, _ = np.histogram(original.ravel(),256,[0,256])
  hist_2, _ = np.histogram(target.ravel(),256,[0,256])
  original_cdf = cumulative_histogram(hist_1, original.shape[0], original.shape[1])
  target_cdf = cumulative_histogram(hist_2, target.shape[0], target.shape[1])
  
  # Perform histogram matching
  l_values = np.linspace(0, 1, num=256)
  hist_matching = histogram_matching(original_cdf, target_cdf)
  
  # Plot cumulative histograms for original, target, and histogram matching
  plt.plot(original_cdf, color='r', label='Original image')
  plt.plot(target_cdf, color='g', label='Target image')
  plt.plot(hist_matching, l_values, '-', label='Histogram matching')
  
  # Define plot properties
  plt.title('Cumulative histograms')
  plt.legend(bbox_to_anchor=(1.2,0.1), loc='lower right', ncol=1)
  plt.xlabel('Pixel range')
  plt.ylabel('Cumulative sum')
  plt.xlim([0,256])
  plt.show()
 

# -----------------------------------------------------------------------
#                             Task 1.7
#------------------------------------------------------------------------

def task_1_7():
  # Define original image and compute histogram and cdf
  original     = imread('images/pout.tif')
  hist, _      = np.histogram(original.ravel(),256,[0,256])
  original_cdf = cumulative_histogram(hist, original.shape[0], original.shape[1])
  
  # Define target cumulative histogram to have a uniform distribution
  uniform_cdf = np.linspace(0, 1, num=256)
  hist_matching = histogram_matching(original_cdf, uniform_cdf)
  
  # Get cumulative histogram of image returned from equalize_hist()
  img_eq    = exposure.equalize_hist(original)
  cdf_eq, _ = exposure.cumulative_distribution(img_eq, 256)
  
  # Compare the two methods by plotting  
  plt.plot(hist_matching, uniform_cdf, label='Our histogram matching')
  plt.plot(cdf_eq, '-',label='equalize_hist()')
  
  plt.title('Cumulative histograms')
  plt.legend(bbox_to_anchor=(1.2,0.1), loc='lower right', ncol=1)
  plt.xlabel('Pixel range')
  plt.ylabel('Cumulative sum')
  plt.xlim([0,256])
  plt.show()

  

# -----------------------------------------------------------------------
#                             Task 1.9
#------------------------------------------------------------------------

def task_1_9():
  # Imports the images, creates histogram, then cumulative histograms.
  I1 = imread('images/movie_flicker1.tif')
  I2 = imread('images/movie_flicker2.tif')

  hist1, bins1  = np.histogram(I1.ravel(),256,[0,256])
  hist2, bins2  = np.histogram(I2.ravel(),256,[0,256])
  C1 = np.cumsum(hist1)
  C2 = np.cumsum(hist2)

  # Calculates the midway specification
  midway = 0.5 * ((C1**(-1.0)) + (C2**(-1.0)))

  # Prints the original 2 images
  plt.imshow(im, cmap='gray')
  plt.show()
  plt.imshow(I2, cmap='gray')
  plt.show()

  # Computes the floating-point image C(I), such that the Intensity at each (x,y) is C(I(x,y))
  newI1  = I1
  newI2  = I2  
  (A,B,C) = I1.shape
  for x in range(0,A):
    for y in range(0,B):
      newI1[x][y] = midway[(I1[x][y])]
      newI2[x][y] = midway[(I2[x][y])]

  plt.imshow(newI1, cmap='gray')
  plt.show()

  plt.imshow(newI2, cmap='gray')
  plt.show()




# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_1_3()
#task_1_4()
#task_1_5()
#task_1_6()
#task_1_7()
#task_1_9()