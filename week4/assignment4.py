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



# -----------------------------------------------------------------------
#                         General Functions
#------------------------------------------------------------------------

  

# -----------------------------------------------------------------------
#                          FEATURE DETECTORS
#                                 1
#------------------------------------------------------------------------

def feat_1():
  I = imread('images/hand.tiff')
  scimmy_I     = filter.canny(I, low_threshold=None, high_threshold=None)
  scimmyLow_I  = filter.canny(I, low_threshold=100.0, high_threshold=None)
  scimmyHigh_I = filter.canny(I, low_threshold=None, high_threshold=200.0)

  sigmaI       = filter.canny(I, sigma=2.0, low_threshold=None, high_threshold=None)
  sigmaLow_I   = filter.canny(I, sigma=2.0, low_threshold=100.0, high_threshold=None)
  sigmaHigh_I  = filter.canny(I, sigma=2.0, low_threshold=None, high_threshold=200.0)
  allI         = filter.canny(I, sigma=2.0, low_threshold=100.0, high_threshold=200.0)

  fig, ax = plt.subplots(2,2)
  ax[0][0].imshow(I, cmap='gray')
  ax[0][0].set_title('Original image')
  ax[0][0].axis('off')
  ax[0][1].imshow(scimmy_I, cmap='gray')
  ax[0][1].set_title('canny-transformed (no thresholds)')
  ax[0][1].axis('off')
  ax[1][0].imshow(scimmyLow_I, cmap='gray')
  ax[1][0].set_title('canny-transformed (low_threshold=100.0)')
  ax[1][0].axis('off')
  ax[1][1].imshow(scimmyHigh_I, cmap='gray')
  ax[1][1].set_title('canny-transformed (high_threshold=200.0)')
  ax[1][1].axis('off')
  plt.show()

  fig, ax = plt.subplots(2,2)
  ax[0][0].imshow(sigmaI, cmap='gray')
  ax[0][0].set_title('Original image w. Sigma 2.0')
  ax[0][0].axis('off')
  ax[0][1].imshow(sigmaLow_I, cmap='gray')
  ax[0][1].set_title('canny-transformed Sigma=2.0, low threshold')
  ax[0][1].axis('off')
  ax[1][0].imshow(sigmaHigh_I, cmap='gray')
  ax[1][0].set_title('canny-transformed Sigma=2.0, High threshold')
  ax[1][0].axis('off')
  ax[1][1].imshow(allI, cmap='gray')
  ax[1][1].set_title('canny-transformed Sigma=2.0, both high and low Thresholds')
  ax[1][1].axis('off')
  plt.show()


# NOTE: ifft( fft(I) * fft(k,I.shape) ))
  
# -----------------------------------------------------------------------
#                          FEATURE DETECTORS
#                                 2
#------------------------------------------------------------------------

def feat_2():
  I = imread('images/modelhouses.png')
  harris_I      = feature.corner_harris(I)
  harrisSigma_I = feature.corner_harris(I, sigma=5.0)
  harrisK_I     = feature.corner_harris(I, method='k', k=0.90)

  harrisSigma2_I = feature.corner_harris(I, sigma=20.0)
  harrisSigmaK_I = feature.corner_harris(I, method='k', sigma=5.0, k=0.90)
  harrisEPS_I    = feature.corner_harris(I, eps=2, method='eps')
  harrisAll_I    = feature.corner_harris(I, eps=1e-48, method='eps', sigma=4.2)

  fig, ax = plt.subplots(2,2)
  ax[0][0].imshow(I, cmap='gray')
  ax[0][0].set_title('Original image "modelhouses.png"')
  ax[0][0].axis('off')
  ax[0][1].imshow(harris_I, cmap='gray')
  ax[0][1].set_title('corner_harris')
  ax[0][1].axis('off')
  ax[1][0].imshow(harrisSigma_I, cmap='gray')
  ax[1][0].set_title('corner_harris Sigma = 5.0')
  ax[1][0].axis('off')
  ax[1][1].imshow(harrisK_I, cmap='gray')
  ax[1][1].set_title('corner_harris k = 0.90')
  ax[1][1].axis('off')
  plt.show()

  fig, ax = plt.subplots(2,2)
  ax[0][0].imshow(harrisSigma2_I, cmap='gray')
  ax[0][0].set_title('corner_harris Sigma = 20.0')
  ax[0][0].axis('off')
  ax[0][1].imshow(harrisSigmaK_I, cmap='gray')
  ax[0][1].set_title('corner_harris K=0.90, Sigma = 5')
  ax[0][1].axis('off')
  ax[1][0].imshow(harrisEPS_I, cmap='gray')
  ax[1][0].set_title('corner_harris eps=2')
  ax[1][0].axis('off')
  ax[1][1].imshow(harrisAll_I, cmap='gray')
  ax[1][1].set_title('corner_harris eps=1e-48, sigma=4.2')
  ax[1][1].axis('off')
  plt.show()


# -----------------------------------------------------------------------
#                          FEATURE DETECTORS
#                                 3
#------------------------------------------------------------------------

def feat_3():
  I = imread('images/modelhouses.png')
  harris_I = feature.corner_harris(I)
  harris2_I = feature.corner_harris(I)
  corner   = feature.corner_peaks(harris_I)

  plt.imshow(I,cmap='gray')
  plt.title('Original "modelhouses.png"')
  plt.axis('off')
  plt.show()

  for x,y in corner:
    harris2_I[x][y] = 15

  fig, ax = plt.subplots(1,2)
  ax[0].imshow(harris_I, cmap='gray')
  ax[0].set_title('corner_harris transformed')
  ax[0].axis('off')
  ax[1].imshow(harris2_I, cmap='gray')
  ax[1].set_title('Highlighted local maxima in feature map (ie. "the white dots")')
  ax[1].axis('off')
  plt.show()  


# -----------------------------------------------------------------------
#                        SCALE-SPACE OPERATORS
#                                 1
#------------------------------------------------------------------------

def scale_2(sigma):
  # Hint for another task:
  # ifft (fft(I) * fft(k, I.shape))
  n = 20
  m = 20
  img = imread('images/modelhouses.png')
  out = np.zeros((n, m))
  for i in range(n):
    for j in range(m):
      out[i][j] = float(
        1 / (2 * math.pi * sigma ** 2) * math.exp((-((i - n / 2) ** 2 + (j - m / 2) ** 2) / (2.0 * sigma ** 2))))
  imshow(out, cmap='gray')
  show()

  img = filters.convolve(img, out)

  new_img = imread('images/modelhouses.png')
  #s = 2
  #w = 5
  #t = (((w - 1) / 2) - 0.5) / s
  new_img = filters.gaussian_filter(new_img, sigma)

  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(img, cmap='gray')
  ax[0].set_title('Own implementation, sigma = 5.0')
  ax[0].axis('off')
  ax[1].imshow(new_img, cmap='gray')
  ax[1].set_title('Scale function, sigma = 5.0')
  ax[1].axis('off')
  plt.show()



# -----------------------------------------------------------------------
#                        SCALE-SPACE OPERATORS
#                                 2
#------------------------------------------------------------------------




# -----------------------------------------------------------------------
#                        SCALE-SPACE OPERATORS
#                                 3
#------------------------------------------------------------------------

def scale_3():
  pass

# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#feat_1()
#feat_2()
#feat_3()
#scale_1()
scale_2(5.0)
#scale_3()