import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float
from skimage.io import imread
from skimage.transform import resize
from skimage import filter
from skimage import exposure


# -----------------------------------------------------------------------
#                             Task 1.3
#------------------------------------------------------------------------

def task_1_3():
  I_1 = imread('images/coins.png')
  I_2 = imread('images/overlapping_euros1.png')

  # Get the otsu value via the 'Otsu Threshold' (where the histogram is to be split/segmented)
  otsu1 = filter.threshold_otsu(I_1)
  otsu2 = filter.threshold_otsu(I_2)

  # Extracting Histograms
  hist1, bins_center1 = exposure.histogram(I_1)
  hist2, bins_center2 = exposure.histogram(I_2)

  # Printing the images, segmenting inside 'imshow'
  plt.figure(figsize=(9, 4))
  plt.subplot(131)
  plt.imshow(I_1, cmap='gray', interpolation='nearest')
  plt.axis('off')
  plt.subplot(132)
  plt.imshow(I_1 < otsu1, cmap='gray', interpolation='nearest')
  plt.axis('off')
  plt.subplot(133)
  plt.plot(bins_center1, hist1, lw=2)
  plt.axvline(otsu1, color='k', ls='--')
  plt.tight_layout()
  plt.show()

  # Printing the images, segmenting inside 'imshow'
  plt.figure(figsize=(9, 4))
  plt.subplot(131)
  plt.imshow(I_2, cmap='gray', interpolation='nearest')
  plt.axis('off')
  plt.subplot(132)
  plt.imshow(I_2 < otsu2, cmap='gray', interpolation='nearest')
  plt.axis('off')
  plt.subplot(133)
  plt.plot(bins_center2, hist2, lw=2)
  plt.axvline(otsu2, color='k', ls='--')
  plt.tight_layout()
  plt.show()


# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

task_1_3()