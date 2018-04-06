import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import hough_line, hough_line_peaks, hough_circle
from skimage.filter import canny



# -----------------------------------------------------------------------
#                             Task 2.1
#------------------------------------------------------------------------

def task_2_1(img):

  # Define distance range (diagonal length of the image)
  rows, cols = img.shape[0], img.shape[1]  
  diag = np.ceil(np.sqrt((rows - 1)**2 + (cols - 1)**2))
  dist_range = np.linspace(-diag, diag, 2*diag+1)
    
  # Define theta range from origin to the line (-90 to 90 degrees)
  theta_range = np.linspace(-90, 90, 181)
  
  # Initialize values in accumulator to zero
  h = np.zeros((len(dist_range), len(theta_range)))
  
  # Iterate through each pixel pair (x,y) that is not 0
  xs, ys = np.nonzero(img)
  for i in range(len(xs)):
    x = xs[i]
    y = ys[i]      
        
    # Calculate rho with a point at each angle from -90 to 90 degrees
    for theta in range(len(theta_range)):
      val = x * np.cos(theta_range[theta] * math.pi/180.0) + y * np.sin(theta_range[theta] * math.pi/180)
      rho = np.nonzero(np.abs(dist_range - val) == np.min(np.abs(dist_range - val)))[0][0]
      h[rho, theta] += 1
  
  return h, theta_range, dist_range
    


# -----------------------------------------------------------------------
#                             Task 2.2
#------------------------------------------------------------------------

def task_2_2():
  img = imread('images/cross.png', asgrey=True)
  plt.imshow(img, cmap='gray')
  plt.title('Original image')
  plt.show()

  # Test our implementation on the image
  h, theta, dist = task_2_1(img)
  
  # Plot hough transform of image and detected lines
  plt.imshow(h, cmap='gray')
  plt.title('Implementation: Hough transform of image')
  plt.show()
  
  plt.imshow(img, cmap='gray')
  for _, angle, dist in zip(*hough_line_peaks(h, theta, dist)):
    y1 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y2 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
    plt.plot((0, img.shape[1]), (y1, y2), '-r')
  plt.title('Implementation: Detected lines')
  plt.show()
      

  # Use hough_line method from scikit-image and plot output
  hspace, angle, dist = hough_line(img)

  plt.imshow(hspace, cmap='gray')
  plt.title('hough_line(): Hough transform of image')
  plt.show()
  
  plt.imshow(img, cmap='gray')
  for _, angle, dist in zip(*hough_line_peaks(hspace, angle, dist)):
    y1 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y2 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
    plt.plot((0, img.shape[1]), (y1, y2), '-r')  
  plt.title('hough_line(): Detected lines')
  plt.show()


# -----------------------------------------------------------------------
#                             Task 2.3
#------------------------------------------------------------------------

def task_2_3():
  I = imread('images/coins.png')
  edges_I = canny(I)

  radi = np.arange(3, 30)
  hough1 = hough_circle(I, radi)
  hough2 = hough_circle(edges_I, radi)

  print hough1.shape

  fig, ax = plt.subplots(1,2)
  ax[0].imshow(I, cmap='gray')
  ax[1].imshow(hough2[0], cmap='gray')
  ax[0].set_title('Original image')
  ax[0].axis('off')
  ax[1].set_title('Original Image segmented via Hough Circle')
  ax[1].axis('off')
  plt.show()


# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_2_1()
task_2_2()
#task_2_3()