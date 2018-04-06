import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import hough_line, hough_line_peaks



# -----------------------------------------------------------------------
#                             Task 2.1
#------------------------------------------------------------------------

def task_2_1(img):
  theta_res, dist_res = 1, 1
  rows, cols = img.shape[0], img.shape[1]
  
  theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1.0)
  theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
 
  D = np.sqrt((rows - 1)**2 + (cols - 1)**2)
  q = np.ceil(D/dist_res)
  n = 2*q + 1
  dist = np.linspace(-q*dist_res, q*dist_res, n)
  
  # Initialize values in accumulator to zero
  h = np.zeros((len(dist), len(theta)))
  
  # Iterate through each pixel pair (x,y)
  for x in range(rows):
    for y in range(cols):
      if img[x, y]:
        for t in range(len(theta)):
          distVal = y*np.cos(theta[t]*np.pi/180.0) + \
              x*np.sin(theta[t]*np.pi/180)
          index = np.nonzero(np.abs(dist-distVal) == np.min(np.abs(dist-distVal)))[0]
          h[index[0], t] += 1
  
  return h, theta, dist  
    


# -----------------------------------------------------------------------
#                             Task 2.2
#------------------------------------------------------------------------

def task_2_2():
  img = imread('images/cross.png', asgrey=True)
  plt.imshow(img, cmap='gray')
  plt.title('Original image')
  plt.show()

  h, theta, dist = task_2_1(img)
  plt.imshow(h, cmap='gray')
  plt.title('Implementation')
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
  plt.title('hough_line: Hough transform of image')
  plt.show()
  
  plt.imshow(img, cmap='gray')
  for _, angle, dist in zip(*hough_line_peaks(hspace, angle, dist)):
    y1 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y2 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
    plt.plot((0, img.shape[1]), (y1, y2), '-r')
    
  plt.title('hough_line: Detected lines')
  plt.show()


# -----------------------------------------------------------------------
#                             Task 2.3
#------------------------------------------------------------------------

def task_2_3():
  pass


# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_2_1()
task_2_2()
#task_2_3()