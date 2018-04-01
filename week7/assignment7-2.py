import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import pi, cos, sin
from skimage.io import imread

# -----------------------------------------------------------------------
#                         General Functions
#------------------------------------------------------------------------
def mr8_filter_bank(I, sigmas, angles):
  
  # Step (1)
  gaussian = cv2.GaussianBlur(I, ksize=(5, 5), sigmaX=sigmas[0])
  
  # Step (2)
  laplacian = cv2.Laplacian(gaussian, cv2.CV_64FC1)    
  
  # Step (3)
  first_order = []
  for sigma in sigmas[1:]:
    values = []
    for theta in angles:
      L = cv2.GaussianBlur(I, ksize=(5, 5), sigmaX=sigma)
      
      # Differentiate L according to x and y
      L_x = np.gradient(L, edge_order=1, axis=0)
      L_y = np.gradient(L, edge_order=1, axis=1)
      
      # Get filter response
      L_theta = cos(theta) * L_x  + sin(theta) * L_y
      values.append(L_theta)
    
    # Get max L_theta
    first_order.append(np.max(values, axis=0))
      
  # Step (4)
  second_order = []
  for sigma in sigmas[1:]:
    values = []
    for theta in angles:
      L = cv2.GaussianBlur(I, ksize=(5, 5), sigmaX=sigma)
      
      # Differentiate L according to xx, xy, and yy
      L_xx = np.gradient(L, edge_order=2, axis=0)
      L_yy = np.gradient(L, edge_order=2, axis=1)
      L_xy = np.gradient(np.gradient(L, edge_order=2, axis=0), edge_order=2, axis=1)
      
      # Get filter response
      L_theta = (cos(theta)**2) * L_xx + 2*cos(theta)*sin(theta)*L_xy + (sin(theta)**2) * L_yy
      values.append(L_theta)
    
    # Get max L_{theta theta}
    second_order.append(np.max(values, axis=0))
  
  
  filters = [gaussian, laplacian, first_order[0], first_order[1], 
             first_order[2], second_order[0], second_order[1], second_order[2]]
  return filters
  

# -----------------------------------------------------------------------
#                             Task 2.1
#------------------------------------------------------------------------
def task_2_1():
  img1 = imread('images/canvas1-a-p001.png'  , asgrey=True)
  img2 = imread('images/cushion1-a-p001.png' , asgrey=True)
  img3 = imread('images/linseeds1-a-p001.png', asgrey=True)
  img4 = imread('images/sand1-a-p001.png'    , asgrey=True)
  img5 = imread('images/seat2-a-p001.png'    , asgrey=True)
  img6 = imread('images/stone1-a-p001.png'   , asgrey=True)
  
  values = []
  images = [img1, img2, img3, img4, img5, img6]
  titles = ['canvas1-a-p001.png', 'cushion1-a-p001.png', 'linseeds1-a-p001.png', 
            'sand1-a-p001.png', 'seat2-a-p001.png', 'stone1-a-p001.png']
  
  # Plot the intensity histograms of the six texture images
  for i in range(6):
    n, bins, _ = plt.hist(images[i].ravel(),256,[0,256], density=True)
    plt.xlabel('Pixel range')
    plt.ylabel('Sum')
    plt.title('Intensity histogram of ' + titles[i])
    plt.show()
    
    # Save the values of the histogram bins
    values.append(n)
    
  # Compute chi^2-distance between each pair of the six histograms
  for i in range(6):
    for j in range(6):
      
      # Get indices where both bins in the two histograms are not 0
      h_index = np.nonzero(values[i])[0]
      g_index = np.nonzero(values[j])[0]
      indices = []
      
      for n in range(255):
        if(n in h_index.tolist() and n in g_index.tolist()):
          indices.append(n)
          
      h = np.take(values[i], indices)
      g = np.take(values[j], indices)

      # Compute chi^2-distance
      chi_squared_dist = np.sum( np.power(h-g, 2) / (h+g) )
      print("Pair ",i+1, j+1, ": ", chi_squared_dist)
      

# -----------------------------------------------------------------------
#                             Task 2.2 i.
#------------------------------------------------------------------------
def task_2_2_i():
  I = imread('images/linseeds1-a-p001.png', asgrey=True)
  plt.imshow(I, cmap='gray')
  plt.title('Original image')
  plt.show()
  
  # Get the 8 filter responses
  sigmas = [10, 1, 2, 4]
  angles = [0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6]
  filter_responses = mr8_filter_bank(I, sigmas, angles)
  
  # Illustrate the 8 filter responses
  titles = ['Gaussian filter', 'Laplacian', 
            '1st order Gaussian derivative for sigma=' + str(sigmas[1]),
            '1st order Gaussian derivative for sigma=' + str(sigmas[2]),
            '1st order Gaussian derivative for sigma=' + str(sigmas[3]),
            '2nd order Gaussian derivative for sigma=' + str(sigmas[1]),
            '2nd order Gaussian derivative for sigma=' + str(sigmas[2]),
            '2nd order Gaussian derivative for sigma=' + str(sigmas[3])]
  
  for i in range(8):
    plt.imshow(filter_responses[i], cmap='gray')
    plt.title(titles[i])
    plt.show()


# -----------------------------------------------------------------------
#                             Task 2.2 ii.
#------------------------------------------------------------------------
def task_2_2_ii():
  images = []
  images.append(imread('images/canvas1-a-p001.png'  , asgrey=True))
  images.append(imread('images/cushion1-a-p001.png' , asgrey=True))
  images.append(imread('images/linseeds1-a-p001.png', asgrey=True))
  images.append(imread('images/sand1-a-p001.png'    , asgrey=True))
  images.append(imread('images/seat2-a-p001.png'    , asgrey=True))
  images.append(imread('images/stone1-a-p001.png'   , asgrey=True))
  
  # Apply the filter bank for each of the six images
  filters = []
  sigmas = [10, 1, 2, 4]
  angles = [0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6]
  
  for i in range(len(images)):
    filters.append(mr8_filter_bank(images[i], sigmas, angles))
    
  # Get training data
  training_data = np.empty([0,8])
  for i in range(len(images)):
    # Get the 8 response filters and flatten them
    filterr = filters[i]
    f_0 = filterr[0].flatten()
    f_1 = filterr[1].flatten()
    f_2 = filterr[2].flatten()
    f_3 = filterr[3].flatten()
    f_4 = filterr[4].flatten()
    f_5 = filterr[5].flatten()
    f_6 = filterr[6].flatten()
    f_7 = filterr[7].flatten()
    
    # Form 8 dimensional vector of the filter responses at each pixel
    data = list(zip(f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7))    
    training_data = np.vstack((training_data, data))

    
  # Apply K-means algorithm with K = 60
  kmeans = KMeans(n_clusters=60).fit(training_data)
  print(kmeans.cluster_centers_)
  


# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_2_1()
#task_2_2_i()
#task_2_2_ii()