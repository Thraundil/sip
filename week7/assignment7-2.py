import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# -----------------------------------------------------------------------
#                         General Functions
#------------------------------------------------------------------------



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
#                             Task 2.2
#------------------------------------------------------------------------
def task_2_2():
  pass




# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_2_1()
#task_2_2()