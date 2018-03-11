import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops

# -----------------------------------------------------------------------
#                         General Functions
#------------------------------------------------------------------------
def get_form_factor(area, perimeter):
  return (4 * math.pi * area) / perimeter**2

def get_roundness(area, max_diameter):
  return (4 * area) / (math.pi * max_diameter**2)

def get_aspect_ratio(max_diameter, min_diameter):
  return max_diameter / min_diameter

def get_solidity(area, convex_area):
  return area / convex_area

def get_extent(p):
  return p.extent

def get_compactness(area, max_diameter):
  return math.sqrt((4 * area) / math.pi) / max_diameter

def get_convexity(convex_perimeter, perimeter):
  return convex_perimeter / perimeter

# -----------------------------------------------------------------------
#                             Task 1.1
#------------------------------------------------------------------------
def task_1_1():
  I = imread('images/basic_shapes.png')
  plt.imshow(I, cmap='gray')
  plt.show()
  labels, num = label(I, background=0, connectivity=2, return_num=True)
  
  # Get area based on the label of the 4 objects
  area = {}
  for i in range(1,num + 1):
    area[i] = np.size(np.where(labels==i)[1])
  
  # Measure properties of labeled image regions
  properties = regionprops(labels, cache=False)
  
  # Get properties
  area2        = [p.area for p in properties]
  perimeter    = [p.perimeter for p in properties]
  convex_area  = [p.convex_area for p in properties]
  convex_img   = [p.convex_image for p in properties]
  min_diameter = []
  max_diameter = []
  for p in properties:
    min_row, min_col, max_row, max_col = p.bbox
    min_diameter.append(min(max_row - min_row, max_col - min_col))
    max_diameter.append(max(max_row - min_row, max_col - min_col))
  

  # Compute the 7 shape descriptors from Table A
  col     = ['blue', 'red', 'green', 'yellow']
  objects = ['Big circle', 'Small circle', 'Rectangle', 'Star']
  for i in range(num):
    form_factor      = get_form_factor(area2[i], perimeter[i])    
    roundness        = get_roundness(area2[i], max_diameter[i])
    aspect_ratio     = get_aspect_ratio(max_diameter[i], min_diameter[i])
    solidity         = get_solidity(area2[i], convex_area[i])
    extent           = get_extent(properties[i])
    compactness      = get_compactness(area2[i], max_diameter[i])
    
    properties2      = regionprops(convex_img[i]*1, cache=False)
    convex_perimeter = [p.perimeter for p in properties2][0]
    convexity        = get_convexity(convex_perimeter, perimeter[i])
    
    # Print results
    print(objects[i], '- Form Factor: ', np.around(form_factor, 3))
    print(objects[i], '- Roundness: ', np.around(roundness, 3))
    print(objects[i], '- Aspect Ratio: ', np.around(aspect_ratio, 3))
    print(objects[i], '- Solidity: ', np.around(solidity, 3))
    print(objects[i], '- Extent: ', np.around(extent, 3))
    print(objects[i], '- Compactness: ', np.around(compactness, 3))
    print(objects[i], '- Convexity: ', np.around(convexity, 3))
    print('\n')

    # Make scatter plot
    results = [form_factor, roundness, aspect_ratio, 
               solidity, extent, compactness, convexity]
    plt.scatter(range(1,8), results, c=col[i], label=objects[i])
  
  # Define scatter plot properties
  plt.title('Combinations of the 7 descriptors for the 4 shapes')
  plt.xlabel('Descriptor #')
  plt.ylabel('Value')
  plt.legend(loc='upper right')
  plt.show()

# -----------------------------------------------------------------------
#                             Task 1.2
#------------------------------------------------------------------------
def task_1_2():
  # Load image as grayscale and blur to remove noise
  I = imread('images/pillsetc.png', as_grey=True)*255
  I_gausBlur = cv2.GaussianBlur(I, (7, 7), 5)

  # Plot the intensity histogram
  plt.hist(I_gausBlur.ravel(),256,[0,256])
  plt.xlabel('Pixel range')
  plt.ylabel('Sum')
  plt.title('Intensity histogram')
  plt.show() 
  
  # Use threshold value of 80 to create a binary image of the objects
  ret, bin_img = cv2.threshold(I_gausBlur, 80, 255,cv2.THRESH_BINARY)
  plt.imshow(bin_img, cmap='gray')
  plt.title('Binary image of the objects with threshold=80')
  plt.show()
  
  # Label each shape and get their areas
  labels, num = label(bin_img, background=0, connectivity=2, return_num=True)
  area = {}
  for i in range(1,num + 1):
    area[i] = np.size(np.where(labels==i)[1])
  
  # Get properties 
  properties = regionprops(labels, cache=False)
  
  area2        = [p.area for p in properties]
  perimeter    = [p.perimeter for p in properties]
  convex_area  = [p.convex_area for p in properties]
  convex_img   = [p.convex_image for p in properties]
  min_diameter = []
  max_diameter = []
  for p in properties:
    min_row, min_col, max_row, max_col = p.bbox
    min_diameter.append(min(max_row - min_row, max_col - min_col))
    max_diameter.append(max(max_row - min_row, max_col - min_col))
    
  # Compute the 7 shape descriptors from Table A for each shape
  col     = ['blue', 'red', 'green', 'yellow', 'black', 'grey', 'purple']
  objects = ['Round thing (right)', 'Round thing (left)', 'Green thing', 
             'Rectangle', 'Pill (left)', 'Pill (right)']
  for i in range(num):
    form_factor      = get_form_factor(area2[i], perimeter[i])    
    roundness        = get_roundness(area2[i], max_diameter[i])
    aspect_ratio     = get_aspect_ratio(max_diameter[i], min_diameter[i])
    solidity         = get_solidity(area2[i], convex_area[i])
    extent           = get_extent(properties[i])
    compactness      = get_compactness(area2[i], max_diameter[i])
    
    properties2      = regionprops(convex_img[i]*1, cache=False)
    convex_perimeter = [p.perimeter for p in properties2][0]
    convexity        = get_convexity(convex_perimeter, perimeter[i])
    
    # Print results
    print(objects[i], '- Form Factor: ', np.around(form_factor, 3))
    print(objects[i], '- Roundness: ', np.around(roundness, 3))
    print(objects[i], '- Aspect Ratio: ', np.around(aspect_ratio, 3))
    print(objects[i], '- Solidity: ', np.around(solidity, 3))
    print(objects[i], '- Extent: ', np.around(extent, 3))
    print(objects[i], '- Compactness: ', np.around(compactness, 3))
    print(objects[i], '- Convexity: ', np.around(convexity, 3))
    print('\n')

    # Make scatter plot
    results = [form_factor, roundness, aspect_ratio, 
               solidity, extent, compactness, convexity]
    plt.scatter(range(1,8), results, c=col[i], label=objects[i])
  
  # Define scatter plot properties
  plt.title('Combinations of the 7 descriptors for the 6 shapes')
  plt.xlabel('Descriptor #')
  plt.ylabel('Value')
  plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
  plt.show()
    


# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_1_1()
task_1_2()