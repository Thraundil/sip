from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.transform import resize
from pylab import ginput
import numpy as np


# -----------------------------------------------------------------------
#                             Task 1
#------------------------------------------------------------------------

def task1():
    Z = np.random.randint(255, size=(20,20))  # Test data
    plt.imshow(Z, cmap='gray',interpolation='nearest')
    print('Click on 3 points in the image')
    coord = ginput(1)
    print('You clicked: ' + str(coord))
    coord = np.around(coord)
    print (coord)
    Z[coord[0,1],coord[0,0]] = 0

    plt.imshow(Z, cmap='gray',interpolation='nearest')

    plt.show()

# -----------------------------------------------------------------------
#                             Task 2
#------------------------------------------------------------------------

def task2():
    img = imread('week 1/pout.tif')

    list = []
    mask = 1
    for i in range(8):
        bits = np.bitwise_and(img, mask)
        # imshow(bits,cmap='gray')
        mask *= 2
        list.append(bits)
        # show()

    f, axarr = plt.subplots(4, 2)
    axarr[0, 0].imshow(list[0], cmap='gray')
    axarr[0, 0].set_title('First bit')
    axarr[0, 1].imshow(list[1], cmap='gray')
    axarr[0, 1].set_title('Second bit')
    axarr[1, 0].imshow(list[2], cmap='gray')
    axarr[1, 0].set_title('Third bit')
    axarr[1, 1].imshow(list[3], cmap='gray')
    axarr[1, 1].set_title('Fourth bit')
    axarr[2, 0].imshow(list[4], cmap='gray')
    axarr[2, 0].set_title('Fifth bit')
    axarr[2, 1].imshow(list[5], cmap='gray')
    axarr[2, 1].set_title('Sixth bit')
    axarr[3, 0].imshow(list[6], cmap='gray')
    axarr[3, 0].set_title('Seventh bit')
    axarr[3, 1].imshow(list[7], cmap='gray')
    axarr[3, 1].set_title('Eighth bit')

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[3, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    f.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.0, hspace=0.5)

    plt.show()

# -----------------------------------------------------------------------
#                             Task 3
#------------------------------------------------------------------------

def task3():
    rgb_img = imread('images/monster.jpg')
    plt.imshow(rgb_img)
    plt.title('RGB')
    plt.show()

    plt.imshow(rgb2hsv(rgb_img)[:,:,0],cmap='gray')
    plt.title('HUE')
    plt.show()

    plt.imshow(rgb2hsv(rgb_img)[:,:,1],cmap='gray')
    plt.title('SATURATION')
    plt.show()

    plt.imshow(rgb2hsv(rgb_img)[:,:,2],cmap='gray')
    plt.title('INTENSITY')
    plt.show()

# -----------------------------------------------------------------------
#                             Task 4
#------------------------------------------------------------------------

def task4():
    img1 = imread('images/toycars1.png')
    img2 = imread('images/toycars2.png')
    img1 = img1 * 0.5
    img2 =  img2 * 2.0
    img3 = img1 * img2
    img3[img3<255] = 255
    imshow(img3)
    show()


# -----------------------------------------------------------------------
#                             Task 5
#------------------------------------------------------------------------

def task5():
    rgb_img  = imread('images/football.jpg')
    plt.imshow(rgb_img)
    plt.title('Original RGB image')
    plt.show()

    gray_img = (rgb_img[:,:,0] * 0.299) + (rgb_img[:,:,1] * 0.587) + (rgb_img[:,:,2] * 0.114)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Gray image, via perceptually pleasing transformation')
    plt.show()

# -----------------------------------------------------------------------
#                             Task 6
#------------------------------------------------------------------------

def task6():
    rgb_img = imread('images/monster.jpg')
    plt.imshow(rgb_img)
    plt.title('Original size')
    plt.axis('off')
    plt.show()

    # Part 1, reducing
    plt.imshow(resize(rgb_img, (192,252)))
    plt.title('Resized image, (192,252)')
    plt.axis('off')
    plt.show()

    small_img = resize(rgb_img, (50,62))
    plt.imshow(small_img)
    plt.title('Resized image, (50,62)')
    plt.axis('off')
    plt.show()

    plt.imshow(resize(rgb_img, (10,12)))
    plt.title('Resized image, (10,12)')
    plt.axis('off')
    plt.show()

    # Part 2, upsizing
    # Orders:
        # 0: Nearest-neighbor - Best
        # 1: Bi-linear (default) - Good
        # 2: Bi-quadratic - Bad
        # 3: Bi-cubic - Okay
        # 4: Bi-quartic - Bad
        # 5: Bi-quintic - Bad
    for x in range(0,6):
        plt.imshow(resize(small_img, (500,610), order = x))
        plt.title('Resized image, (500,610): order=' + str(x))
        plt.axis('off')
        plt.show()


# -----------------------------------------------------------------------
#                             Task 7
#------------------------------------------------------------------------
def task7():
  railway_img = imread('images/railway.png')
  plt.imshow(railway_img)
  
  # Select the 2 points of the railway sleepers in the foreground
  coord_f = np.around(ginput(2))
  len_f = (coord_f[1] - coord_f[0])[0]
  print('\nRailway sleeper coordinates (foreground): ', coord_f[0], coord_f[1])
  print('\nPixel length of railway sleepers in front is: ', len_f)
  
  # Select the 2 points of the railway sleepers in the background
  coord_b = np.around(ginput(2))
  len_b = (coord_b[1] - coord_b[0])[0]
  print('\nRailway sleeper coordinates (background): ', coord_b[0], coord_b[1])
  print('\nPixel length of railway sleepers in back is: ', len_b)

# -----------------------------------------------------------------------
#                             Task 8
#------------------------------------------------------------------------

def task8():
    # Load in the images
    img1 = imread('images/cameraman.tif')
    img2 = imread('images/rice.png')
    print (img1.shape, img2.shape)

    # Resize the image
    img1 = resize(img1, (200,200), mode='reflect')
    img2 = np.resize(img2,(200,200))
    # multiply the image with 255 to make the range go from
    # 0 - 255 instead of 0 - 1
    img1 *= 255
    # make it integer values by throwing away the decimals
    img1 = np.around(img1)

    # add the images
    added_img = img1 + img2
    # make sure no value is greater than 255
    added_img[added_img > 255] = 255
    # Subtract the images
    minus_img = img1 - img2
    # Make sure no value is less than 0
    minus_img[minus_img < 0] = 0
    # visualize the resulting images
    plt.imshow(added_img)
    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    plt.title('Adding two images')

    show()
    imshow(minus_img)
    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    plt.title('subtracting two images')
    show()

# -----------------------------------------------------------------------
#                             Task 9
#------------------------------------------------------------------------



# -----------------------------------------------------------------------
#                             Task 10
#------------------------------------------------------------------------


# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task1()
#task2()
#task3()
#task4()
#task5()
task6()
#task7()
#task8()
#task9()
#task10()