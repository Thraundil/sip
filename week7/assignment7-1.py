import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy import fftpack
from scipy import signal

# -----------------------------------------------------------------------
#                         General Functions
#------------------------------------------------------------------------



# -----------------------------------------------------------------------
#                             Task 1.1
#------------------------------------------------------------------------
#def task_1_1(Im,Kernel,Noise):
# LSI s.16 i slide om Deconvolution
def task_1_1():
  I = imread('images/trui.png')
  Ishape = I.shape
  K = np.zeros((3,3))
  K[0:1] = 1
  N = np.random.normal(0,42,(Ishape))
  G = np.zeros((Ishape))
  convolution = (signal.convolve(K,I))
  for x in xrange(Ishape[0]):
    for y in xrange(Ishape[1]):
      G[x,y] = (convolution[x,y] + N[x,y])

#  fig, ax = plt.subplots(1,2)
#  ax[0].imshow(I, cmap='gray')
#  ax[1].imshow(G, cmap='gray')
#  ax[0].set_title('Original image')
#  ax[0].axis('off')
#  ax[1].set_title('LSI(Original Image)')
#  ax[1].axis('off')
#  plt.show()
  return G,K



# -----------------------------------------------------------------------
#                             Task 1.2
#------------------------------------------------------------------------
def task_1_2():
  g,h = task_1_1()
  Kernel = np.zeros(g.shape)
  Kernel[0:3,0:3] = h

  # The Fourier transform
  H = (fftpack.fft2(Kernel))
  F = (fftpack.fft2(g))

  # Multiply and Divide
  G = np.multiply(F,H)
  Fhat = np.divide(G,H)
  Fhat = np.absolute(fftpack.ifft2(Fhat))

  fig, ax = plt.subplots(1,2)
  ax[0].imshow(g, cmap='gray')
  ax[1].imshow(Fhat, cmap='gray')
  ax[0].set_title('G')
  ax[0].axis('off')
  ax[1].set_title('Out')
  ax[1].axis('off')
  plt.show()

# -----------------------------------------------------------------------
#                             Task 1.3
#------------------------------------------------------------------------
def task_1_3():
  pass



# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_1_1()
task_1_2()
#task_1_3()