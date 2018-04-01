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
  K = np.zeros((5,5))
  K[0:1] = 1
  N = np.random.normal(0,50,(Ishape))
  G = np.zeros((Ishape))
  convolution = (signal.convolve(K,I))
  for x in range(Ishape[0]):
    for y in range(Ishape[1]):
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
  K = 5
  g,h = task_1_1()
  Kernel = np.zeros(g.shape)
  Kernel[0:5,0:5] = h

  # The Fourier transform
  H = (fftpack.fft2(Kernel))
  F = (fftpack.fft2(g))

  # Multiply and Divide
  G = np.multiply(F,H)

  # Math....
  # Creating different images with different values of K
  fHat1 = np.multiply(np.divide(1,H) * np.divide(np.multiply(np.conj(H),H),(np.multiply(np.conj(H),H) + 2)), G)
  fHat1 = np.abs(fftpack.ifft2(fHat1))

  fHat2 = np.multiply(np.divide(1,H) * np.divide(np.multiply(np.conj(H),H),(np.multiply(np.conj(H),H) + 4)), G)
  fHat2 = np.abs(fftpack.ifft2(fHat2))

  fHat3 = np.multiply(np.divide(1,H) * np.divide(np.multiply(np.conj(H),H),(np.multiply(np.conj(H),H) + 6)), G)
  fHat3 = np.abs(fftpack.ifft2(fHat3))

  fHat4 = np.multiply(np.divide(1,H) * np.divide(np.multiply(np.conj(H),H),(np.multiply(np.conj(H),H) + 8)), G)
  fHat4 = np.abs(fftpack.ifft2(fHat4))

  fHat5 = np.multiply(np.divide(1,H) * np.divide(np.multiply(np.conj(H),H),(np.multiply(np.conj(H),H) + 10)), G)
  fHat5 = np.abs(fftpack.ifft2(fHat5))

  fig, ax = plt.subplots(2, 3)
  ax[0][0].imshow(g, cmap='gray')
  ax[0][0].set_title('Original image G')
  ax[0][0].axis('off')

  ax[0][1].imshow(fHat1, cmap='gray')
  ax[0][1].set_title('K = 2')
  ax[0][1].axis('off')

  ax[0][2].imshow(fHat2,cmap='gray')
  ax[0][2].set_title('K = 4')
  ax[0][2].axis('off')

  ax[1][0].imshow(fHat3, cmap='gray')
  ax[1][0].set_title('K = 6')
  ax[1][0].axis('off')

  ax[1][1].imshow(fHat4, cmap='gray')
  ax[1][1].set_title('K = 8')
  ax[1][1].axis('off')

  ax[1][2].imshow(fHat5,cmap='gray')
  ax[1][2].set_title('K = 10')
  ax[1][2].axis('off')

  plt.show()



# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_1_1()
#task_1_2()
task_1_3()
