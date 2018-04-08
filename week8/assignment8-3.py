import h5py
import numpy 
import os, random
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread


## Configure the network

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 135
# number of epochs to train
nb_epoch = 40

# number of convolutional filters to use
nb_filters = 20
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, dim_ordering='th', border_mode='valid',input_shape = (1,29,29)))
convout1=Activation('relu')
model.add(convout1)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering="th"))
#model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, dim_ordering='th', border_mode='valid'))
convout2=Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering="th"))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(4000,batch_input_shape=(None, 1,29,29)))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(135))
model.add(Activation('softmax'))
    
adam = adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

# ## Train model - uncoment to perform the training yourself
#
# train = numpy.load('train.npz')
# x_train = train['x_train']
# y_train = train['y_train']
#
# for m in range(0,nb_epoch):
#     for jj in range(0,x_train.shape[0]-256*20,256*20):
#         print('Epoch number is', m)
        
#         xx = x_train[jj:jj+256*20,:]
#         yy = y_train[jj:jj+256*20,:]
        
#         earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
#         model.fit(xx,yy,epochs=1,batch_size=1, callbacks=[earlyStopping], validation_split=0.20)
#     print("saving weights")
#     model.save_weights('data/keras-epoch-' + str(m) + '.h5') 
   
# model.save_weights('keras.h5')

## Load the pretrained network
model.load_weights('keras.h5')

# Opgave 3.2
def task32():
    test = numpy.load('test.npz')
    x_test = test['x_test']
    y_test = test['y_test']
    score = model.evaluate(x_test, y_test)
    print (score)
    print (model.metrics_names)

# Opgave 3.3
def task33():
    img = imread('test_images/image/1003_3_image.png')
    final_patch = numpy.zeros((65536,1,29,29))
    new_img = numpy.pad(img,pad_width=14,mode='constant')
    k = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = new_img[i:i+29,j:j+29]
            patch =  patch / 255
            patch = numpy.expand_dims(patch, axis=0)
            final_patch[k] = patch
            k += 1
    my_pred = model.predict(final_patch)
    my_pred *= 100 # To get the percentage
    return my_pred

def task34():
    my_pred = task33()
    img = imread('test_images/seg/1003_3_seg.png')


#task32()
#task33()
#task34()

