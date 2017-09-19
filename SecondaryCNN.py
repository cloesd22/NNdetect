# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:02:17 2017

@author: manoj
"""

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import keras.model import load_model

classifier = Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

test_datagen=ImageDataGenerator(rescale=1./255)

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True
                                 )

trainset= train_datagen.flow_from_directory('datasets/train',target_size=(64,64),batch_size=12,class_mode='binary')

testset=test_datagen.flow_from_directory('datasets/test',target_size=(64,64),batch_size=12,class_mode='binary')

classifier.fit_generator(trainset,steps_per_epoch=32,epochs=25,verbose=1,callbacks=None,validation_data=testset,validation_steps=5)

from keras.preprocessing import image
import numpy as np

#Set URL for targetIMG
targetimg="datasets/Singletest/54.jpg"
#SET object for imgobj use Loadiamge to turn picture into an img object
imgobj = image.load_img(targetimg,target_size=(64,64))
#Set that image object to an array
testimg = image.img_to_array(imgobj)
#Add extra dimension to that array by extending by axis=1
testimg = np.expand_dims(testimg,axis=0)
#use predict method:
prediction = classifier.predict(testimg)

classifier.save('nastyDetector.h5')