# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:25:09 2017

@author: manoj
"""

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import load_model

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

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

trainingset = train_datagen.flow_from_directory('datasets/train',target_size=(64,64),batch_size=25,class_mode='binary')

testset = test_datagen.flow_from_directory('datasets/test',target_size=(64,64),batch_size=25,class_mode='binary')

classifier.fit_generator(trainingset,steps_per_epoch=10,epochs=50,verbose=1,callbacks=None,validation_data=testset,validation_steps=10)

from keras.preprocessing import image
import numpy as np

targetimage = "datasets/Singletest/29.jpg"
imageobj = image.load_img(targetimage,target_size=(64,64))
test_img=image.img_to_array(imageobj)
test_img=np.expand_dims(test_img,axis=0)
prediction = classifier.predict(test_img)
trainingset.class_indices


##CNN goes here.