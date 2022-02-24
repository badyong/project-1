#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
import numpy as np
import cv2
 
X_train, X_test, Y_train, Y_test = np.load('C:/yongjae/VideoClassification.npy', allow_pickle=True)


# In[2]:


X_train.shape


# In[3]:



 
model = Sequential()

model.add(Convolution2D(16, 3, 3, padding='same', activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))

model.add(Dense(4,activation = 'softmax'))  
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[4]:


model.summary()


# In[6]:



model.fit(X_train, Y_train, batch_size=32, epochs=10)
 
# VideoClassification0520 날짜 정해넣기


model.save('VideoClassification.h5')

