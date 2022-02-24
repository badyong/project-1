#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os, re, glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model
import collections
import pandas as pd
from pytube import YouTube
import moviepy.editor as mp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser
import shutil


# In[15]:


video_dir = "C:/Users/yongjae/Data/"
video_name = "test13.mp4"


# In[17]:


count = 1
test = []
dic = {}
result = []

shutil.rmtree(video_dir + "frame/")
os.makedirs(video_dir + "frame/")
vidcap = cv2.VideoCapture(video_dir + video_name)

while(vidcap.isOpened()):
    ret, image = vidcap.read()
    if image is None :
        break
    
    # 이미지 사이즈 960x540으로 변경
    image = cv2.resize(image, (960, 540))
     
    # 10프레임당 하나씩 이미지 추출
    if(int(vidcap.get(1)) % 10 == 0):
        cv2.imwrite(video_dir + "frame/" +  video_name + "-%d.png" % count, image)
        count += 1
        
vidcap.release()

# x1 x2 x3 카테고리 나눠보자.

categories = ["angry", "dark", "happy", "sad"]

def Dataization(img_path):
    image_w = 100
    image_h = 100
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)

for file in os.listdir(video_dir + 'frame/'):
    if (file.find('.png')):   
        test.append(Dataization(video_dir + 'frame/' + file))
        
test = np.array(test)
model = load_model('VideoClassification.h5')
predict = model.predict_classes(test)
 
for i in range(len(test)):
    result.append(str(categories[predict[i]]))

dic = collections.Counter(result)

keyword = max(dic.keys(), key=(lambda k:dic[k]))

print("FINAL MOOD :", keyword)


# In[ ]:




