#!/usr/bin/env python
# coding: utf-8

# In[36]:


# -*- coding: utf-8 -*-
__author__ = 'Kang'
 
import cv2

# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
# 영상이 있는 경로
vidcap = cv2.VideoCapture("D:/emotion/train/anger15.m2ts")

count = 1

while(vidcap.isOpened()):
    ret, image = vidcap.read()
    # 이미지 사이즈 960x540으로 변경
    # 15프레임당 하나씩 이미지 추출
    if(int(vidcap.get(1)) % 15 == 0):
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        # 추출된 이미지가 저장되는 경로
        cv2.imwrite("C:/emotion/a15/anger%d.jpg" % count, image)
        print('Saved frame%d.jpg' % count)
        count += 1
        
vidcap.release()


# In[9]:


import cv2
print(cv2.__version__)


# In[ ]:




