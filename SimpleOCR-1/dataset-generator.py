import numpy as np
import cv2

samples =  np.empty((0,100))
responses = []

for j in range(1, 10):
    for i in range(1, 10):
        im = cv2.imread('E:/minor2/English/Fnt/Sample00'+str(j)+'/img00'+str(j)+'-0000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        roismall = cv2.resize(gray,(10,10))
        responses.append(int(47+j))
        sample = roismall.reshape((1,100))
        samples = np.append(samples,sample,0)
    for i in range(10, 100):
        im = cv2.imread('E:/minor2/English/Fnt/Sample00'+str(j)+'/img00'+str(j)+'-000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        roismall = cv2.resize(gray,(10,10))
        responses.append(int(47+j))
        sample = roismall.reshape((1,100))
        samples = np.append(samples,sample,0)
for i in range(1, 10):
    im = cv2.imread('E:/minor2/English/Fnt/Sample010/img010-0000'+str(i)+'.png')
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    roismall = cv2.resize(gray,(10,10))
    responses.append(int(57))
    sample = roismall.reshape((1,100))
    samples = np.append(samples,sample,0)
for i in range(10, 100):
    im = cv2.imread('E:/minor2/English/Fnt/Sample010/img010-000'+str(i)+'.png')
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    roismall = cv2.resize(gray,(10,10))
    responses.append(int(57))
    sample = roismall.reshape((1,100))
    samples = np.append(samples,sample,0)
for j in range(11, 37):
    for i in range(1, 10):
        im = cv2.imread('E:/minor2/English/Fnt/Sample0'+str(j)+'/img0'+str(j)+'-0000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        roismall = cv2.resize(gray,(10,10))
        responses.append(int(54+j))
        sample = roismall.reshape((1,100))
        samples = np.append(samples,sample,0)
    for i in range(10, 100):
        im = cv2.imread('E:/minor2/English/Fnt/Sample0'+str(j)+'/img0'+str(j)+'-000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        roismall = cv2.resize(gray,(10,10))
        responses.append(int(54+j))
        sample = roismall.reshape((1,100))
        samples = np.append(samples,sample,0)
for j in range(37, 63):
    for i in range(1, 10):
        im = cv2.imread('E:/minor2/English/Fnt/Sample0'+str(j)+'/img0'+str(j)+'-0000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        roismall = cv2.resize(gray,(10,10))
        responses.append(int(60+j))
        sample = roismall.reshape((1,100))
        samples = np.append(samples,sample,0)
    for i in range(10, 100):
        im = cv2.imread('E:/minor2/English/Fnt/Sample0'+str(j)+'/img0'+str(j)+'-000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        roismall = cv2.resize(gray,(10,10))
        responses.append(int(60+j))
        sample = roismall.reshape((1,100))
        samples = np.append(samples,sample,0)
        
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
np.savetxt('E:\minor2\SimpleOCR-1\generalsamples.data',samples)
np.savetxt('E:\minor2\SimpleOCR-1\generalresponses.data',responses)
