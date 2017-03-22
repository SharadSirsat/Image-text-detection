import cv2
import numpy as np
from hoggenerator import hog

samples =  np.empty((0,64))
responses = []

for j in range(1, 10):
    for i in range(1, 10):
        im = cv2.imread('E:/minor2/English/Fnt/Sample00'+str(j)+'/img00'+str(j)+'-0000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        responses.append(int(47+j))
        sample = hog(gray)
        sample = np.float32(sample).reshape(-1,64)
        samples = np.append(samples,sample,0)
    
    for i in range(10, 100):
        im = cv2.imread('E:/minor2/English/Fnt/Sample00'+str(j)+'/img00'+str(j)+'-000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        responses.append(int(47+j))
        sample = hog(gray)
        sample = np.float32(sample).reshape(-1,64)
        samples = np.append(samples,sample,0)

for i in range(1, 10):
    im = cv2.imread('E:/minor2/English/Fnt/Sample010/img010-0000'+str(i)+'.png')
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    responses.append(int(57))
    sample = hog(gray)
    sample = np.float32(sample).reshape(-1,64)
    samples = np.append(samples,sample,0)

for i in range(10, 100):
    im = cv2.imread('E:/minor2/English/Fnt/Sample010/img010-000'+str(i)+'.png')
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    responses.append(int(57))
    sample = hog(gray)
    sample = np.float32(sample).reshape(-1,64)
    samples = np.append(samples,sample,0)
    
for j in range(11, 37):
    for i in range(1, 10):
        im = cv2.imread('E:/minor2/English/Fnt/Sample0'+str(j)+'/img0'+str(j)+'-0000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        responses.append(int(54+j))
        sample = hog(gray)
        sample = np.float32(sample).reshape(-1,64)
        samples = np.append(samples,sample,0)
        
    for i in range(10, 100):
        im = cv2.imread('E:/minor2/English/Fnt/Sample0'+str(j)+'/img0'+str(j)+'-000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        responses.append(int(54+j))
        sample = hog(gray)
        sample = np.float32(sample).reshape(-1,64)
        samples = np.append(samples,sample,0)
        
for j in range(37, 63):
    for i in range(1, 10):
        im = cv2.imread('E:/minor2/English/Fnt/Sample0'+str(j)+'/img0'+str(j)+'-0000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        responses.append(int(60+j))
        sample = hog(gray)
        sample = np.float32(sample).reshape(-1,64)
        samples = np.append(samples,sample,0)
        
    for i in range(10, 100):
        im = cv2.imread('E:/minor2/English/Fnt/Sample0'+str(j)+'/img0'+str(j)+'-000'+str(i)+'.png')
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        responses.append(int(60+j))
        sample = hog(gray)
        sample = np.float32(sample).reshape(-1,64)
        samples = np.append(samples,sample,0)
        
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
np.savetxt('E:\minor2\SimpleOCR-2\generalsamples.data',samples)
np.savetxt('E:\minor2\SimpleOCR-2\generalresponses.data',responses)
