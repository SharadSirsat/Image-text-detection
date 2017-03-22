import cv2
import numpy as np
from hoggenerator import hog

svm = cv2.ml.SVM_load('E:\minor2\SimpleOCR-2\svm_data.dat')

im = cv2.imread('E:/minor2/TestImages/testimage.jpeg')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

out = np.zeros(im.shape,np.uint8)

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>35 and w>35:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            sample = hog(roi)
            sample = np.float32(sample).reshape(-1,64)
            result = svm.predict(sample)
            ans = result[1][0][0].astype(int)
            string = chr(ans)
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
            
cv2.imshow('im',im)
cv2.imwrite('E:/minor2/TestImages/testimagecontour.jpeg', im)
cv2.imshow('out',out)
cv2.waitKey(0)