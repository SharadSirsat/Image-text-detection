import cv2
import numpy as np

samples = np.loadtxt('E:/minor2/SimpleOCR-2/generalsamples.data',np.float32)
responses = np.loadtxt('E:/minor2/SimpleOCR-2/generalresponses.data',np.float32)
responses = responses.astype(int)
responses = responses.reshape((responses.size,1))
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(samples, cv2.ml.ROW_SAMPLE, responses)
svm.save('E:\minor2\SimpleOCR-2\svm_data.dat')