import cv2
import numpy as np
SZ=20
bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
samples = np.loadtxt('E:/minor2/generalsamples.data',np.float32)
responses = np.loadtxt('E:/minor2/generalresponses.data',np.float32)
responses = responses.astype(int)
responses = responses.reshape((responses.size,1))
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(samples, cv2.ml.ROW_SAMPLE, responses)
svm.save('E:\minor2\svm_data.dat')

for i in range(1,10):
    im = cv2.imread('E:/minor2/English2/English/Fnt/Sample00'+str(i)+'/img00'+str(i)+'-00500.png')
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sample = hog(gray)
    sample = np.float32(sample).reshape(-1,64)
    result = svm.predict(sample)
    ans = result[1][0][0].astype(int)
    string = chr(ans)
    print string
    
for i in range(11,63):
    im = cv2.imread('E:/minor2/English2/English/Fnt/Sample0'+str(i)+'/img0'+str(i)+'-00500.png')
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sample = hog(gray)
    sample = np.float32(sample).reshape(-1,64)
    result = svm.predict(sample)
    ans = result[1][0][0].astype(int)
    string = chr(ans)
    print string