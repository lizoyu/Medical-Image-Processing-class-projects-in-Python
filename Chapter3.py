import cv2
import numpy as np
import matplotlib.pyplot as plt

##############################################################
#                                                            #
#  Chapter3 - Intensity Transformations & Spatial Filtering  #
#                                                            #
##############################################################

#Histogram
def hist(img):
    plt.hist(img.ravel(),256,[0,255])
    plt.show()

#Histogram Equalization
def equHist(img):
    size = img.size
    x = np.linspace(0,(size-1),num = size)
    nk = np.zeros(256, np.uint16)
    ra = np.ravel(img)

    for i in x:
        grad = ra[i]
        nk[grad] += 1

    pr = nk/(size*1.0)
    lut = np.zeros(256, np.float16)
    y = np.linspace(1,255,num = 255)
    lut[0] = pr[0] * 255
        
    for j in y:
        lut[j] = lut[j-1] + pr[j] * 255

    lut = np.array(lut, np.uint8)
    res = cv2.LUT(img, lut)
    return res

#Discreet Fourier Transform
def fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    spect = 20 * np.log(np.abs(fshift))
    avg = f[0,0]
  
    plt.subplot(121), plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(spect, cmap = 'gray')
    plt.title('Average Value:%d' %avg), plt.xticks([]), plt.yticks([])
    plt.show()

#Log Transform
def log(img, coefficient):
    c = coefficient
    logimg = np.array(img, np.uint16)
    logimg = c * np.log10(logimg + 1)
    logimg = (255 * (logimg - np.min(logimg)))/(np.max(logimg)-np.min(logimg))
    res = np.array(logimg, np.uint8)

    return res

#Scalar Transform
def scalr(img, coefficient, gamma):
    c = coefficient
    y = gamma
    scaimg = np.array(img, np.float32)
    scaimg = c * np.power(scaimg, y)
    scaimg = (255 * (scaimg - np.min(scaimg)))/(np.max(scaimg)-np.min(scaimg))
    res = np.array(scaimg, np.uint8)

    return res
