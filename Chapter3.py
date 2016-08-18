import cv2
import numpy as np
import matplotlib.pyplot as plt

##############################################################
#                                                            #
#  Chapter3 - Intensity Transformations & Spatial Filtering  #
#                                                            #
##############################################################

#Histogram
#NOTE
# show the histogram of the image
#INPUT
# img: input image
#OUTPUT
# none
def hist(img):
    plt.hist(img.ravel(),256,[0,255])
    plt.show()

#Histogram Equalization
#NOTE
# equalize the distribution of the grey scale in an image according to its
# histogram
#INPUT
# img: input image
#OUTPUT
# res: processed image
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

#Discrete Fourier Transform
#NOTE
# Do the DFT on an image and show it
#INPUT
# img: input image
#OUTPUT
# none
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
#NOTE
# do the log transform on an image
#INPUT
# img: input image
# coefficient: the coefficient of the log transform function
#OUTPUT
# res: processed image
def log(img, coefficient):
    c = coefficient
    logimg = np.array(img, np.uint16)
    logimg = c * np.log10(logimg + 1)
    logimg = (255 * (logimg - np.min(logimg)))/(np.max(logimg)-np.min(logimg))
    res = np.array(logimg, np.uint8)

    return res

#Scalar Transform
#NOTE
# do the scalar transform on an image
#INPUT
# img: input image
# coefficient: the coefficient of the scalar transform function
# gamma: the power of the scalar transform function
#OUTPUT
# res: processed image
def scalr(img, coefficient, gamma):
    c = coefficient
    y = gamma
    scaimg = np.array(img, np.float32)
    scaimg = c * np.power(scaimg, y)
    scaimg = (255 * (scaimg - np.min(scaimg)))/(np.max(scaimg)-np.min(scaimg))
    res = np.array(scaimg, np.uint8)

    return res
