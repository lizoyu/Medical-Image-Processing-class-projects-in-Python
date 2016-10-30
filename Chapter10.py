import cv2
import numpy as np

###################################################
#                                                                                                  #
#                      Chapter10 - Image Segmentation                      #
#                                                                                                  #
###################################################

#Global Thresholding
#NOTE
# threshold an image according to entire image
#INPUT
# img: input image
#OUTPUT
# res: threshold image
def globalThres(img):
    T = np.mean(img)
    thres = 0; dT = 1

    while(dT > thres):
        g1 = img[img > T]
        g2 = img[img <= T]
        Tpri = T
        T = (np.mean(g1) + np.mean(g2)) / 2
        dT = np.abs(T - Tpri)

    img[img > T] = 255
    img[img <= T] = 0
    res = img[:]
    
    return res

#Optimal Global Thresholding using Otsu‘s
#NOTE
# threshold an image in a sense that optimally separates intensity classes
#INPUT
# img: input image
#OUTPUT
# res: threshold image
def otsu(img):
    hist = cv2.calcHist([img], [0], None, [256], [0,256]) / img.size
    mg = np.average(img)
    p1 = np.zeros((256,1))
    m = np.zeros((256,1))
    p1[0] = hist[0]

    for i in range(1,256):
        p1[i] = p1[i-1] + hist[i]

    hist *= np.reshape(np.linspace(0,255,num = 256), (256,1))
    for i in range(1,256):
        m[i] = m[i-1] + hist[i]

    variance = np.nan_to_num(np.power(mg * p1 - m, 2) / (p1 * (1 - p1)))

    variance[variance != np.max(variance)] = 0
    variance[variance == np.max(variance)] = 1
    variance *= np.reshape(np.linspace(0,255,num = 256), (256,1))
    k = np.around(np.average(variance[variance != 0]))

    img[img > k] = 255
    img[img <= k] = 0

    res = img[:]
    return res
