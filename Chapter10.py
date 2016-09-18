import cv2
import numpy as np

###################################################
#                                                 #
#         Chapter10 - Image Segmentation          #
#                                                 #
###################################################

#Global Threshold
#NOTE
# threshold an image according to its global gray scale
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

def otsu():

    return res

def regGrow():

    return res
