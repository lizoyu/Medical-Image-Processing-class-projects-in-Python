import cv2
import numpy as np

###################################################
#                                                 #
#        Chapter6 - Color Image Processing        #
#                                                 #
###################################################

#Web-safe Color
def webcolor(img):
    from math import ceil
    res = np.vectorize(ceil)(img / 51.0) * 51

    return res

#RGB Histogram Equalization
def rgbequHist(img):
    b,g,r = cv2.split(img)

    b = lizoyu.equHist(b)
    g = lizoyu.equHist(g)
    r = lizoyu.equHist(r)

    res = cv2.merge([b,g,r])
    return res

#Pseudo Color Processing
def pseudo(img):
