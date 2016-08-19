import cv2
import numpy as np

###################################################
#                                                 #
#        Chapter6 - Color Image Processing        #
#                                                 #
###################################################

#Web-safe Color
#NOTE
# use web-safe color to create a image from input image
#INPUT
# img: input image
#OUTPUT
# res: processed image
def webcolor(img):
    from math import ceil
    res = np.vectorize(ceil)(img / 51.0) * 51

    return res

#RGB Histogram Equalization
#NOTE
# equalize the distribution of (R,G,B) according to each histogram, notice that
# histogram equalization to each channel is achieved by importing the function
# from Chapter3.py
#INPUT
# img: input image
#OUTPUT
# res: processed image
def rgbequHist(img):
    from Chapter3 import equHist
    b,g,r = cv2.split(img)

    b = equHist(b)
    g = equHist(g)
    r = equHist(r)

    res = cv2.merge([b,g,r])
    return res

#Pseudo Color Processing NEED REWRITE
#NOTE
# create the pseudo color effect on an image
#INPUT
# img: input image
#OUTPUT
# res: processed image
def pseudo(img):
    b,g,r = cv2.split(img)

    lut = np.linspace(0,255,num = 256)
    lut[0:25] = 255
    r = cv2.LUT(r, lut)
    lut[0:25] = 243
    g = cv2.LUT(g, lut)
    lut[0:25] = 63
    b = cv2.LUT(b, lut)

    res = cv2.merge([b,g,r])
    return res
