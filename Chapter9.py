# -*- coding: cp936 -*-
import cv2
import numpy as np

###################################################
#                                                 #
#  Chapter9 - Morphological Image Processing      #
#                                                 #
###################################################

#Erosion
#Note:
#SElength - 结构元的边长
#SEMorph - 结构元的形状
#  包含在openCV中,形式是cv2.MORPH_*,具体请查看opencv说明
#
def erosion(img,SElength,SEMorph):
    size = SElength
    b = (size-1)/2
    img = cv2.copyMakeBorder(img,b,b,b,b,cv2.BORDER_CONSTANT,value = 0)
    SE = cv2.getStructuringElement(SEMorph,(size,size))
    height,width = img.shape[:2]
    img[img==255] = 1
    newimg = np.zeros(img.shape)

    x = 0;y = 0;
    while(x<=(height-size)):
        imgx = (img[x:x+size,y:y+size]+SE)*SE
        if(np.min(imgx[imgx!=0]) == 1):
            newimg[x+1,y+1] = 0
        else:
            newimg[x+1,y+1] = 1
        y += 1
        if(y>=(width-size)):
            y = 0
            x += 1

    newimg[newimg==1] = 255
    newimg = newimg[b:height-b,b:width-b] 
    return newimg

#Dilation
#Note:
#输入参数说明与腐蚀的相同
#
def dilation(img,SElength,SEMorph):
    size = SElength
    b = (size-1)/2
    img = cv2.copyMakeBorder(img,b,b,b,b,cv2.BORDER_CONSTANT,value = 0)
    SE = cv2.getStructuringElement(SEMorph,(size,size))
    height,width = img.shape[:2]
    img[img==255] = 1
    newimg = np.zeros(img.shape)

    x = 0;y = 0;
    while(x<=(height-size)):
        if(np.max((img[x:x+3,y:y+3]+SE)*SE) == 2):
            newimg[x+1,y+1] = 1
        else:
            newimg[x+1,y+1] = 0
        y += 1
        if(y>=(width-size)):
            y = 0
            x += 1

    newimg[newimg==1] = 255
    newimg = newimg[b:height-b,b:width-b] 
    return newimg

#Intersection
def intersection(img1,img2):
    img1[img1==255] = 1
    img2[img2==255] = 1
    
    img3 = img1 * img2

    img3[img3==1] = 255
    return img3

#Differencing
def differencing(img1,img2):
    img3 = img1-img2
    return img3

#Complementation
def complementation(img):
    img[img==255] = 1

    img += 1
    img[img==2] = 0

    return img
