import cv2
import numpy as np

###################################################
#                                                 #
#  Chapter2 - The Fundamentals of Digital Image   #
#                                                 #
###################################################

#Grey Scale Quantization / Intensity Reduction
#NOTE
# It's used for quantizing the grey scale in an image.
#INPUT
# img: input image
# GreyScaleLevel: the amount of grey scale used for quantization
#   - range: 0~8 int
#OUTPUT
# res: processed image
def gsq(img, GreyScaleLevel):
    IntLv = GreyScaleLevel
    img = np.array(img, np.int16)
    res = ((img+1)//pow(2,8-IntLv)*pow(2,8-IntLv))-1
    return res

#Halftone Printing
#NOTE
# Create the effect of halftone printing on newspaper
#INPUT
# img: input image
#OUTPUT
# empty: processed image
def halftone(img):
    height,width = img.shape[:2]
    empty = np.zeros((3*height,3*width), np.uint8)
    x=0;y=0;x1=0;y1=0;

    while(x<(3*height-1)):
        grey = img[x1,y1]
        if(grey==0):
            pass
        elif(grey>0):
            empty.itemset((x,y+1),255)
            if(grey>32):
                empty.itemset((x+2,y+2),255)
                if(grey>64):
                    empty.itemset((x,y),255)
                    if(grey>96):
                        empty.itemset((x+2,y),255)
                        if(grey>128):
                            empty.itemset((x,y+2),255)
                            if(grey>160):
                                empty.itemset((x+1,y+2),255)
                                if(grey>192):
                                    empty.itemset((x+2,y+1),255)
                                    if(grey>224):
                                        empty.itemset((x+1,y),255)
                                    else:
                                        empty.itemset((x+1,y+1),255)
        y += 3
        y1 += 1
        if(y1>(width-1)):
            y1 = 0
            x1 += 1
        if(y>(3*width-1)):
            y = 0
            x += 3

    return empty

#Resize
#NOTE
# change the size of the image according to the ratio
#INPUT
# img: input image
# ratio: the resize ratio, shape x ratio = new shape
# Interpolation: method of interpolation
#   - range:
#       cv2.INTER_AREA (recommend when ratio < 1)
#       cv2.INTER_CUBIC
#       cv2.INTER_LINEAR (recommend when ratio > 1)
#OUTPUT
# res: processed image
def resize(img, ratio, Interpolation):
    n = ratio
    itp = Interpolation
    height,width = img.shape[:2]
    res = cv2.resize(img,(n*width,n*height), interpolation = itp)

    return res

#Arithmetic
#NOTE
# add, minus, multiply and divide two images
#INPUT
# img1: input image 1
# img2: input image 2
# method: the way to deal with these two images
#   - range:
#       'add'
#       'minus'
#       'multi'
#       'divide'
#OUTPUT
# res: processed image
def arith(img1,img2,method):
    height,width = img1.shape
    img2 = cv2.resize(img2,(width,height))
    img1 = np.array(img1, np.float32)
    img2 = np.array(img2, np.float32)

    if(method == 'add'):
        img = img1 + img2
    if(method == 'minus'):
        img = img1 - img2
    if(method == 'multi'):
        img = img1 * img2
    if(method == 'divide'):
        img = img1/img2

    res = (255 * (img - np.min(img))) / (np.max(img) - np.min(img))
    res = np.array(res, np.uint8)
    return res
