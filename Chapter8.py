import cv2
import numpy as np

###################################################
#                                                 #
#          Chapter8 - Image Compression           #
#                                                 #
###################################################

#Improved Grey Scale Quantization
#NOTE
# an improved algorithm compared to the one in Chapter 2
#INPUT
# img: input image
# GreyScaleLevel: the amount of grey scale used for quantization
#   - range: 0~8 int
#OUTPUT
# res: processed image

def igsq(img, GreyScaleLevel):
    height,width = img.shape[:2]
    x = 0;y = 0;sum = 0;

    while(x<(height-1)):
        grey = img[x,y]
        if(grey<240):
            sum = np.uint8((sum&15) + grey)
            img.itemset((x,y),sum)
        else:
            sum = grey
            img.itemset((x,y),sum)
        y += 1
        if(y>(width-1)):
            y = 0
        x += 1
    
    IntLv = GreyScaleLevel
    img = np.array(img, np.int16)
    res = ((img+1)//pow(2,8-IntLv)*pow(2,8-IntLv))-1
    return res

#RMS & SNR
#NOTE
# calculate the root mean square and signal-to-noise ratio of the image
#INPUT
# rawImg: input original image
# processedImg: input processed or noised image
#OUTPUT
# rms: root mean square
# snr: signal-to-noise ratio
def ofc(rawImg, processedImg):
    size = rawImg.size
    rawImg = np.array(rawImg, np.float32)
    processedImg = np.array(processedImg, np.float32)

    rms = np.sqrt((sum(np.ravel(np.square(processedImg - rawImg))))/size)
    snr = sum(np.ravel(np.square(processedImg)))/sum(np.ravel(np.square(processedImg - rawImg)))

    print 'RMS = ', rms, 'SNR = ', snr
    return rms, snr

#1-order & 2-order Image Entropy
#NOTE
# calculate 1 & 2-order image entropy of the image
#INPUT
# img: input original image
#OUTPUT
# entro: root mean square
# entro2: signal-to-noise ratio
def entropy(img):
    x = np.linspace(0, (img.size-1), num = img.size)
    y = np.linspace(1,255, num = 255)
    nk = np.zeros(256, np.uint16)
    ra = np.ravel(img)
    nk2 = np.zeros([256,256], np.uint16)

    for i in x:
        grey = ra[i]
        nk[grey] += 1

    nk2[0,0] = ra[0]
    for j in y:
        grey = ra[i]
        grey1 = ra[i-1]
        nk2[grey,grey1] += 1

    pr = nk/(img.size*1.0)
    pr2 = nk2/(img.size*1.0)
    pr = pr[pr != 0]
    pr2 = pr2[pr2 != 0]
    entro = -sum(pr * np.log2(pr))
    entro2 = -sum(pr2 * np.log2(pr2))
    print '1-order entropy is ', entro
    print '2-order entropy is ', entro2
    return entro, entro2
