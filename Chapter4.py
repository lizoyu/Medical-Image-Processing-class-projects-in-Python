import cv2
import numpy as np

###################################################
#                                                 #
#   Chapter4 - Filtering in the Frequency Domain  #
#                                                 #
###################################################

#Gaussian Blurring
#NOTE
# create the blur effect on an image
#INPUT
# img: input image
# cutOffFreq: the cut-off frequency of the gaussian filter
#OUTPUT
# res: processed image
def gaussian(img,cutOffFreq):
    height, width = img.shape
    img = np.array(img, dtype = np.float32)
    img = cv2.copyMakeBorder(img, 0, height, 0, width, cv2.BORDER_CONSTANT, value = 0)

    for x in range(0, 2*height):
        for y in range(0, 2*width):
            img[x,y] = img[x,y] * np.power(-1, (x+y))
            
    img_fq = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

    D = np.zeros((2*height, 2*width))
    D0 = cutOffFreq
    for u in range(0, 2*height):
        for v in range(0, 2*width):
            D[u,v] = np.sqrt(np.power((u - 2*height/2), 2)+np.power((v - 2*width/2), 2))
    H = np.zeros((2*height, 2*width))
    for u in range(0, 596):
        for v in range(0, 400):
            H[u,v] = np.exp(-np.power(D[u,v], 2)/(2*np.power(D0, 2))) # Gaussian

    img_fq[:,:,0] = img_fq[:,:,0] * H
    img_fq[:,:,1] = img_fq[:,:,1] * H
    img_sp = cv2.idft(img_fq)
    
    for x in range(0, 2*height):
        for y in range(0, 2*width):
            img_sp[x,y] = img_sp[x,y] * np.power(-1, (x+y))
            
    img_sp = img_sp[:,:,0]

    res = img_sp[0:height,0:width]
    res = np.round((255 * (res - np.min(res))) / (np.max(res) - np.min(res)))

    return res

#Laplacian Image Sharpening
#NOTE
# create the image sharpening effect using Laplacian
#INPUT
# img: input image
#OUTPUT
# res: processed image
def laplacian(img):
    height, width = img.shape
    img = np.array(img, dtype = np.float32)
    img_cp = img / np.max(img) # Normalization to [0,1]
    img = cv2.copyMakeBorder(img, 0, height, 0, width, cv2.BORDER_CONSTANT, value = 0)

    for x in range(0, 2*height):
        for y in range(0, 2*width):
            img[x,y] = img[x,y] * np.power(-1, (x+y))
    img = img / np.max(img)
    
    img_fq = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

    D = np.zeros((2*height, 2*width))
    for u in range(0, 2*height):
        for v in range(0, 2*width):
            D[u,v] = np.sqrt(np.power((u - 2*height/2), 2) + np.power((v - 2*width/2), 2))
    H = np.zeros((2*height, 2*width))
    for u in range(0, 596):
        for v in range(0, 400):
            H[u,v] = -4 * np.power(np.pi, 2) * np.power(D[u,v], 2) # Laplacian

    img_fq[:,:,0] = img_fq[:,:,0] * H
    img_fq[:,:,1] = img_fq[:,:,1] * H
    img_sp = cv2.idft(img_fq)
    
    for x in range(0, 2*height):
        for y in range(0, 2*width):
            img_sp[x,y] = img_sp[x,y] * np.power(-1, (x+y))

    img_sp = img_sp[:,:,0]

    res = img_sp[0:height,0:width]
    res = res / np.max(res) # Normalization to [0,1]

    res = 255 * (-res + img_cp) # Return to grey scale after enhancement
    return res

#Sobel Gradient
#NOTE
# calculate the gradient of an image using Sobel operators
#INPUT
# img: input image
#OUTPUT
# res: gradient image
# ang: the direction of the gradient
def sobel(img):
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 0)
    height, width = img.shape
    mag = np.zeros(img.shape)
    ang = np.zeros(img.shape)

    for i in range(1,height-1):
        for j in range(1,width-1):
            zdown = img[i+1,j-1] + 2*img[i+1,j] + img[i+1,j+1]
            zup = img[i-1,j-1] + 2*img[i-1,j] + img[i-1,j+1]
            zright = img[i-1,j+1] + 2*img[i,j+1] + img[i+1,j+1]
            zleft = img[i-1,j-1] + 2*img[i,j-1] + img[i+1,j-1]
            mag[i,j] = np.abs(zdown-zup) + np.abs(zright-zleft)
            ang[i,j] = np.arctan2(zright-zleft,zdown-zup) * 180 / np.pi
            if (ang[i,j] < 0):
                ang[i,j] = ang[i,j] + 180
            if (ang[i,j] == 180):
                ang[i,j] = 0

    mag = mag[1:height-1,1:width-1]
    ang = ang[1:height-1,1:width-1]
    res = np.round((255 * (mag - np.min(mag))) / (np.max(mag) - np.min(mag)))
    return res, ang

#Canny Edge Detecting
#NOTE
# detect the edge of an image
#INPUT
# img: input image
# lowThres,highThres: double threshold value
#OUTPUT
# mag_nms: processed image
def canny(img, lowThres, highThres):
    height, width = img.shape

    img = gaussian(img, 400)

    mag, ang = sobel(img)

    #non-maximum suppression
    ang = np.vectorize(round)(ang / 45.0) * 45
    mag_nms = mag[:]
    for i in range(1,height-1):
        for j in range(1,width-1):
            if (ang[i,j] == 0 and max(mag[i,j-1],mag[i,j],mag[i,j+1]) != mag[i,j]):
                mag_nms[i,j] = 0
            if (ang[i,j] == 45 and max(mag[i-1,j+1],mag[i,j],mag[i+1,j-1]) != mag[i,j]):
                mag_nms[i,j] = 0
            if (ang[i,j] == 90 and max(mag[i-1,j],mag[i,j],mag[i+1,j]) != mag[i,j]):
                mag_nms[i,j] = 0
            if (ang[i,j] == 135 and max(mag[i-1,j-1],mag[i,j],mag[i+1,j-1]) != mag[i,j]):
                mag_nms[i,j] = 0

    #double threshold
    lowThres = 50
    highThres = 70
    for i in range(1,height-1):
        for j in range(1,width-1):
            if (mag_nms[i,j] > highThres):
                mag_nms[i,j] = 255
            elif (mag_nms[i,j] < lowThres):
                mag_nms[i,j] = 0
            else:
                mag_nms[i,j] = 1

    #weak edge analysis
    for i in range(1,height-1):
        for j in range(1,width-1):
            if (mag_nms[i,j] == 1 and np.sum(mag_nms[i-1:i+1,j-1:j+1]) >= 255):
                mag_nms[i,j] = 255
    mag_nms[mag_nms==1] = 0

    return mag_nms
