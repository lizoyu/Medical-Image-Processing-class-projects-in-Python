import cv2
import numpy as np

###################################################
#                                                 #
#  Chapter5 - Image Restoration & Reconstruction  #
#                                                 #
###################################################

#Gaussian Noise Addition
#NOTE
# add a gaussian noise to an image
#INPUT
# img: input image
# mean: the average value of the noise
# variance: the variance of the noise
#OUTPUT
# img_gaussed: processed image
def addGaussianNoise(img, mean, variance):
    height,width = img.shape[:2]
    gauss = np.random.normal(mean, np.sqrt(variance), (height,width))
    img_gaussed = np.around(img + gauss)
    img_gaussed = (255 * (img_gaussed - np.min(img_gaussed)))/(
                    np.max(img_gaussed)-np.min(img_gaussed))
    return img_gaussed


#Salt-and-Pepper Noise Addition
#NOTE
# add salt and pepper noise to an image
#INPUT
# img: input image
# saltprob: the occuring probability of salt noise
#   - range: 0 ~ 1
# pepperprob: the occuring probability of pepper noise
#   - range: 0 ~ 1
#OUTPUT
# img_sped: processed image
def addSaltPepperNoise(img, saltprob, pepperprob):
    height,width = img.shape[:2]

    salt = np.ones(round(saltprob*img.size), np.uint8)
    salt[:] = 255
    salt = np.reshape((np.random.permutation(np.append(salt,np.zeros(img.size-round(saltprob*img.size), np.uint8)))), img.shape)

    pepper = np.zeros(round(pepperprob*img.size), np.uint8)
    pepper = np.reshape((np.random.permutation(np.append(pepper, np.ones(img.size-round(pepperprob*img.size), np.uint8)))), img.shape)

    img_sped = cv2.add(img, salt) * pepper
    return img_sped

#1-D Random Number Generator
#NOTE
# generate an array of random numbers
#INPUT
# size: the length of the array
#OUTPUT
# r: an array of random numbers
def rand_1D(size):
    r = np.random.rand(size)
    return r

#Motion Blur
#NOTE
# create the motion blur effect on an image
#INPUT
# img: input image
# a,b: scalars of input image
# T: duration of exposure
#OUTPUT
# res: processed image
def blurring(img,a,b,T):
    height,width = img.shape[:2]
    H = np.zeros((height,width), np.complex)

    u=0;v=0;
    while(u<height):
        c = np.pi*((u+1-(height/2))*a+(v+1-(width/2))*b)
        if c == 0:
            H[u,v] = 1
        else:
            H[u,v] = (T * np.sin(c) * np.exp(complex(0,-c))) / c
        v += 1
        if(v>(width-1)):
            v = 0
            u += 1

    f = np.fft.fftshift(np.fft.fft2(img))
    f = f * H
    res = np.fft.ifft2(np.fft.ifftshift(f))
    res = np.array(res, dtype = np.int)
           
    return res

#Wiener Filtering
#NOTE
# restore the degraded image using Wiener filtering, if you want to change the
# degradation function, please change H[u,v] by yourself.
#INPUT
# img: input image (degraded image)
# K: interactive constant to get the best result
#OUTPUT
# res: processed image
def wiener(img, K):
    height,width = img.shape[:2]
    H = np.zeros((height,width), np.complex)
    a = 0.1
    b = 0.1
    T = 1

    u=0;v=0;
    while(u<height):
        c = np.pi*((u+1-(height/2))*a+(v+1-(width/2))*b)
        if c == 0:
            H[u,v] = 1
        else:
            H[u,v] = (T * np.sin(c) * np.exp(complex(0,-c))) / c
        v += 1
        if(v>(width-1)):
            v = 0
            u += 1

    #K = 0.065
    coe = (np.power(np.abs(H),2)) / (H * (np.power(np.abs(H),2)+K))
    G = np.fft.fftshift(np.fft.fft2(img))
    F = coe * G
    img = np.fft.ifft2(np.fft.ifftshift(F))

    res = np.array(img, np.uint8)
    return res
