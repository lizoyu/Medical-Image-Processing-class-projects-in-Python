import cv2
import numpy as np

###################################################
#                                                 #
#   Chapter4 - Filtering in the Frequency Domain  #
#                                                 #
###################################################

#Gaussian Filtering
def gaussian(img,coresize):
    res = cv2.GaussianBlur(img, coresize, 0)

    return res

#Laplacian Filtering
def laplaian(img):
    laplacian = cv2.Laplacian(img, cv2.CV_16S)
    res = cv2.convertScaleAbs(laplacian)
           
    res = np.array(res, np.uint8)
    return res

#Sobel Filtering
def sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    res = sobelx + sobely

    return res

#Canny Filtering
def canny(img, lower, upper):
    res = cv2.Canny(img, lower, upper)

    return res
