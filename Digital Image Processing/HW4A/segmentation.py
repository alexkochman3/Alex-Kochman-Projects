from pathlib import Path
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import cv2 as cv




img = cv.imread(Path(sys.argv[1]), 1) 
#img2 = cv.imread(Path(sys.argv[2]), 1) 
sigma = int(sys.argv[3]) #third arg is sigma/filter size if median
filterType = int(sys.argv[2]) #1 for gaussian, 0 for median
cv.imshow("Image 1: ", img1)
#cv.imshow("Image 2: ", img2)
cv.waitKey(0)

if(filterType == 1):
  cv.GaussianBlur(img, (sigma, sigma), 0)
elif(filterType == 0):
  cv.MedianBlur(img, sigma)
"""elif(filterType == 2):
  amfImg = adaptiveMedFilter(img2, sigma)
  cv.imshow("AMF: ", amfImg)
  cv.waitKey(0)
  cv.imwrite("adaptiveMedImg.png", amfImg)
  #print("MSE between original and computed: " + str(meanSqErr(img1, amfImg)))"""