from pathlib import Path
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import cv2 as cv

def cvGaussian(img, sigma):
  gaussian = cv.GaussianBlur(img, (0,0), sigma)
  cv.imshow("Gaussian: ", gaussian)
  cv.imwrite("gaussian.png", gaussian)
  cv.waitKey(0)

def cvMedian(img, filterDim):
  median = cv.medianBlur(img, filterDim)
  cv.imshow("Median: ", median)
  cv.imwrite("median.png", median)
  cv.waitKey(0)

def blob_detection_pre_blurred(imagePath, kernelSize, thresholdBright, thresholdDark):
  grayImg = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)

  logImg = cv.Laplacian(grayImg, cv.CV_64F, ksize=kernelSize)

  #Separate the bright and dark blobs
  brightBlobs = np.where(logImg < 0, logImg, 0)
  darkBlobs = np.where(logImg > 0, logImg, 0) 

  #Normalize the bright and dark blobs for visualization
  brightBlobsNormalized = cv.normalize(brightBlobs, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
  darkBlobsNormalized = cv.normalize(darkBlobs, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

  #Apply thresholding to both bright and dark blobs to create binary detection masks
  _, binaryBrightBlobs = cv.threshold(brightBlobsNormalized, thresholdBright, 255, cv.THRESH_BINARY)
  _, binaryDarkBlobs = cv.threshold(darkBlobsNormalized, thresholdDark, 255, cv.THRESH_BINARY)

  #Display the images (grayscale, bright blobs, dark blobs)
  fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  
  #Show grayscale image
  axes[0].imshow(grayImg, cmap='gray')
  cv.imwrite("grayImage.png", grayImg)
  axes[0].set_title("Grayscale Image (Pre-blurred)")
  axes[0].axis('off')

  #Show bright blobs
  axes[1].imshow(brightBlobs, cmap='gray')
  cv.imwrite("brightImage.png", binaryBrightBlobs)
  axes[1].set_title("Bright Blobs (Negative LoG)")
  axes[1].axis('off')

  #Show dark blobs
  axes[2].imshow(darkBlobs, cmap='gray')
  cv.imwrite("darkImage.png", binaryDarkBlobs)
  axes[2].set_title("Dark Blobs (Positive LoG)")
  axes[2].axis('off')

  #Show binary detection masks (combined)
  combinedMask = cv.bitwise_or(binaryBrightBlobs, binaryDarkBlobs)
  cv.imwrite("output.png", combinedMask)
  axes[3].imshow(combinedMask, cmap='gray')
  axes[3].set_title("Binary Detection Mask (Bright & Dark Blobs)")
  axes[3].axis('off')

  plt.show()

img1 = cv.imread(Path(sys.argv[1]), 1) #First arg is clear filtered image
sigma = sys.argv[2] #Second arg is sigma/filter size if median or LoG
filterType = int(sys.argv[3]) #1 for gaussian, 0 for median
threshold1 = int(sys.argv[4]) #If using blob detection function
threshold2 = int(sys.argv[5])
cv.imshow("Image: ", img1)
cv.waitKey(0)
if(filterType == 1):
  cvGaussian(img1, float(sigma))
elif(filterType == 0):
  cvMedian(img1, sigma)
elif(filterType == 2):
  blob_detection_pre_blurred(Path(sys.argv[1]), int(sigma), threshold1, threshold2)




