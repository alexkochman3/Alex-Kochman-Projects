from pathlib import Path
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import cv2 as cv


def meanSqErr(img1, img2):
  m = img1.shape[0]
  n = img1.shape[1]

  if(m != img2.shape[0] or n != img2.shape[1]):
    print("Images have different dimensions")
    return
  
  totalError = 0
  for y in range(n):
    for x in range(m):
      error = np.float64(img1[x, y][0]) - np.float64(img2[x,y][0])
      error = error ** 2
      totalError += error
    
  return totalError / (m*n)

def cvGaussian(img, toCompareTo, sigma):
  gaussian = cv.GaussianBlur(img, (0,0), sigma)
  cv.imshow("Gaussian: ", gaussian)
  cv.imwrite("gaussian.png", gaussian)
  cv.waitKey(0)
  #print("MSE: " + str(meanSqErr(toCompareTo, gaussian)))

def cvMedian(img, toCompareTo, filterDim):
  median = cv.medianBlur(img, filterDim)
  cv.imshow("Median: ", median)
  cv.imwrite("median.png", median)
  cv.waitKey(0)
  #print("MSE: " + str(meanSqErr(toCompareTo, median)))

#Helper function to extract a neighborhood window
def getWindow(img, x, y, windowSize, row, col):
  halfSize = windowSize // 2
  return img[max(0, x-halfSize):min(row, x+halfSize+1), 
          max(0, y-halfSize):min(col, y+halfSize+1)]

def adaptiveMedFilter(img, S_max):
  # Define the output image
  startTime = time.time()
  outputImage = np.copy(img)
  row = img.shape[0]
  col = img.shape[1]
  
  #Define the initial window size (start from 3x3)
  initialWindowSize = 3
  
  #Loop through every pixel in the image
  for x in range(row):
    for y in range(col):
      windowSize = initialWindowSize
          
      while True:
        #Get the neighborhood window around pixel (x, y)
        window = getWindow(img, x, y, windowSize, row, col)
        
        zMin = np.min(window)
        #print(zMin)
        zMax = np.max(window)
        #print(zMax)
        zMed = np.int64(np.median(window))
        #print(zMed)
        zxy = img[x, y][0]
        
        #Stage A
        if zMin < zMed < zMax:
          #Stage B
          if zMin < zxy < zMax:
            outputImage[x, y] = zxy  #zxy is not an impulse
          else:
            outputImage[x, y] = zMed  #zxy is an impulse
          break  #Exit loop for this pixel
            
        #Increase window size if possible
        if windowSize < S_max:
          windowSize += 2  #Increase window size (3x3 -> 5x5 -> 7x7, etc.)
        else:
          outputImage[x, y] = zMed  #If window size exceeds S_max, use zMed
          break
  endTime = time.time() - startTime
  print(endTime)
  return outputImage

img1 = cv.imread(Path(sys.argv[1]), 1) #first arg is clear image
img2 = cv.imread(Path(sys.argv[2]), 1) #second arg is noisy image
sigma = int(sys.argv[3]) #third arg is sigma/filter size if median
filterType = int(sys.argv[4]) #2 for adaptive median, 1 for gaussian, 0 for median
cv.imshow("Image 1: ", img1)
cv.imshow("Image 2: ", img2)
cv.waitKey(0)
print("MSE between two original: " + str(meanSqErr(img1, img2)))
if(filterType == 1):
  cvGaussian(img2, img1, sigma)
elif(filterType == 0):
  cvMedian(img2, img1, sigma)
elif(filterType == 2):
  amfImg = adaptiveMedFilter(img2, sigma)
  cv.imshow("AMF: ", amfImg)
  cv.waitKey(0)
  cv.imwrite("adaptiveMedImg.png", amfImg)
  print("MSE between original and computed: " + str(meanSqErr(img1, amfImg)))



