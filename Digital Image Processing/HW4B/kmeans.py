import cv2 as cv
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def segmentImage(imagePath, k, featureSet='gray'):
  # Read the image
  image = cv.imread(imagePath)
  imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  
  # Feature extraction
  if featureSet == 'gray':
    # Convert to grayscale
    gray = cv.cvtColor(imageRGB, cv.COLOR_RGB2GRAY)
    features = gray.reshape((-1, 1))  # Reshape to (n, 1)
  elif featureSet == 'rgb':
    features = imageRGB.reshape((-1, 3))  # Reshape to (n, 3)
  else:
    raise ValueError("Invalid feature set.")
  
  # K-means clustering
  kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
  segmentedImg = kmeans.labels_.reshape(imageRGB.shape[:2])
  if featureSet == 'gray':
    return imageRGB, gray, segmentedImg
  else:
    return imageRGB, segmentedImg

def generateNucleiMask(segmentedImg, cluster_id=1):
  nucleiMask = (segmentedImg == cluster_id).astype(np.uint8) * 255
  return nucleiMask

def saveImages(imageRGB, segmentedImg, nucleiMask, gray = []):
  # Save images
  cv.imwrite('rgbImg.png', cv.cvtColor(imageRGB, cv.COLOR_RGB2BGR))
  if gray[0][0]:
    cv.imwrite('grayImg.png', gray)
  cv.imwrite('segmentedImg.png', (segmentedImg * 127).astype(np.uint8))
  cv.imwrite('nucleiMaskImg.png', nucleiMask)
    
def displayResults(imageRGB, gray, segmentedImg, nucleiMask, featureSet):
  plt.figure(figsize=(20, 5))
  
  plt.subplot(1, 5, 1)
  plt.imshow(imageRGB)
  plt.title(f'Original Image ({featureSet})')
  plt.axis('off')
  
  plt.subplot(1, 5, 2)
  plt.imshow(gray, cmap='gray')
  plt.title(f'Pre-processed Image ({featureSet})')
  plt.axis('off')
  
  plt.subplot(1, 5, 3)
  plt.imshow(segmentedImg, cmap='gray')
  plt.title(f'K-means Clustering Output ({featureSet})')
  plt.axis('off')
  
  plt.subplot(1, 5, 4)
  plt.imshow(nucleiMask, cmap='gray')
  plt.title(f'Nuclei Mask ({featureSet})')
  plt.axis('off')
  
  plt.show()

# File paths
imagePaths = [sys.argv[1], sys.argv[2]]

# Segment the images with two different feature sets
for imagePath in imagePaths:
  for featureSet in ['gray', 'rgb']:
    if featureSet == 'gray':
      imageRGB, gray, segmentedImg = segmentImage(imagePath, k=2, featureSet=featureSet)
    else:
      imageRGB, segmentedImg = segmentImage(imagePath, k=2, featureSet=featureSet)
    nucleiMask = generateNucleiMask(segmentedImg)
    # Save images
    if gray[0][0]:
      saveImages(imageRGB, segmentedImg, nucleiMask, gray)
    else:
      saveImages(imageRGB, segmentedImg, nucleiMask)
    
    # Display results
    displayResults(imageRGB, gray, segmentedImg, nucleiMask, featureSet)
