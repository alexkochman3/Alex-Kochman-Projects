import cv2
import numpy as np
from pathlib import Path
import sys

#Translate the input image by (tx, ty) pixels
def translate(img, tx, ty):
  rows, cols = img.shape[:2]
  M = np.float32([[1, 0, tx], [0, 1, ty]])
  translated = cv2.warpAffine(img, M, (cols, rows))
  return translated

#Crop a rectangular region and scale it
def cropScale(img, x1, y1, x2, y2, s):
  cropped = img[y1:y2, x1:x2]
  newSize = (int(cropped.shape[1] * s), int(cropped.shape[0] * s))
  scaled = cv2.resize(cropped, newSize, interpolation=cv2.INTER_LINEAR)
  return scaled

#Flip image about the y-axis (vertical flip)
def verticalFlip(img):
  M = np.float32([[-1, 0, img.shape[1]], [0, 1, 0]])
  flipped = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
  return flipped

#Flip image about the x-axis (horizontal flip)
def horizontalFlip(img):
  M = np.float32([[1, 0, 0], [0, -1, img.shape[0]]])
  flipped = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
  return flipped

#Rotate the image by a given angle
def rotate(img, angle):
  rows, cols = img.shape[:2]
  center = (cols // 2, rows // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1)
  rotated = cv2.warpAffine(img, M, (cols, rows))
  return rotated

#Fill a rectangular region with an intensity value
def fill(img, x1, y1, x2, y2, val):
  filled = img.copy()
  filled[y1:y2, x1:x2] = val
  return filled


img = cv2.imread(Path(sys.argv[1]), 1)
translated = translate(img, 300, 200)
cv2.imshow('Translated', translated)
cv2.imwrite('translated.png', translated)
cv2.waitKey(0)
cropped = cropScale(img, 500, 1, 1000, 800, 0.5)
cv2.imshow('Cropped', cropped)
cv2.imwrite('cropped.png', cropped)
cv2.waitKey(0)
vFlipped = verticalFlip(img)
cv2.imshow('Vertical Flip', vFlipped)
cv2.imwrite('vflipped.png', vFlipped)
cv2.waitKey(0)
hFlipped = horizontalFlip(img)
cv2.imshow('Horizontal Flip', hFlipped)
cv2.imwrite('hflipped.png', hFlipped)
cv2.waitKey(0)
rotated = rotate(img, 60)
cv2.imshow('Rotated', rotated)
cv2.imwrite('rotated.png', rotated)
cv2.waitKey(0)
filled = fill(img, 500, 1, 1000, 800, 150)
cv2.imshow('Filled', filled)
cv2.imwrite('filled.png', filled)
cv2.waitKey(0)


