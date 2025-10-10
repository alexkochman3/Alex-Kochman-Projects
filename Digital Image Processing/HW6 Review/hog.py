import sys
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def computeGradients(image, size):
  Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=size)
  Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=size)
  return Ix, Iy

def computeGradientMagnitude(Ix, Iy):
  return np.sqrt(Ix**2 + Iy**2)

def computeGradientOrientation(Ix, Iy):
  theta = np.degrees(np.arctan2(Iy, Ix))
  theta = np.mod(theta + 360, 360)  #Ensure the range is [0, 360)
  return theta

def computeHistogram(theta, bins=360):
  H, bin_edges = np.histogram(theta, bins=bins, range=(0, 360))
  return H, bin_edges

image = cv2.imread(Path(sys.argv[1]), cv2.IMREAD_GRAYSCALE)

Ix, Iy = computeGradients(image, int(sys.argv[2])) #gradient scale is argument 2 in program call

M = computeGradientMagnitude(Ix, Iy)
theta = computeGradientOrientation(Ix, Iy)

H, bin_edges = computeHistogram(theta)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Gradient in X (Ix)")
plt.imshow(Ix, cmap='gray')
cv2.imwrite('gradientX.png', Ix)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Gradient in Y (Iy)")
plt.imshow(Iy, cmap='gray')
cv2.imwrite('gradientY.png', Iy)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Gradient Magnitude (M)")
plt.imshow(M, cmap='gray')
cv2.imwrite('gradientMag.png', M)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Gradient Orientation (Theta)")
plt.imshow(theta, cmap='hsv')
cv2.imwrite('gradientOrientation.png', theta)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Histogram of Orientations")
plt.plot(bin_edges[:-1], H)
plt.xlabel("Orientation (degrees)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

