from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2 as cv
import time
import skimage
from skimage.filters import threshold_otsu




# Otsu's method for a single channel
def otsuRecursive(channel):
    startTime = time.time()
    hist, binEdges = np.histogram(channel.ravel(), bins=256, range=(0, 1))
    histSum = np.sum(hist)

    # Ensure the histogram has non-zero data
    if histSum == 0:
        print("Warning: Histogram sum is zero, possibly an empty or invalid image.")
        return 0

    pixelProbability = hist / histSum  # Normalize the histogram to get probabilities

    # Initialize variables
    q1 = np.zeros(256)
    mu1 = np.zeros(256)
    mu2 = np.zeros(256)
    variance = np.zeros(256)  # Array to store between-class variance

    q1[0] = pixelProbability[0]  # q1(1) = P(1)
    mu1[0] = 0  # mu1(0) = 0
    totalMean = np.dot(np.arange(256), pixelProbability)  # Total mean (needed for mu2)

    for t in range(1, 256):
        q1[t] = q1[t - 1] + pixelProbability[t]
        if q1[t] > 0:
            mu1[t] = (mu1[t - 1] * q1[t - 1] + t * pixelProbability[t]) / q1[t]
        if q1[t] < 1:
            mu2[t] = (totalMean - q1[t] * mu1[t]) / (1 - q1[t])
        if q1[t] > 0 and q1[t] < 1:
            variance[t] = q1[t] * (1 - q1[t]) * (mu1[t] - mu2[t]) ** 2

    # Find the threshold that maximizes between-class variance
    optimalThreshold = np.argmax(variance)
    print("Otsu calculation time: " + str(time.time() - startTime))
    return optimalThreshold

# Create histograms for each channel
def histogramDisplay(channel, threshold, colorName):
    print(f"Otsu's Threshold for {colorName} channel: {threshold}")
    # Plot histogram and threshold for each channel
    plt.figure(figsize=(6, 4))
    plt.hist(channel.ravel(), bins=256, range=(0, 1), color=colorName.lower(), alpha=0.7)
    plt.axvline(threshold / 255, color='black', linestyle='dashed', linewidth=1)
    plt.title(f"{colorName} Channel - Otsu's Threshold")
    plt.show()

# Function to set up the channels for otsu calculations
def channelSetup(image):
    # Check if the image is RGB (3D)
    if len(image.shape) == 3:
        if np.max(image) > 1:
            image = image/255.0
        # Separate the image into R, G, and B channels
        redChannel = image[:, :, 0]
        greenChannel = image[:, :, 1]
        blueChannel = image[:, :, 2]
        # Use the skimage built-in otsu method to compare values with those computed
        print("Built in otsu red channel result: " + str(threshold_otsu(redChannel)*255))
        print("Built in otsu green channel result: " + str(threshold_otsu(greenChannel)*255))
        print("Built in otsu blue channel result: " + str(threshold_otsu(blueChannel)*255))
        # Perform Otsu's method for each channel
        thresholds = []
        for channel, colorName in zip([redChannel, greenChannel, blueChannel], ['Red', 'Green', 'Blue']):
            threshold = otsuRecursive(channel)
            thresholds.append(threshold)
            histogramDisplay(channel, threshold, colorName)
        
        return thresholds
    
    else: #only works if image is single-channel grayscale, most grayscale images will have three equal channels
        threshold = otsuRecursive(image)
        histogramDisplay(image, threshold, 'Grayscale')

        return threshold

def binarizedDisplay(image, thresholds):
    if np.max(image) <= 1.0:
        image = (image * 255).astype(np.uint8)

    redChannel = image[:, :, 0]
    greenChannel = image[:, :, 1]
    blueChannel = image[:, :, 2]

    binRed = (redChannel > thresholds[0]).astype(np.uint8) * 255
    binGreen = (greenChannel > thresholds[1]).astype(np.uint8) * 255
    binBlue = (blueChannel > thresholds[2]).astype(np.uint8) * 255

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(binRed, cmap='gray')
    plt.imsave("redBinarized.png", binRed, cmap = 'gray')
    plt.title(f'Red Channel Binarized (Threshold: {thresholds[0]})')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(binGreen, cmap='gray')
    plt.imsave("greenBinarized.png", binGreen, cmap = 'gray')
    plt.title(f'Green Channel Binarized (Threshold: {thresholds[1]})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(binBlue, cmap='gray')
    plt.imsave("blueBinarized.png", binBlue, cmap = 'gray')
    plt.title(f'Blue Channel Binarized (Threshold: {thresholds[2]})')
    plt.axis('off')

    plt.show()

img = cv.imread(Path(sys.argv[1]), 1)
cv.imshow('Input image', img)
cv.waitKey(0)
optThresholds = channelSetup(img)
print("Calculated Thresholds [R, G, B]: " + str(optThresholds))
binarizedDisplay(img, optThresholds)



