import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from pathlib import Path

def loadimage(path):
    #Load the image in grayscale
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def kmeanssegmentation(image, clusters=3):
    #Flatten the image and apply KMeans clustering
    pixelvalues = image.reshape((-1, 1)).astype(np.float32)
    kmeans = KMeans(nclusters=clusters, random_state=42)
    kmeans.fit(pixelvalues)
    segmentedimage = kmeans.labels_.reshape(image.shape)
    return segmentedimage

def createforegroundmask(segmentedimage):
    #Create a binary mask by thresholding the darkest K-means class
    foregroundmask = np.where(segmentedimage == segmentedimage.min(), 0, 255).astype(np.uint8)

    #Define a kernel for morphological operations
    erosionkernel = np.ones((3, 3), np.uint8)
    dilationkernel = np.ones((2, 2), np.uint8)

    #Apply erosion and dilation to separate touching cells
    foregroundmask = cv2.erode(foregroundmask, erosionkernel, iterations=2)
    foregroundmask = cv2.dilate(foregroundmask, dilationkernel, iterations=2)
    
    return foregroundmask

def computedistancetransform(foregroundmask):
    #Apply distance transform
    disttransform = cv2.distanceTransform(foregroundmask, cv2.DIST_L2, 5)
    _, surefg = cv2.threshold(disttransform, 0.1 * disttransform.max(), 255, 0)
    surefg = np.uint8(surefg)

    #Define sure background by dilating the foreground mask
    surebg = cv2.dilate(foregroundmask, np.ones((3, 3), np.uint8), iterations=3)
    
    #Create markers for watershed
    _, markers = cv2.connectedComponents(surefg)
    

    return disttransform, surefg, surebg, markers

def applywatershed(image, markers, unknownRegion):
    #Convert grayscale image to BGR for visualization
    boundaryimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #Adjust markers for watershed
    markers = markers + 1  #Increment all markers so background is 1
    markers[unknownRegion == 255] = 0  #Unknown region (typically set to 0)

    #Apply the watershed algorithm
    markers = cv2.watershed(boundaryimage, markers)

    #Color each unique marker region
    boundaryimage[markers == -1] = [0, 0, 255]  #Red boundaries
    uniquemarkers = np.unique(markers)
    for marker in uniquemarkers:
        if marker > 1:  #Skip background and boundary markers
            boundaryimage[markers == marker] = [255,255,255]

    return boundaryimage, markers

def createunknownregion(surebg, surefg):
    unknownregion = cv2.subtract(surebg, surefg)
    return unknownregion

def plotresults(imagepath):
    #Load image
    image = loadimage(imagepath)
    if image is None:
        print("Error loading image:", imagepath)
        return

    #Process each step
    grayimage = loadimage(imagepath)
    _, binaryimage = cv2.threshold(grayimage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fgmask = createforegroundmask(binaryimage)
    disttransform, surefg, surebg, markers = computedistancetransform(fgmask)
    
    unknownRegion = createunknownregion(fgmask, surefg)
    boundaryimage, watershedmarkers = applywatershed(grayimage, markers, unknownRegion)
    
    #Plot each stage in a 3x3 grid
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB))
    plt.title("1. Raw image")

    plt.subplot(3, 3, 2)
    plt.imshow(grayimage, cmap='gray')
    plt.title("2. Gray scale image")

    plt.subplot(3, 3, 3)
    plt.imshow(binaryimage, cmap='gray')
    plt.title("3. Thresholded image (binary mask)")

    plt.subplot(3, 3, 4)
    cv2.imwrite("surebg.png", surebg)
    plt.imshow(surebg, cmap='gray')
    plt.title("4. Sure bg (binary mask dilated)")

    plt.subplot(3, 3, 5)
    cv2.imwrite("distTransform.png", disttransform)
    plt.imshow(disttransform, cmap='gray')
    plt.title("5. Distance transform")

    plt.subplot(3, 3, 6)
    cv2.imwrite("surefg.png", surefg)
    plt.imshow(surefg, cmap='gray')
    plt.title("6. Sure fg")

    plt.subplot(3, 3, 7)
    cv2.imwrite("unknownRegion.png", unknownRegion)
    plt.imshow(unknownRegion, cmap='gray')
    plt.title("7. Unknown region")

    plt.subplot(3, 3, 8)
    cv2.imwrite("markers.png", markers)
    plt.imshow(markers, cmap='gray')
    plt.title("8. Markers (connected components)")

    plt.subplot(3, 3, 9)
    cv2.imwrite("boundaryImg.png", boundaryimage)
    plt.imshow(boundaryimage)
    plt.title("9. Watershed of markers")

    plt.tight_layout()
    plt.show()

#Path to the image
imagepath = Path(sys.argv[1])

#Run the pipeline and plot results
plotresults(imagepath)
