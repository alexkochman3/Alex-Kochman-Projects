import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def loadImages():
    img1 = cv2.imread(Path(sys.argv[1]), 1)
    img2 = cv2.imread(Path(sys.argv[2]), 1)
    return img1, img2


def showMatches(img1, img2, keypoints1, keypoints2, matches):
    """Visualize feature matches."""
    #Extract only the best matches from the knnMatch results
    bestMatches = [m for m, n in matches]  #Take only the first (best) match
    
    #Visualize the top 50 matches
    matchedImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, bestMatches[:50], None)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(matchedImg, cv2.COLOR_BGR2RGB))
    plt.title("Feature Matches")
    plt.axis("off")
    plt.show()

def stitchImages(img1, img2):
    #Detect SIFT features and compute descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    #Match features using FLANN
    indexes = dict(algorithm=1, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexes, searchParams)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    showMatches(img1, img2, keypoints1, keypoints2, matches)

    #Apply Lowe's ratio test with strict threshold
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    #Check if there are enough matches
    if len(goodMatches) > 10:
        #Extract matching points
        srcPts = np.float32([keypoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPts = np.float32([keypoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        #Compute the homography matrix with RANSAC
        homographyMatrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)

        #Warp img1 into the perspective of img2
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        #Calculate bounds of the stitched image
        corners1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
        corners1Warped = cv2.perspectiveTransform(corners1, homographyMatrix)
        corners2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
        allCorners = np.concatenate((corners1Warped, corners2), axis=0)

        #Compute new canvas size
        [x_min, y_min] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(allCorners.max(axis=0).ravel() + 0.5)
        translationDist = [-x_min, -y_min]

        #Translate the homography to fit the full panorama into the canvas
        translationMatrix = np.array([[1, 0, translationDist[0]], [0, 1, translationDist[1]], [0, 0, 1]])
        stitchedImg = cv2.warpPerspective(img1, translationMatrix @ homographyMatrix, (x_max - x_min, y_max - y_min))

        #Add img2 into the stitched canvas
        stitchedImg[translationDist[1] : translationDist[1] + height2, translationDist[0] : translationDist[0] + width2] = img2

        return stitchedImg
    else:
        print("Not enough matches found.")
        return None




def main():
    img1, img2 = loadImages()

    
    if img1 is None or img2 is None:
        print("Error loading images.")
        return

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original 1")
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Original 2")
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    stitchedImage = stitchImages(img1, img2)

    if stitchedImage is not None:
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(stitchedImage, cv2.COLOR_BGR2RGB))
        #cv2.imwrite("final33.png", stitchedImage)
        plt.title("Final Stitched Panorama")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
