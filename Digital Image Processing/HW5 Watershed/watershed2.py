import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from pathlib import Path

def load_image(path):
    # Load the image in grayscale
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def kmeans_segmentation(image, clusters=3):
    # Flatten the image and apply KMeans clustering
    pixel_values = image.reshape((-1, 1)).astype(np.float32)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(pixel_values)
    segmented_image = kmeans.labels_.reshape(image.shape)
    return segmented_image

def create_foreground_mask(segmented_image):
    # Create a binary mask by thresholding the darkest K-means class
    foreground_mask = np.where(segmented_image == segmented_image.min(), 0, 255).astype(np.uint8)

    # Define a kernel for morphological operations
    erosion_kernel = np.ones((3, 3), np.uint8)
    dilation_kernel = np.ones((2, 2), np.uint8)

    # Apply erosion and dilation to separate touching cells
    foreground_mask = cv2.erode(foreground_mask, erosion_kernel, iterations=2)
    foreground_mask = cv2.dilate(foreground_mask, dilation_kernel, iterations=2)
    
    
    return foreground_mask

def compute_distance_transform(foreground_mask):
    # Apply distance transform and create internal markers
    dist_transform = cv2.distanceTransform(foreground_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Create markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Increment markers to ensure background is not 0
    markers[foreground_mask == 0] = 1
    return markers

def apply_watershed(image, markers):
    # Convert the grayscale image to BGR format for adding boundaries
    boundary_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Apply the watershed algorithm
    markers = cv2.watershed(boundary_image, markers)
    
    # Color the boundaries in red
    boundary_image[markers == -1] = [0, 0, 255]
    
    return boundary_image, markers

def connected_components_and_area(markers):
    # Perform connected component labeling and calculate areas
    labeled_image = np.zeros_like(markers, dtype=np.uint8)
    labeled_image[markers > 1] = 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(labeled_image, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip the background
    return labels, areas

def create_size_based_mask(labels, areas, size_thresholds=(15, 50, 70)):
    # Color-code cells based on area: large=red, medium=green, small=blue
    size_based_mask = np.zeros((*labels.shape, 3), dtype=np.uint8)
    small_thresh, medium_thresh, large_thresh = size_thresholds

    for label, area in enumerate(areas, start=1):
        color = (255,0,0) if area > large_thresh else (0, 255, 0) if area > medium_thresh else (0, 0, 255)
        size_based_mask[labels == label] = color
    
    return size_based_mask

def plot_results(image, segmented_image, foreground_mask, markers, boundary_image, labels, areas, size_based_mask):
    # Display images and plots
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")

    plt.subplot(2, 4, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title("K-means Segmentation")

    plt.subplot(2, 4, 3)
    plt.imshow(foreground_mask, cmap='gray')
    plt.title("Foreground Mask")

    plt.subplot(2, 4, 4)
    plt.imshow(markers)
    plt.title("Markers")

    plt.subplot(2, 4, 5)
    plt.imshow(boundary_image)
    plt.title("Watershed Boundaries")

    plt.subplot(2, 4, 6)
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title("Final Segmentation")

    plt.subplot(2, 4, 7)
    plt.imshow(size_based_mask)
    plt.title("Color-coded Cell Sizes")

    plt.tight_layout()
    plt.show()

    # Plot area distribution
    plt.figure(figsize=(6, 4))
    plt.hist(areas[areas<150], bins=20, color='blue', edgecolor='black')
    plt.title("Distribution of Cell Areas")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    plt.show()

def watershed_segmentation_pipeline(image_path):
    # Load image
    image = load_image(image_path)
    if image is None:
        print("Error loading image:", image_path)
        return

    # Process each step
    segmented_image = kmeans_segmentation(image)
    foreground_mask = create_foreground_mask(segmented_image)
    markers = compute_distance_transform(foreground_mask)
    boundary_image, markers = apply_watershed(image, markers)
    labels, areas = connected_components_and_area(markers)
    size_based_mask = create_size_based_mask(labels, areas)

    # Display results
    plot_results(image, segmented_image, foreground_mask, markers, boundary_image, labels, areas, size_based_mask)
    print("Total number of cells:", len(areas))

    return len(areas), areas

# Path to the image
image_path = Path(sys.argv[1])

# Run the pipeline on the image
total_cells, cell_areas = watershed_segmentation_pipeline(image_path)

print(f"Total cells: {total_cells} \nCell areas: {cell_areas}")
