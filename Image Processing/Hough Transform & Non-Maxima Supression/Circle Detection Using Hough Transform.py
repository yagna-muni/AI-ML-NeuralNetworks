import cv2
import numpy as np
import math

def compute_edge_image(bgr_img):
    """ Compute the edge magnitude of an image using a pair of Sobel filters """

    sobel_v = np.array([[-1, -2, -1],   # Sobel filter for the vertical gradient. Note that the filter2D function computes a correlation
                        [ 0,  0, 0],    # instead of a convolution, so the filter is *not* rotated by 180 degrees.
                        [ 1,  2, 1]])
    sobel_h = sobel_v.T                 # The convolution filter for the horizontal gradient is simply the transpose of the previous one

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gradient_v = cv2.filter2D(gray_img, ddepth=cv2.CV_32F, kernel=sobel_v)
    gradient_h = cv2.filter2D(gray_img, ddepth=cv2.CV_32F, kernel=sobel_h)
    gradient_magni = np.sqrt(gradient_v**2 + gradient_h**2)
    gradient_orient = np.arctan2(gradient_v, gradient_h)

    near_max = np.percentile(gradient_magni, 99.5)      # Clip magnitude at percentile 99.5 to prevent outliers from determining the range of relevant magnitudes
    edge_img = np.clip(gradient_magni * 255.0 / near_max, 0.0, 255.0).astype(np.uint8)  # Normalize and convert magnitudes into grayscale image
    return edge_img, gradient_orient

# Function to compute the Hough transform for circles
def hough_transform_circles(input_space, num_row_bins, num_col_bins, num_radius_bins, min_radius, max_radius):
    # Get the input image dimensions and create an empty output space
    rows, cols = input_space.shape
    output_space = np.zeros((num_row_bins, num_col_bins, num_radius_bins), dtype=int)

    # Find the edge coordinates and orientations for the input image
    edge_coords = np.column_stack(np.nonzero(input_space >= 64))
    edge_orientations = gradient_orient[edge_coords[:, 0], edge_coords[:, 1]]

    # Looping through each edge coordinate and orientation.. row, col
    for row, col, angle in zip(edge_coords[:, 0], edge_coords[:, 1], edge_orientations):
        # Loop through each radius bins
        for radius_bin in range(num_radius_bins):
            # Get radius corresponding to the current radius bin
            radius = min_radius + (max_radius - min_radius) * radius_bin / (num_radius_bins - 1)

            # Compute circle center coordinates...
            row_center = row - radius * math.sin(angle)
            col_center = col - radius * math.cos(angle)

            # Compute the bin indices for the circle center..
            row_bin = int(row_center * num_row_bins / rows)
            col_bin = int(col_center * num_col_bins / cols)

            # Increment the vote count in the output space if the bin indices are there..
            if 0 <= row_bin < num_row_bins and 0 <= col_bin < num_col_bins:
                output_space[row_bin, col_bin, radius_bin] += 1
    return output_space

def find_hough_maxima(output_space, num_maxima, min_dist_alpha, min_dist_d):
    """ Find the given number of vote maxima in the output space with certain minimum distances in alpha and d between them """
    maxima = []
    output_copy = output_space.copy()
    height, width, radius = output_copy.shape

    for i in range(num_maxima):
        row, col, radius = np.unravel_index(np.argmax(output_copy),
                                    output_copy.shape)  # Get coordinates (alpha, d) of global maximum
        maxima.append((row, col, radius))
        output_copy[max(0, row - min_dist_alpha):min(height - 1, row + min_dist_alpha),
        # Set all cells within the minimum distances to -1
        max(0, col - min_dist_d):    min(width - 1,
                                         col + min_dist_d)] = -1.0  # so that no further maxima will be selected from this area
    return maxima

def draw_hough_circle(img, row, col, radius):
    row, col, radius = int(row), int(col), int(radius)
    cv2.circle(img, (col, row), radius, (255, 0, 0), 5)

# Please uncomment for required images.
#
# NUM_ROW_BINS = 500
# NUM_COL_BINS = 500
# NUM_RADIUS_BINS = 30
# MIN_RADIUS = 30
# MAX_RADIUS = 50
# NUM_MAXIMA = 3
# INPUT_IMAGE = 'Bicycles.png'

# NUM_ROW_BINS = 500
# NUM_COL_BINS = 500
# NUM_RADIUS_BINS = 30
# MIN_RADIUS = 20
# MAX_RADIUS = 50
# NUM_MAXIMA = 3
# INPUT_IMAGE = 'Desk.png'

# NUM_ROW_BINS = 500
# NUM_COL_BINS = 500
# NUM_RADIUS_BINS = 30
# MIN_RADIUS = 45
# MAX_RADIUS = 60
# NUM_MAXIMA = 5
# INPUT_IMAGE = 'Coins.png'

NUM_ROW_BINS = 580
NUM_COL_BINS = 820
NUM_RADIUS_BINS = 8
MIN_RADIUS = 32
MAX_RADIUS = 34
NUM_MAXIMA = 31
INPUT_IMAGE = 'pizza.png'

# Read the input image and display it
orig_img = cv2.imread(INPUT_IMAGE)
cv2.imshow('Input Image', orig_img)
cv2.waitKey(1)

# Compute the edge image and its gradient orientations
edge_img, gradient_orient = compute_edge_image(orig_img)

# Display the edge image
cv2.imshow('Edge Image', edge_img)
cv2.waitKey(1)

# Perform Hough transform for circles on the edge image
output_space = hough_transform_circles(edge_img, NUM_ROW_BINS, NUM_COL_BINS, NUM_RADIUS_BINS, MIN_RADIUS, MAX_RADIUS)

# Sum votes over the radius dimension to obtain a 2D projection..
output_projection = np.sum(output_space, axis=2)

# Scale the output projection to the range [0, 255] and convert it to uint8..
output_img = (output_projection * 255.0 / np.max(output_projection)).astype(np.uint8)

# Convert the output_img to a 3-channel BGR image
output_max_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

# Create a copy of the original image for drawing detected circles
output_circles_img = orig_img.copy()

# Find the Hough maxima (circle parameters) in the output space
circle_parameters = find_hough_maxima(output_space, NUM_MAXIMA, 60, 20)

# Draw the detected circles on the output_max_img and output_circles_img
for (row_bin, col_bin, radius_bin) in circle_parameters:
    row = row_bin * orig_img.shape[0] / NUM_ROW_BINS
    col = col_bin * orig_img.shape[1] / NUM_COL_BINS
    radius = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * radius_bin / NUM_RADIUS_BINS
    cv2.circle(output_max_img, (col_bin, row_bin), 10, (255, 0, 0), 2)
    draw_hough_circle(output_circles_img, row, col, radius)

# Output
cv2.imshow('Output Space with Maxima', output_max_img)
cv2.imshow('Input Image with Circles', output_circles_img)
cv2.waitKey(0)
cv2.destroyAllWindows()