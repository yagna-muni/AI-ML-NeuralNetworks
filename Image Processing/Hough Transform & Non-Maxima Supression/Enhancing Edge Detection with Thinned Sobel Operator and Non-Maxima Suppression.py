import cv2
import numpy as np

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

def non_maximum_suppression(edge_img, gradient_orient):
    padded_img = np.pad(edge_img, ((1, 1), (1, 1)), 'constant')
    suppressed_img = np.zeros_like(edge_img, dtype=np.uint8)

    for i in range(1, padded_img.shape[0] - 1):
        for j in range(1, padded_img.shape[1] - 1):
            # Compute the angle in the range [0, pi)
            angle = gradient_orient[i - 1, j - 1] % np.pi

            # Initialize n1,n2 = 0
            n1, n2 = 0, 0

            # Determining neighboring pixels based on the angle
            if angle < np.pi / 8 or angle >= 7 * np.pi / 8:
                n1, n2 = padded_img[i, j - 1], padded_img[i, j + 1]
            elif angle >= np.pi / 8 and angle < 3 * np.pi / 8:
                n1, n2 = padded_img[i - 1, j], padded_img[i + 1, j + 1]
            elif angle >= 3 * np.pi / 8 and angle < 5 * np.pi / 8:
                n1, n2 = padded_img[i - 1, j], padded_img[i + 1, j]
            else:
                n1, n2 = padded_img[i - 1, j + 1], padded_img[i + 1, j - 1]

            # Check if the current pixel has a higher gradient magnitude than its neighbors
            if edge_img[i - 1, j - 1] >= n1 and edge_img[i - 1, j - 1] >= n2:
                # If so, keep the pixel in the suppressed image
                suppressed_img[i - 1, j - 1] = edge_img[i - 1, j - 1]

    return suppressed_img

def thin_sobel(input_image):
    edge_img, gradient_orient = compute_edge_image(input_image)
    thinned_edge_img = non_maximum_suppression(edge_img, gradient_orient)
    return thinned_edge_img

# Load the input image
input_image = cv2.imread('pizza.png', cv2.IMREAD_COLOR)

# Compute the original edge image
edge_img, _ = compute_edge_image(input_image)

# Compute the thinned edge image
thinned_edge_img = thin_sobel(input_image)

# Display the original edge image and the thinned edge image
cv2.imshow('Original Edge Image', edge_img)
cv2.imshow('Thinned Edge Image', thinned_edge_img)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()