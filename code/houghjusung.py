import numpy as np
import cv2
from skimage import color, util, filters, feature, img_as_float32, io
from matplotlib import pyplot as plt

im_size = (850, 1100)
threshold = im_size[0] * 0.5


def circle_height(image):
    inverted_image = util.invert(image)
    horiz_sum = np.sum(inverted_image, axis=1)
    horiz_sum[horiz_sum < 600] = 0
    sum = horiz_sum > 500
    print(np.sum(sum))
    staff_lines = feature.peak_local_max(horiz_sum).flatten()
    staff_lines = np.reshape(staff_lines, (-1, 5))
    staff_lines = np.sort(staff_lines, axis=0)
    return staff_lines


def hough_circle(height):
    # Read image.
    img = cv2.imread('../data/nice.png', cv2.IMREAD_COLOR)

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    edge_detected_image = cv2.Canny(gray_blurred, 75, 200)
    cv2.imshow('Edge', edge_detected_image)
    cv2.waitKey(0)

    detected_circles = cv2.HoughCircles(edge_detected_image,
                                        cv2.HOUGH_GRADIENT, 2, 41, param1=100,
                                        param2=8, minRadius=2, maxRadius=8)

    print(detected_circles)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", img)
            cv2.waitKey(0)

    print("yoo")

# rawImage = img_as_float32(io.imread('original.png', as_gray=False))
# staff_lines = circle_height(rawImage)
# sorted_indices = np.sort(staff_lines[1,:])
# height = sorted_indices[1] - sorted_indices[0]


hough_circle(0)

