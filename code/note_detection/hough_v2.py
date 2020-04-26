import numpy as np
import cv2
from skimage import color, util, filters, feature, img_as_float32, io
from matplotlib import pyplot as plt

im_size = (850, 1100)
threshold = im_size[0] * 0.5

# based off Sohum's staff detection
# needed to find radius parameter for Hough circle
# returns pixel height between staff lines


def circle_height(image):
    # modify image
    inverted_image = util.invert(image)
    horiz_sum = np.sum(inverted_image, axis=1)
    horiz_sum[horiz_sum < 600] = 0

    # maxes for staff lines
    staff_lines = feature.peak_local_max(horiz_sum).flatten()
    staff_lines = np.reshape(staff_lines, (-1, 5))
    staff_lines = np.sort(staff_lines, axis=0)

    # return just height
    sorted_indices = np.sort(staff_lines[1, :])
    height = sorted_indices[1] - sorted_indices[0]
    return height


# creates hough circles and returns notehead coordinates
def hough_circle(height):
    # read image
    img = cv2.imread('processed.png', cv2.IMREAD_COLOR)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur using 3 * 3 kernel
    gray_blurred = cv2.blur(gray, (3, 3))

    # edges
    edge_detected_image = cv2.Canny(gray_blurred, 75, 200)

    # I HAVE NO TESTED THIS WITH HEIGHT
    detected_circles = cv2.HoughCircles(edge_detected_image,
                                        cv2.HOUGH_GRADIENT, 2, 41, param1=100,
                                        param2=8, minRadius=0, maxRadius=height/2)

    return detected_circles

    # draw circles that are detected
    # if detected_circles is not None:

    #     # Convert the circle parameters a, b and r to integers.
    #     detected_circles = np.uint16(np.around(detected_circles))

    #     for pt in detected_circles[0, :]:
    #         a, b, r = pt[0], pt[1], pt[2]

    #         # Draw the circumference of the circle.
    #         cv2.circle(img, (a, b), r, (0, 255, 0), 2)

    #         # Draw a small circle (of radius 1) to show the center.
    #         cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
    #         cv2.imshow("Detected Circle", img)
    #         cv2.waitKey(0)

# combines both functions
# THIS IS NOT USING THE GET IMAGE STUFF YET NEEDS TO BE MODIFIED AYEEEEEEEEEEEEEEEEEEEEEEEEEEEE


def note_array(image):
    # rawImage = img_as_float32(io.imread('original.png', as_gray=False))
    height = circle_height(image)
    return hough_circle(height)
