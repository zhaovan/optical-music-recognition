import numpy as np
import cv2
from skimage import color, util, filters, feature, img_as_float32, io
from matplotlib import pyplot as plt
from matplotlib import patches

im_size = (850, 1100)
threshold = 500

# creates hough circles and returns notehead coordinates in x, y, r format
# height: height of each notehead
def hough_circle(height):
    # read image
    img = cv2.imread('../results/processed.png', cv2.IMREAD_COLOR)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur using 3 * 3 kernel
    gray_blurred = cv2.blur(gray, (3, 3))

    # edges
    edge_detected_image = cv2.Canny(gray, 75, 200)

    cv2.imshow("Detected Circle", edge_detected_image)
    cv2.waitKey(0)

    # returns all circles
    detected_circles = cv2.HoughCircles(edge_detected_image,
                                        cv2.HOUGH_GRADIENT, 1, 41, param1=100,
                                        param2=9, minRadius=height - 2, maxRadius=height + 4)

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

    return detected_circles

# takes bounding boxes, returns list of x, y, r for each notehead for original image
# img: original image
# bb: list of bounding boxes
# height: height of notes
# simg: staff removed image
def hough_circle_input(img, bb, height, simg):
    fig, ax = plt.subplots(1)
    ax.imshow(simg, cmap='gray_r')
    for i in range(bb.shape[0]):
        rect = patches.Rectangle((bb[i, 0], bb[i, 1]), bb[i, 2],
                                 bb[i, 3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


    for i in range(len(bb)):
        dim = bb[i]
        cutout = img[dim[1]: (dim[1] + dim[3]), dim[0]: (dim[0] + dim[2])]

        # blur using 3 * 3 kernel
        gray_blurred = cv2.blur(cutout, (3, 3))

        # edges
        edge_detected_image = cv2.Canny(cutout, 75, 200)

        # I HAVE NO TESTED THIS WITH HEIGHT
        detected_circles = cv2.HoughCircles(edge_detected_image,
                                            cv2.HOUGH_GRADIENT, 1, 41, param1=100,
                                            param2=9, minRadius=int(height) - 2, maxRadius=int(height) + 4)

        # Convert the circle parameters a, b and r to integers.
        if detected_circles is not None:

            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(cutout, (a, b), r, (255, 255, 230), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(cutout, (a, b), 1, (255, 255, 255), 3)
                cv2.imshow("Detected Circle", cutout)
                cv2.waitKey(0)

    return detected_circles