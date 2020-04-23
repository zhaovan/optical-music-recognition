from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage import io, filters, feature, img_as_float32
import matplotlib.pyplot as plt
import csv
import sys
import argparse
import numpy as np
import scipy.io as scio
import matplotlib
import cv2 as cv
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

import matplotlib
matplotlib.use("TkAgg")


def main():
    #image1_file = "../data/easy.png"

    # image1_color = img_as_float32(io.imread(image1_file))
    # image1 = rgb2gray(image1_color)
    # scale_factor = 0.5
    # image1 = np.float32(rescale(image1, scale_factor))

    # implot = plt.imshow(image1_color)

    # # put a blue dot at (10, 20)
    # plt.scatter([10], [20])

    # # put a red dot, size 40, at 2 locations:
    # plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)

    # plt.show()

    img = cv.imread('../data/easy.png', 0)
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=100)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv.imshow('detected circles', cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Load picture, convert to grayscale and detect edges
    # image_rgb = img_as_float32(io.imread(image1_file))
    # image_gray = color.rgb2gray(image_rgb)
    # edges = canny(image_gray, sigma=2.0,
    #               low_threshold=0.55, high_threshold=0.8)

    # # Perform a Hough Transform
    # # The accuracy corresponds to the bin size of a major axis.
    # # The value is chosen in order to get a single high accumulator.
    # # The threshold eliminates low accumulators
    # result = hough_ellipse(edges, accuracy=20, threshold=250,
    #                        min_size=0, max_size=100)
    # result.sort(order='accumulator')

    # # Estimated parameters for the ellipse
    # best = list(result[-1])
    # yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    # orientation = best[5]

    # # Draw the ellipse on the original image
    # cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    # image_rgb[cy, cx] = (0, 0, 255)
    # # Draw the edge (white) and the resulting ellipse (red)
    # edges = color.gray2rgb(img_as_ubyte(edges))
    # edges[cy, cx] = (250, 0, 0)

    # fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
    #                                 sharex=True, sharey=True)

    # ax1.set_title('Original picture')
    # ax1.imshow(image_rgb)

    # ax2.set_title('Edge (white) and result (red)')
    # ax2.imshow(edges)

    # plt.show()
    return


if __name__ == '__main__':
    main()
