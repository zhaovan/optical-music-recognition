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
from skimage import data, color, img_as_ubyte, util
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from staff_detection import detect_staff_lines, find_staff_distance, process_image
from sorin_cheat_staff_removal import staff_removal

import matplotlib
matplotlib.use("TkAgg")


def main():

    # img = util.invert(process_image('../midi_conversion/data/fuzzy-wuzzy.png'))
    # staffs = detect_staff_lines(img)
    # dist = int(find_staff_distance(staffs))

    # img = cv.imread('../midi_conversion/data/fuzzy-wuzzy.png', 0)
    img = staff_removal('../midi_conversion/data/fuzzy-wuzzy.png')
    img = resize_percentage(img, 200, 200)
    # blob_detection(img)

    circle_detection(img)
    return


def resize_percentage(img, width_percent, height_percent):
    width = int(img.shape[1] * width_percent / 100)
    height = int(img.shape[0] * height_percent / 100)
    dim = (width, height)
    # resize image
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img


def blob_detection(im):
    # Read image
    # im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array(
        []), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv.imshow("Keypoints", im_with_keypoints)
    cv.waitKey(0)


def circle_detection(img):
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 40,
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


if __name__ == '__main__':
    main()
