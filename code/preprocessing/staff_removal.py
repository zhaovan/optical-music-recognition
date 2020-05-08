import cv2
import numpy as np


def staff_removal(image_path, staff_dist):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.bitwise_not(img)
    th2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    vertical = th2
    rows, cols = vertical.shape

    verticalsize = int(staff_dist)
    verticalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    vertical = cv2.bitwise_not(vertical)
    smooth = vertical.copy()
    smooth = cv2.blur(smooth, (4, 4))
    (rows, cols) = np.where(img == 0)
    vertical[rows, cols] = smooth[rows, cols]

    return vertical
