import numpy as np
import skimage
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import img_as_float32
from scipy import ndimage
import numpy as np
from skimage import color, util, filters, feature

im_size = (850, 1100)
threshold = im_size[0] * 0.5


def load_image(path, as_gray):
    if (as_gray):
        return img_as_float32(io.imread(path, as_gray=True))
    else:
        return img_as_float32(io.imread(path))


def process_image(path):
    image = load_image(path, as_gray=True)
    image = util.invert(image)
    np.resize(image, im_size)
    return image


def detect_staff_lines(image):
    horiz_sum = np.sum(image, axis=1)
    horiz_sum[horiz_sum < threshold] = 0
    staff_lines = feature.peak_local_max(horiz_sum).flatten()
    staff_lines = np.reshape(staff_lines, (-1, 5))
    staff_lines = np.sort(staff_lines, axis=0)
    return staff_lines


def my_dsl(image):
    horiz_sum = np.sum(image, axis=1)
    horiz_sum[horiz_sum < threshold] = 0
    staff_lines = feature.peak_local_max(horiz_sum).flatten()
    staff_lines = np.reshape(staff_lines, (-1, 5))
    staff_lines = np.sort(staff_lines, axis=0)
    return staff_lines


def staff_removal(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('img', img)
    cv2.waitKey(10000)

    img = cv2.bitwise_not(img)
    th2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # cv2.imshow("th2", th2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    horizontal = th2
    vertical = th2
    rows, cols = horizontal.shape

    # inverse the image, so that lines are black for masking
    horizontal_inv = cv2.bitwise_not(horizontal)
    # perform bitwise_and to mask the lines with provided mask
    masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
    # reverse the image back to normal
    masked_img_inv = cv2.bitwise_not(masked_img)
    # cv2.imshow("masked img", masked_img_inv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    horizontalsize = int(cols / 30)
    horizontalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    # cv2.imshow("horizontal", horizontal)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    verticalsize = int(rows / 90)
    verticalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    # cv2.imshow("vertical", vertical)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    vertical = cv2.bitwise_not(vertical)
    # cv2.imshow("vertical_bitwise_not", vertical)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # step1
    edges = cv2.adaptiveThreshold(
        vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # step2
    kernel = np.ones((2, 2), dtype="uint8")
    dilated = cv2.dilate(edges, kernel)
    # cv2.imshow("dilated", dilated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # step3
    smooth = vertical.copy()

    # step 4
    smooth = cv2.blur(smooth, (4, 4))
    # cv2.imshow("smooth", smooth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # step 5
    (rows, cols) = np.where(img == 0)
    vertical[rows, cols] = smooth[rows, cols]

    cv2.imshow('vertical', vertical)
    cv2.waitKey(10000)

    return vertical


def main():
    path = '../../data/fuzzed.png'
    image = img_as_float32(io.imread(path, as_gray=True))

    newimage = staff_removal(path)

    print("yes")
    plt.imshow(newimage)


if __name__ == "__main__":
    main()
