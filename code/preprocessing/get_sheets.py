import numpy as np
import skimage
import cv2
import glob
from scipy import ndimage

# Function that creates a gaussian kernel for two dimensions


def gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_sheets(file_path):
    images = [cv2.imread(image) for image in glob.glob(file_path + "/*.jpg")]

    if len(images) == 0:
        print("Should've received at least 1 image")
        return []

    images = [skimage.transform.resize(image, (584, 584)) for image in images]

    gaussian_kernel = gauss2D((5, 5), 2)

    images = [ndimage.convolve(image, gaussian_kernel) for image in images]

    return images
