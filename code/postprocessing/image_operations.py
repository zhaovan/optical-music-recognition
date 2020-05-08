import os
import cv2
import numpy as np
import argparse
from skimage import io, img_as_ubyte, img_as_float32, color, util
from scipy import misc
import matplotlib
#matplotlib.use('macosx')
from matplotlib import pyplot as plt
import matplotlib.lines as lines

def visualize_staff_lines(im, staff_lines):
    for i in range(staff_lines.shape[0]):
        c = np.random.rand(3,) / 2 + .25
        for j in range(staff_lines.shape[1]):
            plt.axhline(y=staff_lines[i, j],color=c)

def visualize_notes(im, features, staffs, matched_staffs, pitches, staff_dist):
    for i in range(features.shape[0]):
        feature = features[i]
        x, y, length, type = feature
        x, y, length = float(x), float(y), float(length)
        highest_line = np.max(staffs[matched_staffs[i].astype(int)])
        staff_line = pitches[i]

        if type == b'note':
            plt.plot(x, highest_line - staff_line * staff_dist, color='green', marker='o', markersize=length * 10)
        elif type == b'rest':
            plt.plot(x, highest_line - staff_line * 5, color='blue', marker='s', markersize=length * 20)

def load_image(path, as_gray):
    if (as_gray):
        return img_as_float32(io.imread(path, as_gray=True))
    else:
        return img_as_float32(io.imread(path))

def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))

def visualize_image(im, as_gray):
    if (as_gray):
        plt.imshow(im, cmap='gray_r')
    else:
        plt.imshow(im)

def show_image():
    plt.show()