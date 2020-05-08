import numpy as np
import pandas as pa
import warnings
import argparse
from pathlib import Path
import cv2
import tensorflow as tf
import sys
import skimage
import matplotlib
from matplotlib import patches
from matplotlib import pyplot as plt
from skimage import io, img_as_ubyte, img_as_float32, color, util

# Imports from other files as needed
from postprocessing.midi_conversion import create_midi
from preprocessing.staff_detection import \
    process_image, \
    detect_staff_lines, \
    load_features, \
    find_feature_staffs, \
    find_pitches, \
    find_staff_distance, \
    construct_notes
from postprocessing.image_operations import \
    load_image, \
    save_image, \
    show_image, \
    visualize_image, \
    visualize_staff_lines, \
    visualize_notes
from preprocessing.staff_removal import staff_removal
from note_detection.hough_circles import hough_circle, hough_circle_input
from note_detection.contour import make_bounding_boxes
from deep_learning.model import NoteClassificationModel

import matplotlib

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def dl_classification(model, img, class_names, bounding_boxes):
    classified_list = []
    new_img = 1 - img
    for i in range(len(bounding_boxes)):
        x, y, w, h = bounding_boxes[i]
        center_x = int(x + 0.5 * w)
        center_y = int(y + 0.5 * h)
        
        resized_shape = (220, 120)
        block_img = new_img[(center_y - 110):(center_y +
                                              110), (center_x - 60):(center_x + 60)]
        if not (block_img.shape == resized_shape):
            continue

        boxed_image = tf.Variable(block_img, dtype=tf.float32)

        reshaped_img = tf.reshape(
            boxed_image, [-1, resized_shape[0], resized_shape[1], 1])

        layer = model.call(reshaped_img)

        classified_list = np.append(classified_list, np.argmax(layer))

    return classified_list


def circles_to_features(circles):
    features = []
    circles = circles[0]
    for i in range(circles.shape[0]):
        x, y = circles[i, 0:2]
        features.append((x, y, 0.25, b'note'))

    return np.array(features)


def create_features(classified_elements, class_names, bounding_boxes):
    feature_list = []
    print(classified_elements)

    for i in range(len(classified_elements)):
        x, y, w, h = bounding_boxes[i]
        avg_x = (x + w) // 2
        avg_y = (y+h) // 2
        class_index = classified_elements[i]
        class_name = str((class_names[1])[class_index])
        feature_list.append((avg_x, avg_y, 0.25, class_name))

    return np.array(feature_list)


def command_line_args():
    parser = argparse.ArgumentParser(
        description='A program that creates a MIDI file from an image and extracted musical features!')

    parser.add_argument("--image-path",
                        default='../data/fuzzy-wuzzy.png',
                        type=str,
                        help="This is the path to your image!")
    parser.add_argument("--features-path",
                        default='../data/fuzzy_wuzzy_features.csv',
                        type=str,
                        help="This is the path to your image's features!")
    parser.add_argument("--no-vis",
                        action="store_true",
                        help="This is a variable representing whether to visualize results or not!")

    parser.add_argument("--feature-creator",
                        default="hough_circle",
                        type=str,
                        help="Use hough_circle or cnn for feature matching")

    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')

    return parser.parse_args()


def main():
    args = command_line_args()

    # The current image to process
    sheet_img = process_image(args.image_path)
    staff_lines = detect_staff_lines(sheet_img)

    print("Number of staffs: " + str(staff_lines.shape))

    # A [# of features x 4] array holding all of the features for the image
    truth_features = load_features(args.features_path)

    # PREPROCESSING
    staff_dist = find_staff_distance(staff_lines)
    removed_staff_img = staff_removal(args.image_path, staff_dist)
    save_image("../results/processed.png", removed_staff_img)

    # Hough Detection
    detected_circles = hough_circle(int(staff_dist))
    features = circles_to_features(detected_circles)

    # Bounding Boxes
    print("Finding Bounding Boxes")
    bounding_boxes = make_bounding_boxes(removed_staff_img)
    
    # hough_circle_input(removed_staff_img, bounding_boxes, staff_dist, sheet_img)

    print("Bounding Boxes have been Found")


    # Uncomment this if you want to look at your bounding boxes

    # fig, ax = plt.subplots(1)
    # ax.imshow(sheet_img, cmap='gray_r')
    # for i in range(bounding_boxes.shape[0]):
    #     rect = patches.Rectangle((bounding_boxes[i, 0], bounding_boxes[i, 1]), bounding_boxes[i, 2],
    #                              bounding_boxes[i, 3], linewidth=1, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)
    # plt.show()

    # DL Model Classification
    
    class_names = pa.read_csv(
        "./deep_learning/dataset/class_names.csv", header=None)

    num_classes = len(class_names)

    model = NoteClassificationModel(num_classes)
    model(tf.keras.Input(
        shape=(220, 120, 1)))
    model.load_weights(args.load_checkpoint)



    print("DL Classification")
    if (args.feature_creator == "cnn"):
        classified_list = dl_classification(model, sheet_img, class_names, bounding_boxes)
        features = create_features(classified_list, class_names, bounding_boxes)


    # Feature Matching
    matched_staffs = find_feature_staffs(features, staff_lines)
    print("Matched features to staffs.")

    pitches = find_pitches(features, staff_lines, matched_staffs)
    print("Matched features to pitches.")

    if (not(args.no_vis)):
        visualize_image(sheet_img, as_gray=True)
        visualize_staff_lines(sheet_img, staff_lines)
        visualize_notes(sheet_img, features, staff_lines,
                        matched_staffs, pitches, staff_dist)
        show_image()

    notes = construct_notes(features, staff_lines, matched_staffs, pitches)

    path = Path(args.image_path).stem
    create_midi(path, notes)


if __name__ == "__main__":
    main()
