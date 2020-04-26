import numpy as np
from midi_conversion import create_midi
from staff_detection import \
    process_image, \
    detect_staff_lines, \
    load_features, \
    find_feature_staffs, \
    find_pitches, \
    find_staff_distance, \
    construct_notes
from utility.image_operations import \
    load_image, \
    save_image, \
    show_image, \
    visualize_image, \
    visualize_staff_lines, \
    visualize_notes
from note_detection.hough_v2 import note_array
from preprocessing.staff_removal import staff_removal
import warnings
import argparse
from pathlib import Path

import sys

# sys.path.insert(
#     1, r"C:\Users\Ivan Zhao\Documents\GitHub\cs1430-final-project\code\note_detection")
# sys.path.insert(
#     2, r"C:\Users\Ivan Zhao\Documents\GitHub\cs1430-final-project\code\midi_conversion")
# sys.path.insert(
#     3, r"C:\Users\Ivan Zhao\Documents\GitHub\cs1430-final-project\code\preprocessing")


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def circles_to_features(circles):
    features = []
    circles = circles[0]
    for i in range(circles.shape[0]):
        x, y = circles[i, 0:2]
        features.append((x, y, 0.25, b'note'))

    return np.array(features)

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

    return parser.parse_args()


def main():
    args = command_line_args()

    # The current image to process
    I = process_image(args.image_path)

    staff_lines = detect_staff_lines(I)
    print("Number of staffs: " + str(staff_lines.shape))

    # A [# of features x 4] array holding all of the features for the image
    features = load_features(args.features_path)

    # Code Ivan added for feature detection using methods
    # Function to remove staff
    removed_staff_img = staff_removal(args.image_path)

    save_image("../results/processed.png", removed_staff_img)

    # Function to get Hough
    detected_circles = note_array(I)

    features = circles_to_features(detected_circles)

    matched_staffs = find_feature_staffs(features, staff_lines)
    print("Matched features to staffs.")

    pitches = find_pitches(features, staff_lines, matched_staffs)
    print("Matched features to pitches.")

    staff_dist = find_staff_distance(staff_lines)

    if (not(args.no_vis)):
        visualize_image(I, as_gray=True)
        visualize_staff_lines(I, staff_lines)
        visualize_notes(I, features, staff_lines,
                        matched_staffs, pitches, staff_dist)
        show_image()

    notes = construct_notes(features, staff_lines, matched_staffs, pitches)

    path = Path(args.image_path).stem
    create_midi(path, notes)

if __name__ == "__main__":
    main()
