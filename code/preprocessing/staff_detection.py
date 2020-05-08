import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from skimage import color, util, filters, feature

im_size = (850, 1100)
threshold = im_size[0] * .5

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

def load_features(path):
    return np.genfromtxt(
        path, 
        dtype="i8,i8,f8,S5",
        names=['x','y','length','type'],
        delimiter=',')

def find_feature_staffs(features, staffs):
    matched_staffs = np.zeros(features.shape[0])
    for f in range(features.shape[0]):
        _, y, _, _ = features[f]
        y = float(y)
        staff_dists = np.sum(np.absolute(staffs - y), axis=1)
        matched_staffs[f] = np.argmin(staff_dists)
    return matched_staffs

def find_staff_distance(staffs):
    avg_dist = 0
    for s in range(staffs.shape[0]):
        staff = staffs[s]
        dist = 0
        for l in range(staff.shape[0] - 1):
            dist += np.absolute(staff[l] - staff[l + 1])
        avg_dist += dist / float(staff.shape[0] - 1)
    avg_dist = (avg_dist / float(staffs.shape[0])) / 2
    return avg_dist

def find_pitches(features, staffs, matched_staffs):
    staff_dist = find_staff_distance(staffs)
    matched_pitches = np.zeros(features.shape[0])

    for f in range(features.shape[0]):
        staff = staffs[matched_staffs[f].astype(int)]
        highest_line = np.max(staff)
        _, y, _, _ = features[f]
        y = float(y)

        staff_line = -np.round((y - highest_line) / staff_dist)
        matched_pitches[f] = staff_line

    return matched_pitches

def construct_note(feature, pitch):
    _, _, length, type = feature
    return (type, float(length), float(pitch))

def construct_notes(features, staffs, matched_staffs, pitches):
    notes = []
    num_staffs = staffs.shape[0]

    num_notes = 0
    for i in range(num_staffs):
        notes_indices = np.where(matched_staffs == i)

        feature_x = features[notes_indices, 0][0]
        feature_x = [float(x) for x in feature_x]
        sorted_notes_indices = np.argsort(feature_x)

        these_pitches = (pitches[notes_indices])[sorted_notes_indices]
        these_features = (features[notes_indices])[sorted_notes_indices]

        these_notes = [construct_note(these_features[i], these_pitches[i]) for i in range(len(these_pitches))]

        notes[num_notes:num_notes + sorted_notes_indices.shape[0]] = these_notes
        num_notes += sorted_notes_indices.shape[0]
    return notes