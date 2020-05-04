import numpy as np
import os
import pandas as pa
import imageio
import re


class Dataset_Reader():

    def __init__(self, path, split=0.2, min_number=2, one_hot=True):

        self.path = path

        self.class_names = pa.read_csv(
            self.path+"/class_names.csv", header=None)

        self.split = split
        self.one_hot = one_hot
        self.nr_classes = 0

        config = open(self.path+"/config.txt", "r")
        config_str = config.read()
        self.tile_size = re.split('\)|,|\(', config_str)[4:6]

        self.tile_size[0] = int(self.tile_size[0])
        self.tile_size[1] = int(self.tile_size[1])

        self.images = []

        self.annotations = []

    def read_images(self):
        for folder in os.listdir(self.path):
            if os.path.isdir(self.path + "/"+folder) and max(self.class_names[1].isin([folder])):
                class_index = int(
                    self.class_names[self.class_names[1] == folder][0])
                self.load_class(folder, class_index)
                print(folder + " loaded")

        self.images = np.array(self.images)
        self.annotations = np.array(self.annotations)

        # extract testing data
        test_indices = []
        train_indices = []

        for curr_class in np.unique(self.annotations):
            class_index = np.where(self.annotations == curr_class)[0]
            np.random.shuffle(class_index)
            train_indices.append(
                class_index[0:int(len(class_index) * (1 - self.split))])
            test_indices.append(
                class_index[int(len(class_index) * (1 - self.split)):len(class_index)])

        train_indices = np.concatenate(train_indices)
        test_indices = np.concatenate(test_indices)

        self.test_images = self.images[test_indices]
        self.test_annotations = self.annotations[test_indices]

        self.images = self.images[train_indices]
        self.annotations = self.annotations[train_indices]

        # Shuffle the data
        perm = np.arange(self.images.shape[0])
        np.random.seed(0)
        np.random.shuffle(perm)
        self.images = self.images[perm]
        self.annotations = self.annotations[perm]

        # Reshape to fit Tensorflow
        self.images = np.expand_dims(self.images, -1)
        self.test_images = np.expand_dims(self.test_images, -1)

        self.nr_classes = max(self.test_annotations) + 1
        if self.one_hot:
            self.annotations = np.eye(self.nr_classes, dtype=np.uint8)[
                self.annotations]
            self.test_annotations = np.eye(self.nr_classes, dtype=np.uint8)[
                self.test_annotations]

    def load_class(self, folder, class_index):
        # move through images in folder
        i = 0
        for image in os.listdir(self.path + "/"+folder):
            if i >= 1:
                return
            self.load_image(folder, image, class_index)
            i += 1

    def load_image(self, folder, image, class_index):
        image = imageio.imread(self.path + "/" + folder + "/" + image)
        nr_y = image.shape[0] // self.tile_size[0]
        nr_x = image.shape[1] // self.tile_size[1]

        for x_i in range(0, nr_x):
            for y_i in range(0, nr_y):
                self.images.append(image[y_i*self.tile_size[0]:(
                    y_i+1)*self.tile_size[0], x_i*self.tile_size[1]:(x_i+1)*self.tile_size[1]])
                self.annotations.append(class_index)
