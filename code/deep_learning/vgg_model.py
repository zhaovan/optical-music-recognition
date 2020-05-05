import tensorflow as tf

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D

#from preprocess import Dataset_Reader

import argparse
import pickle
import numpy as np


class NoteClassificationModel_Vgg4(tf.keras.Model):

    def __init__(self, number_classes):
        super(NoteClassificationModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.number_classes = number_classes

        self.batch_size = 50

        self.epochs = 10  # no clue
        # self.epsilon = 0.001
        self.dropout_rate = 0.5

        # Follows implementation similar to deepscores model: conv, relu, maxpool, dropout
        self.architecture = [
        Conv2D(32, 3, 1),
        Conv2D(32, 3, 1),
        MaxPooling2D(),

        Conv2D(64, 3, 1),
        Conv2D(64, 3, 1),
        MaxPooling2D(),

        Conv2D(128, 3, 1),
        Conv2D(128, 3, 1),
        Conv2D(128, 3, 1),
        MaxPooling2D(),

        Conv2D(256, 3, 1),
        Conv2D(256, 3, 1),
        Conv2D(256, 3, 1),
        MaxPooling2D(),

        Conv2D(512, 3, 1),
        Conv2D(512, 3, 1),
        Conv2D(512, 3, 1),
        AveragePooling2D(),

        Flatten()  # Flatten
        #Dropout(0.5))
        Dense(units=self.number_classes, activation='softmax', name='output_class'))
        ]

    def call(self, image):
        for layer in self.architecture:
            image = layer(image)

        return image

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
