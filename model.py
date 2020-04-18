import tensorflow as tf

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense


class NoteClassificationModel(tf.keras.Model):

    def __init__(self, number_classes):
        super(NoteClassificationModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.number_classes = number_classes

        self.batch_size = 100

        self.epochs = 50  # no clue
        self.epsilon = 0.001

        # Bunch of convolution that I'm too lazy to implement rn
        self.architecture = [

        ]

    def call(self, image):
