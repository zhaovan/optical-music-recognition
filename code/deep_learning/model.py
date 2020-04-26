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
        self.dropout_rate = 0.2

        # Follows implementation similar to deepscores model: conv, relu, maxpool, dropout
        self.architecture = [
            Conv2D(32, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            Dropout(self.dropout_rate),
            # Set of layers 2
            Conv2D(64, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            Dropout(self.dropout_rate),

            # Set of layers 3
            Conv2D(128, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            Dropout(self.dropout_rate),

            # set of layers 4
            Conv2D(256, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            Dropout(self.dropout_rate),

            # Set of layers
            Conv2D(64, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            Dropout(self.dropout_rate),

            # Flattens
            Flatten(),

            # Two dense layers
            Dense(1024, activation="relu"),
            Dense(self.number_classes, activation="softmax")
        ]

    def call(self, image):

        for layer in self.architecture:
            image = layer(image)

        return image
