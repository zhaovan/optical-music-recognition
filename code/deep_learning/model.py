import tensorflow as tf

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

# from preprocess import Dataset_Reader

import argparse
import pickle
import numpy as np


class NoteClassificationModel(tf.keras.Model):

    def __init__(self, number_classes):
        super(NoteClassificationModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.number_classes = number_classes

        self.batch_size = 160

        self.epochs = 10  # no clue
        self.epsilon = 0.001
        self.dropout_rate = 0.5

        # Follows implementation similar to deepscores model: conv, relu, maxpool, dropout
        self.architecture = [
            Conv2D(32, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            # Dropout(self.dropout_rate),
            # Set of layers 2
            Conv2D(64, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            # Dropout(self.dropout_rate),

            # Set of layers 3
            Conv2D(128, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            # Dropout(self.dropout_rate),

            # set of layers 4
            Conv2D(256, 3, 1, padding="same", activation="relu"),
            MaxPool2D(2, padding="same"),
            # Dropout(self.dropout_rate),

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
        print(image.shape)
        for layer in self.architecture:
            image = layer(image)

        return image

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)


def train(model, datasets):
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/" +
            "weights.e{epoch:02d}-" +
            "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True)
    ]

    model.fit(
        x=np.array(datasets.images, dtype=np.float32) / 255.,
        y=datasets.annotations,
        validation_data=(np.array(datasets.test_images,
                                  dtype=np.float32) / 255., datasets.test_annotations),
        epochs=model.epochs,
        batch_size=model.batch_size,  # none for right now
        callbacks=callback_list
    )


def test(model, datasets):
    model.evaluate(
        x=np.array(datasets.test_images,
                   dtype=np.float32) / 255.,
        y=datasets.test_annotations,
        verbose=1,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="OMR TIME")
    parser.add_argument(
        '--old_data',
        action='store_true',
        help='''Loads old preprocess''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')

    return parser.parse_args()


def main():
    data_reader = Dataset_Reader("./dataset")
    data_reader.one_hot = False
    if not ARGS.old_data:
        print("Reading new data")
        data_reader.read_images()
        print("Saving new data")
        # with open('dataset.pkl', 'wb') as output:
        #     pickle.dump(data_reader, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading old data")
        # with open('dataset.pkl', 'rb') as input:
        #     data_reader = pickle.load(input)

    print("Shape of images")
    print(data_reader.images.shape)
    print("Entry of images")
    print(data_reader.images[0])
    print("Shape of annotations")
    print(data_reader.annotations.shape)
    print("Entry of annotations")
    print(data_reader.annotations[0])

    model = NoteClassificationModel(data_reader.nr_classes)
    model(tf.keras.Input(
        shape=(data_reader.tile_size[0], data_reader.tile_size[1], 1)))
    model.summary()

    if ARGS.load_checkpoint is not None:
        print("Loading checkpoint")
        model.load_weights(ARGS.load_checkpoint)

    print("Compile model")
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        print("Start testing")
        test(model, data_reader)
    else:
        print("Start training")
        train(model, data_reader)


# ARGS = parse_args()
# main()
