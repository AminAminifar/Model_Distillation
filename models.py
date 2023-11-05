from tensorflow import keras
from keras import layers
import tensorflow as tf


def import_teacher(model_num=0):
    if model_num == 0:
        # Create the teacher
        teacher = keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=[28, 28, 1]),


                # third hidden layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=800),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),

                tf.keras.layers.Dense(units=800),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),

                # output layer
                tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
            ],
            name="teacher",
        )

    if model_num == 1:
        # Create the teacher
        teacher = keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=[28, 28, 1]),

                # first hidden layer
                tf.keras.layers.Conv2D(filters=100, kernel_size=3, strides=1, padding="same"),  # , input_shape=(28, 28, 1)
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

                # second hidden layer
                tf.keras.layers.Conv2D(filters=100, kernel_size=3, strides=1, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

                # third hidden layer
                tf.keras.layers.Flatten(),

                # output layer
                tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
            ],
            name="teacher",
        )

    if model_num == 2:
        # Create the teacher
        teacher = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
                layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
                layers.Flatten(),
                layers.Dense(10),
            ],
            name="teacher",
        )

    if model_num == 3:
        # Create the teacher
        teacher = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
                layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
                layers.Flatten(),

                tf.keras.layers.Lambda(lambda x: x / 2),  # x / temperature

                # output layer
                tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
            ],
            name="teacher",
        )

    return teacher


def import_student(model_num=0):

    if model_num == 0:
        # Create the student
        student = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=300),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),

                tf.keras.layers.Dense(units=300),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),

                # layers.Flatten(),
                layers.Dense(10),
            ],
            name="student",
        )

    if model_num == 1:
        # Create the student
        student = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters=50, kernel_size=3, strides=1, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),


                tf.keras.layers.Conv2D(filters=50, kernel_size=3, strides=1, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

                tf.keras.layers.Flatten(),

                # layers.Flatten(),
                layers.Dense(10),
            ],
            name="student",
        )

    if model_num == 2:
    # Create the student
        student = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
                layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
                layers.Flatten(),
                layers.Dense(10),
            ],
            name="student",
        )

    if model_num == 3:
    # Create the student
        student = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
                layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
                layers.Flatten(),
                tf.keras.layers.Lambda(lambda x: x / 2),  # x / temperature
                layers.Dense(10, activation=tf.nn.softmax),
            ],
            name="student",
        )

    return student
