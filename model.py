from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D


def srcnn():
    model = Sequential()

    model.add(
        Conv2D(
            filters=64,
            kernel_size=9,
            activation="relu",
            padding="same",
            input_shape=(None, None, 3),
        )
    )
    model.add(Conv2D(filters=32, kernel_size=5, activation="relu", padding="same"))
    model.add(Conv2D(filters=3, kernel_size=5, padding="same"))

    return model
