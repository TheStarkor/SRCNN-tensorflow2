from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import numpy as np


def drop_resolution(x, scale):
    size = (x.shape[0], x.shape[1])
    small_size = (int(size[0] / scale), int(size[1] / scale))
    img = array_to_img(x)
    small_img = img.resize(small_size, 3)
    return img_to_array(small_img.resize(img.size, 3))


def train_data_generator(
    data_dir, mode, scale, target_size=(256, 256), batch_size=32, shuffle=True
):
    for imgs in ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    ).flow_from_directory(
        directory=data_dir,
        classes=[mode],
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
    ):
        x = np.array([drop_resolution(img, scale) for img in imgs])
        yield x / 255.0, imgs / 255.0


def test_data_generator(
    data_dir, mode, scale, target_size=(256, 256), batch_size=32, shuffle=True
):
    for imgs in ImageDataGenerator().flow_from_directory(
        directory=data_dir,
        classes=[mode],
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
    ):
        x = np.array([drop_resolution(img, scale) for img in imgs])
        yield x / 255.0, imgs / 255.0
