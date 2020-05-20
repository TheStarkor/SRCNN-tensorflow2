from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import prepare_data as pd
import numpy


def model():
    SRCNN = Sequential()
    SRCNN.add(
        Conv2D(
            128,
            (9, 9),
            kernel_initializer="glorot_uniform",
            activation="relu",
            padding="valid",
            use_bias=True,
            input_shape=(32, 32, 1),
        )
    )
    SRCNN.add(
        Conv2D(
            64,
            (3, 3),
            kernel_initializer="glorot_uniform",
            activation="relu",
            padding="same",
            use_bias=True,
        )
    )
    SRCNN.add(
        Conv2D(
            1,
            (5, 5),
            kernel_initializer="glorot_uniform",
            activation="linear",
            padding="valid",
            use_bias=True,
        )
    )
    adam = Adam(lr=0.0003)
    SRCNN.compile(
        optimizer=adam, loss="mean_squared_error", metrics=["mean_squared_error"]
    )
    return SRCNN


def train():
    srcnn = model()
    print(srcnn.summary())
    data, label = pd.read_training_data("./train.h5")
    val_data, val_label = pd.read_training_data("./test.h5")

    checkpoint = ModelCheckpoint(
        "SRCNN_check.h5",
        monitor="val_loss",
        verbose=2,
        save_best_only=True,
        save_weights_only=False,
        model="min",
    )
    callbacks_list = [checkpoint]

    srcnn.fit(
        data,
        label,
        batch_size=128,
        validation_data=(val_data, val_label),
        callbacks=callbacks_list,
        shuffle=True,
        epochs=200,
        verbose=2,
    )


def predict():
    srcnn_model = predict_model()
    srcnn_model.load_weights("SRCNN_check.h5")
    IMG_NAME = "./train/im_012.png"
    INPUT_NAME = "input.jpg"
    OUTPUT_NAME = "predict.jpg"

    import cv2

    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = cv2.resize(
        img[:, :, 0], (int(shape[1] / 2), int(shape[0] / 2)), cv2.INTER_CUBIC
    )
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(INPUT_NAME, img)

    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.0
    pre = srcnn_model.predict(Y, batch_size=1) * 255.0
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6:-6, 6:-6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6:-6, 6:-6, 0]
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6:-6, 6:-6, 0]
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6:-6, 6:-6, 0]

    print("bicubic:")
    print(cv2.PSNR(im1, im2))
    print("SRCNN:")
    print(cv2.PSNR(im1, im3))


if __name__ == "__main__":
    train()
    predict()
