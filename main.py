from tensorflow.keras.callbacks import ModelCheckpoint
import argparse

from data_generator import train_data_generator, test_data_generator
from utils import psnr
from model import srcnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N_TRAIN_DATA", type=int)
    parser.add_argument("N_TEST_DATA", type=int)
    parser.add_argument("BATCH_SIZE", type=int)
    parser.add_argument("EPOCHS", type=int)
    args = parser.parse_args()

    DATA_DIR = "../src/"
    FILE_PATH = "./models/srcnn_div2k.hdf5"
    TRAIN_PATH = "DIV2K_train_HR"
    TEST_PATH = "DIV2K_valid_HR"

    N_TRAIN_DATA = args.N_TRAIN_DATA
    N_TEST_DATA = args.N_TEST_DATA
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS

    train_data_generator = train_data_generator(
        DATA_DIR, "DIV2K_train_HR", scale=4.0, batch_size=BATCH_SIZE
    )

    test_x, test_y = next(
        test_data_generator(
            DATA_DIR, "DIV2K_valid_HR", scale=4.0, batch_size=N_TEST_DATA, shuffle=False
        )
    )

    model = srcnn()

    model.summary()

    save_checkpoint = ModelCheckpoint(
        FILE_PATH,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=[psnr])

    model.fit(
        train_data_generator,
        validation_data=(test_x, test_y),
        steps_per_epoch=N_TRAIN_DATA // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[save_checkpoint],
    )
