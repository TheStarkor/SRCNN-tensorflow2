from tensorflow.keras.callbacks import ModelCheckpoint

from data_generator import train_data_generator, test_data_generator
from utils import psnr
from model import srcnn

DATA_DIR = "../src/"
N_TRAIN_DATA = 800
N_TEST_DATA = 100
BATCH_SIZE = 16
FILE_PATH = "./models/srcnn_div2k.hdf5"

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

model.fit_generator(
    train_data_generator,
    validation_data=(test_x, test_y),
    steps_per_epoch=N_TRAIN_DATA // BATCH_SIZE,
    epochs=200,
    callbacks=[save_checkpoint],
)
