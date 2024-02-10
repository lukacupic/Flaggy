import os

IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 64
EPOCHS = 10

BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])