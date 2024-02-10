import os

RANDOM_STATE = 46

IMG_SHAPE = (256, 256, 3)
BATCH_SIZE = 8
EPOCHS = 200

BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])