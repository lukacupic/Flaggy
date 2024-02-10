import config
import utils
import os
from tensorflow import keras
from keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

def main():
    np.random.seed(config.RANDOM_STATE)

    model = load_model(config.MODEL_PATH)

    # path = '/media/luka/89683ab3-c0a6-4837-be8c-03edb0c1d685/Projects/Programming/Flaggy/2020/Dataset/flags/test/'
    path = '/media/luka/89683ab3-c0a6-4837-be8c-03edb0c1d685/gen/'
    test_image_paths = list(list_images(path))

    pairs = np.random.choice(test_image_paths, size=(25, 2))
    for (i, (pathA, pathB)) in enumerate(pairs):
        imageA = utils.load_flag(pathA)
        imageB = utils.load_flag(pathB)

        origA = imageA.copy()
        origB = imageB.copy()

        imageA = np.expand_dims(imageA, axis=0)
        imageB = np.expand_dims(imageB, axis=0)

        preds = model.predict([imageA, imageB])
        utils.compare(origA, origB, preds)


if __name__ == "__main__":
	main()