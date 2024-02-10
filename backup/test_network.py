import config
import utils
import os
from tensorflow import keras
from keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input directory of testing images")
    args = vars(ap.parse_args())

    test_image_paths = list(list_images(args["input"]))
    np.random.seed(42)
    pairs = np.random.choice(test_image_paths, size=(10, 2))
    
    model = load_model(config.MODEL_PATH)

    for (i, (pathA, pathB)) in enumerate(pairs):
        imageA = cv2.imread(pathA, 0)
        imageB = cv2.imread(pathB, 0)
        
        origA = imageA.copy()
        origB = imageB.copy()
        
        imageA = np.expand_dims(imageA, axis=-1)
        imageB = np.expand_dims(imageB, axis=-1)
        
        imageA = np.expand_dims(imageA, axis=0)
        imageB = np.expand_dims(imageB, axis=0)
        
        imageA = imageA / 255.0
        imageB = imageB / 255.0
        
        preds = model.predict([imageA, imageB])
        prob = preds[0][0]

        fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
        plt.suptitle("Similarity: {:.2f}".format(prob))
        
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(origA, cmap=plt.cm.gray)
        plt.axis("off")
        
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(origB, cmap=plt.cm.gray)
        plt.axis("off")
        
        plt.show()


if __name__ == "__main__":
	main()