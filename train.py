
from siamese_network import build_siamese_model
import utils
import config
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda
import numpy as np
import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

def main():
    np.random.seed(config.RANDOM_STATE)

    # (trainX, trainY), (testX, testY) = mnist.load_data()
    (trainX, trainY), (testX, testY) = utils.load_flags()
    
    (pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
    (pairTest, labelTest) = utils.make_pairs(testX, testY)

    imgA = Input(shape=config.IMG_SHAPE)
    imgB = Input(shape=config.IMG_SHAPE)
    featureExtractor = build_siamese_model(config.IMG_SHAPE)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)

    distance = Lambda(utils.euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(
        [pairTrain[:, 0], pairTrain[:, 1]],
        labelTrain[:],
        validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS
    )

    model.save(config.MODEL_PATH)

    imageA = trainX[0][np.newaxis, ...]
    imageB = trainX[1][np.newaxis, ...]

    preds = model.predict([imageA, imageB])
    utils.compare(trainX[0], trainX[1], preds)

if __name__ == "__main__":
	main()