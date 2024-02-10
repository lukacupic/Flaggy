from distutils.command.config import config
from tensorflow import keras
import keras.backend as K
import matplotlib.pyplot as plt
from imutils import build_montages
import numpy as np
import cv2
import glob
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import re
import config

def make_pairs(images, labels):
	pairImages = []
	pairLabels = []

	idx = {}

	unique_labels = np.unique(labels)
	for l in unique_labels:
		idx[l] = [np.where(labels == l)]

	for i in range(len(images)):
		currentImage = images[i]
		label = labels[i]
		
		posIdxs = np.where(labels == label)[0]
		posIdx = np.random.choice(posIdxs)
		posImage = images[posIdx]
		
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])

		negIdxs = np.where(labels != label)[0]
		negIdx = np.random.choice(negIdxs)
		negImage = images[negIdx]
		
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])

	return (np.array(pairImages), np.array(pairLabels))

def visualize_pairs(X, Y):
	(pairTrain, labelTrain) = make_pairs(X, Y)

	images = []

	for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
		imageA = pairTrain[i][0]
		imageB = pairTrain[i][1]
		label = labelTrain[i]
		
		output = np.zeros((36, 60), dtype="uint8")
		pair = np.hstack([imageA, imageB])
		output[4:32, 0:56] = pair

		text = "neg" if label[0] == 0 else "pos"
		color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)

		vis = cv2.merge([output] * 3)
		vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
		cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
			color, 2)
			
		images.append(vis)

	montage = build_montages(images, (96, 51), (7, 7))[0]

	cv2.imshow("Siamese Image Pairs", montage)
	cv2.waitKey(0)
    
def euclidean_distance(vectors):
	(featsA, featsB) = vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

def preprocess_image(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
	return image

def load_flag(path):
	im = Image.open(path).convert('RGB').resize((256, 256))
	im = np.array(im) / 255.0
	return im

def load_flags():
	origin_folder = '/media/luka/89683ab3-c0a6-4837-be8c-03edb0c1d685/gen/'
	imagelist = natural_sort(glob.glob(origin_folder + '*.jpg'))

	images = []
	for imagepath in imagelist:
		im = load_flag(imagepath)
		images.append(im)

	x = np.asarray(images).astype(np.float32)
	y = np.loadtxt(os.path.join(origin_folder, "labels.txt"), dtype=object)

	trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.20, random_state=config.RANDOM_STATE)
	return (trainX, trainY), (testX, testY)


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def compare(imageA, imageB, preds):
	prob = preds[0][0]

	fig = plt.figure("Pair", figsize=(4, 2))
	plt.suptitle("Similarity: {:.2f}".format(prob))

	fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, interpolation='nearest')
	plt.axis("off")

	fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, interpolation='nearest')
	plt.axis("off")

	plt.show()