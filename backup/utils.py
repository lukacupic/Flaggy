from tensorflow import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

def make_pairs(images, labels):
	pairImages = []
	pairLabels = []

	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

	for i in range(len(images)):
		currentImage = images[i]
		label = labels[i]
		
		posIdx = np.random.choice(idx[label])
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