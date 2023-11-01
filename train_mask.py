# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"D:\MY PROJECTS\Face_Mask_Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("loading...")

data = []  # to store the images as arrays, because deep learning can only work with arrays.
labels = [] # to store labels of images (with_mask or without_mask)

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224)) # convert all images to one size.
    	image = img_to_array(image) # convert image to an array.
    	image = preprocess_input(image) 

    	data.append(image) 
    	labels.append(category)

lb = LabelBinarizer() # assigns either 0 or 1, with or without mask. 
labels = lb.fit_transform(labels)
labels = to_categorical(labels) # converting the class labels from integers to binary vectors. 
# binary vectors where each element represents a class, and the only element with a value of 1 is the element that corresponds to the class of the data point.

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# This helps to generate many images from one image by changing some of it's properties.
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2 is a light weighted CNN because it takes less parameters, which is suitable for our project
# It achieves high accuracy on a variety of image classification and object detection benchmarks.
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# We have to freeze the layers in the base model, so that they are not updated during training process.
for layer in baseModel.layers:
	layer.trainable = False

#compiled with the Adam optimizer, binary cross-entropy loss (suitable for binary classification),
# and accuracy as the evaluation metric.
print("compiling...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("training...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

print("evaluating...")

# make predictions on the testing data (testX) using the predict method. 
# The predictions are stored in predIdxs.
predIdxs = model.predict(testX, batch_size=BS)

#argmax used to find predicted class index for each sample.
# either 0 or 1 with the highest predicted probability.
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

print("saving...")
model.save("mask_detector.model", save_format="h5")

N = EPOCHS # EPOCHS is basically how many times we need to completely pass through 
# the entire training dataset.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")