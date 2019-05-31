import cv2
import pickle 
import os
import numpy as np
import imutils 
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers.convolutional import Conv2D , MaxPooling2D
from keras.layers.core import Flatten , Dense

#desired width and height in pixel 
def resize_to_fit(image, width, height):
	# grab the dimension of the image
	(h,w) = image.shape[:2]
	# if the width is greater than the height then resize along the width
	if (w > h) :
		image = imutils.resize(image, width=width)
	else :
		image = imutils.resize(image, height=height)

	#determine the padding values for the width and height to
	#obtain the target dimension
	padW = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0])/ 2.0)

	#pad the image then apply once more resizing to handle any
	#rounding issues
	image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
	image = cv2.resize(image, (width, height))
	#return the pre-processed image 
	return image


letter_image_folder = "extracted_letter_images"
model_filename = "captcha_model.hdf5"
model_labels_filename = "model_labels.dat"

data = []
labels = []

#loop over the input images
folder_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + letter_image_folder 
for letter_text_folder in os.listdir(folder_dir):
	if letter_text_folder == ".DS_Store" :
		continue 
	next_path = folder_dir + "/" + letter_text_folder 
	for letter_image in os.listdir(next_path):
		if letter_image == ".DS_Store" :
			continue 
		#load the image and convert it into grayscale
		image = cv2.imread(next_path+"/"+letter_image)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#resize the image so that it fits in a 20*20 pixel box
		image = resize_to_fit(image,20,20)
		#add a third channel dimension to the image to make keras happy
		image = np.expand_dims(image, axis=2)
		data.append(image)
		labels.append(letter_text_folder)

#scale the raw pixel intensities to range of [0,1] - this improves training
data = np.array(data , dtype="float") / 255.0
labels = np.array(labels)

#split the data into train and test sets 
x_train, x_test , y_train, y_test = train_test_split(data,labels, test_size=0.25, random_state = 0)
#convert the letter_text intp one-hot encoding that keras can work with it 
lb = LabelBinarizer().fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

#save the one-hot encoding of lettertext
#this will be required to decode what its prediction means
with open(model_labels_filename , "wb") as aditya:
	pickle.dump(lb,aditya)

#now build neural network
model = Sequential()

#first covolutional layer with max pooling 
model.add(Conv2D(20, (5,5), padding="same", input_shape = (20,20,1), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

#second convolutional layer with max pooling 
model.add(Conv2D(50, (5,5), padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

#hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

#output layer with 32 nodes
model.add(Dense(32, activation="softmax"))

#ask keras to build the TensorFlow model behind the scenes 
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#train the neural network 
model.fit(x_train,y_train, validation_data = (x_test,y_test), batch_size=32, epochs=10, verbose=1)

#save the train model to disk 
model.save(model_filename)

















