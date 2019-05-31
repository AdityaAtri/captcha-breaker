from keras.models import load_model
from imutils import paths
import numpy as np
import imutils 
import cv2
import pickle
import os
from selenium import webdriver


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


model_filename = "captcha_model.hdf5" #trained model 
model_labels_filename = "model_labels.dat" # one-hot encoding
captcha_image_folder = "generated_captcha_images"

#load the model labels so we can translate model_prediction into actual letters
with open(model_labels_filename, "rb") as aditya:
	lb = pickle.load(aditya)

#load the trained neural network
model = load_model(model_filename)
#take the captcha for test from website
#grab some random captch to test against
executable_path = os.path.dirname(os.path.abspath(__file__)) + "/chromedriver"
#print(executable_path)
# loading the page , loading the captch and saving its screenshot
driver1 = webdriver.Chrome(executable_path)
driver1.get("file:///Users/adityaatri/Desktop/projects/captcha_breaker/main_page.html?")
button = driver1.find_element_by_id('loadpage')
button.click()
screenshot = driver1.save_screenshot('screenshot.png')
img = cv2.imread("screenshot.png")
crop_img = img[130:170, 40:180]
cv2.imwrite("test.png", crop_img)
test_image = cv2.imread("test.png")
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#add some extra padding around the image 
test_image = cv2.copyMakeBorder(test_image,20,20,20,20, cv2.BORDER_REPLICATE)
#convert it to pure black and white
threshold_image = cv2.threshold(test_image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#find contours 
contours = cv2.findContours(threshold_image.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Hack for compatibility with different OpenCV versions
contours = contours[0] if imutils.is_cv2() else contours[1]
letter_image_regions = []
for contour in contours :
	# get the rectangle of each letter captured
	(x, y, w, h) = cv2.boundingRect(contour)
	#compare the width and height of the contour
	if w / h > 3.0 :
	# it means two letter attached
		half_width = int(w/2)
		letter_image_regions.append((x,y,half_width,h))
		letter_image_regions.append((x+half_width,y,half_width,h))
	else :
		letter_image_regions.append((x,y,w,h))
print(len(letter_image_regions))
if len(letter_image_regions) == 4:
	#sort the dectected letter based on the x coordinates
	letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
	predictions = []
	for letter_bounding_box in letter_image_regions:
		x,y,w,h = letter_bounding_box
		letter_image = test_image[y :y + h , x :x + w]
		letter_image = resize_to_fit(letter_image, 20, 20)
		# Turn the single image into a 4d list of images to make Keras happy
		letter_image = np.expand_dims(letter_image, axis=2)
		letter_image = np.expand_dims(letter_image, axis=0)
	    # Ask the neural network to make a prediction
		prediction = model.predict(letter_image)
		letter = lb.inverse_transform(prediction)[0]
		predictions.append(letter)
	string_result = "".join(predictions)
	search_form = driver1.find_element_by_id('CaptchaEnter')
	search_form.send_keys(string_result)
	button = driver1.find_element_by_id('submitbutton')
	button.click()
	print("Detected Captch is :" + str(predictions))
	cv2.destroyAllWindows()

