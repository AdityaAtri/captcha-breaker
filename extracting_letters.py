import os
import os.path
import cv2
import glob
import imutils


input_folder = "generated_captcha_images"
output_folder = "extracted_letter_images"
counts = {}
curr_directory = os.path.dirname(os.path.abspath(__file__))  + "/" + input_folder + "/"
for image in os.listdir(curr_directory):
    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = image.split(".")[0]
    # Load the image and convert it to grayscale
    image = cv2.imread(curr_directory + image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Add some extra padding around the image
    gray_image = cv2.copyMakeBorder(gray_image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    # threshold the image (convert it to pure black and white)
    threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # find the contours (continuous blobs of pixels) the image
    contours= cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]
    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))  #first half 
            letter_image_regions.append((x + half_width, y, half_width, h)) #second half 
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 4:
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_box, letter_text in zip(letter_image_regions, filename):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_box
        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray_image[y - 2:y + h + 2, x - 2:x + w + 2]
        # Get the folder to save the image in
        save_path = os.path.dirname(os.path.abspath(__file__)) + "/" + output_folder + "/" + letter_text
        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        path = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(path, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1







