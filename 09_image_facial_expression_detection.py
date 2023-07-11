import numpy as np
import cv2
import face_recognition
from keras.preprocessing import image
from keras.models import model_from_json

# below 5 lines of code were written from stack overflow to handle an error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



# load image to detect
image_to_detect = cv2.imread('image-library/baby-mom.jpg')

# face expression model initialization
face_exp_model = model_from_json(open("dataset/dataset/facial_expression_model_structure.json", "r").read())
# load weights into the model
face_exp_model.load_weights('dataset/dataset/facial_expression_model_weights.h5')
# list of emotion labels
dataset_emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


# find and print total no of faces
all_face_locations = face_recognition.face_locations(image_to_detect, model='hog') # by default model will be hog
# print the no of faces in the array
print("There are {} face(s) in this image".format(len(all_face_locations)))

# loop through faces
for index, current_face_location in enumerate (all_face_locations): # index, tuple in enumerate(array)
    # print location of each faces inside the loop
    # split the tuple
    top_pos, right_pos, bottom_pos, left_pos = current_face_location # goes in clockwise direction
    print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))
    # extracting faces from image
    # slice image array by positions inside the loop
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]

    # preprocessing the input image
    # converting it to an image like as the data in the dataset (kaggle dataset)
    # convert to greyscale
    current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
    # RESIZE TO 48X48 px size
    current_face_image = cv2.resize(current_face_image, (48, 48))
    # convert the PIL image into a 3d numpy array
    img_pixels = image.img_to_array(current_face_image)
    # expand the shape of an array into single row multiple columns
    img_pixels = np.expand_dims(img_pixels, axis=0)
    # pixels are in range of [0,2550], normalize all pixels in scale of [0,1]
    img_pixels /= 255

    # do the predictions and display the results
    # get the prediction values for all 7 expressions
    exp_predictions = face_exp_model.predict(img_pixels)
    # find max indexed prediction value (0 till 7)
    max_index = np.argmax(exp_predictions[0])
    # get corresponding label from emotions label
    emotion_label = dataset_emotions_label[max_index]

    # display the name as a text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

    # draw rectangle around each face location
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

cv2.imshow("image", image_to_detect)
cv2.waitKey(3000)



