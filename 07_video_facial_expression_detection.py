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


webcam_video_stream = cv2.VideoCapture('videos-library/Man-angry.mp4')

# face expression model initialization
face_exp_model = model_from_json(open("dataset/dataset/facial_expression_model_structure.json", "r").read())
# load weights into the model
face_exp_model.load_weights('dataset/dataset/facial_expression_model_weights.h5')
# list of emotion labels
dataset_emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

all_face_locations = []

# create while loop, to loop through each frame of the video
# get current frame
while True:
    # get single frame of video as image
    # get current frame
    ret, current_frame = webcam_video_stream.read()

    # optional step
    # resize the frame to a quarter of size so that the computer can process it faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    # finding total number of faces
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2,
                                                         model='hog')

    # loop through faces
    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        # getting right coordinates
        top_pos = top_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        right_pos = right_pos * 4
        print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index + 1, top_pos, right_pos, bottom_pos,
                                                                             left_pos))

        # slicing the current face from main image
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

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
        cv2.putText(current_frame, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

        # draw rectangle around each face location
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

    # showing the current face with rectangular drawn
    cv2.imshow("Webcam Video", current_frame)

    # wait for a key press to break the while loop
    # press 'q' on keyboard to break the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# once the loop breaks, release the camera resources and close all open windows
webcam_video_stream.release()
cv2.destroyAllWindows()
