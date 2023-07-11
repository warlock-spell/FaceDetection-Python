# downloading the pretrained model
# model created and trained by Gil Levi and Tal Hassner using the Adience dataset
# files to download (available in dataset folder)
# 1. age classification caffe face_recognition_models
# 2. age classification prototxt file
# 3. gender classification caffe face_recognition_models
# 4. gender classification model prototxt files_required
# 5. the mean image

# caffe is a deep learning framework
# prototxt is a configuration file to tell caffe how you want the network trained
# after training the model, we will get the trained model in a file with extension .caffemodel

# the AGE_GENDER_MODEL_MEAN_VALUES calculated by using the numpy.mean()
# step 1 - convert mean.binaryproto to blob
# step 2 - numpy.mean(blob.channels, blob.height, blob.width)
# result = ( 78.4263377603, 87.7689143744, 114.895847746 )



import face_recognition
import cv2


# load image to detect
webcam_video_stream = cv2.VideoCapture(0) # 1 is for usb webcam, 0 for system webcam


# find and print total no of faces
# initialize empty array for face locations
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
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='hog')


    # loop through faces
    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        # getting right coordinates
        top_pos = top_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        right_pos = right_pos * 4
        # print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))

        # slicing the current face from main image
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

        AGE_GENDER_MODEL_MEAN_VALUE = (78.4263377603, 87.7689143744, 114.895847746 )

        # create blob of current face slice
        current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUE, swapRB=False)

        # predicitng gender
        # declare gender labels, protext and caffemodel file paths
        gender_label_list = ['Male', 'Female']
        gender_protext = 'dataset/dataset/gender_deploy.prototxt'
        gender_caffemodel = 'dataset/dataset/gender_net.caffemodel'

        # create model from files and provide blob as input
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_cov_net.setInput(current_face_image_blob)

        # to run model
        gender_predictions = gender_cov_net.forward()
        gender = gender_label_list[gender_predictions[0].argmax()]


        # predicting age
        # declaring the labels
        age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20),' '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        age_protext = 'dataset/dataset/age_deploy.prototxt'
        age_caffemodel = 'dataset/dataset/age_net.caffemodel'

        # create model from files and provide blob as input
        age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        age_cov_net.setInput(current_face_image_blob)

        # to run model
        age_predictions = age_cov_net.forward()
        age = age_label_list[age_predictions[0].argmax()]

        # draw rectangle around face
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
        # display the name as text in the image
        font = cv2 .FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, gender+ " "+age+ " "+"years", (left_pos, bottom_pos), font, 0.5, (0, 255, 0), 1)

    # showing the current face with rectangular drawn
    cv2.imshow("Webcam Video", current_frame)

    # wait for a key press to break the while loop
    # press 'q' on keyboard to break the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# once the loop breaks, release the camera resources and close all open windows
webcam_video_stream.release()
cv2.destroyAllWindows()


