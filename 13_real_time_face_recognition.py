import face_recognition
import cv2


# load image to detect
webcam_video_stream = cv2.VideoCapture(0) # 1 is for usb webcam, 0 for system webcam

rdj_image = face_recognition.load_image_file('face-recognition-library/sample/rdj.jpg')
rdj_face_encodings = face_recognition.face_encodings(rdj_image)[0] # sample image must contain one face only


eo_image = face_recognition.load_image_file('face-recognition-library/sample/eo.jpg')
eo_face_encodings = face_recognition.face_encodings(eo_image)[0] # sample image must contain one face only

# create an array to keep the encodings
known_face_encodings = [rdj_face_encodings, eo_face_encodings]

# create another array to hold the labels
known_face_names = ['Iron Man', 'Scarlet Witch']


# find and print total no of faces
# initialize empty array for face locations
all_face_locations = []
all_face_encodings = []
all_face_names = []


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
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=1, model='hog')

    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)
    all_face_names = []
    # print the no of faces in the array
    # print("There are {} face(s) in this image".format(len(all_face_locations)))

    # loop through face locations and encodings
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        # print location of each faces inside the loop
        # split the tuple
        top_pos, right_pos, bottom_pos, left_pos = current_face_location  # goes in clockwise direction
        top_pos = top_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        right_pos = right_pos * 4

        # compare faces and get the matches list
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)

        # initialize name string
        name_of_person = "Unknown Face"

        # use first match and get name from the respective index
        # if the match was found in the known_face_encodings, use the first one
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]

        # draw rectangle around the face
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        # write name below face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

    # show the image with rectangle and text
    cv2.imshow('Webcam Video', current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# once the loop breaks, release the camera resources and close all open windows
webcam_video_stream.release()
cv2.destroyAllWindows()


