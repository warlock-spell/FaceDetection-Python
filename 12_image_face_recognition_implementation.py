import face_recognition
import cv2

# load sample images and extract the face encodings
# returns a list of 128 dimensional encodings
# one for each face in the image
# load image to detect
rdj_image = face_recognition.load_image_file('face-recognition-library/sample/rdj.jpg')
rdj_face_encodings = face_recognition.face_encodings(rdj_image)[0] # sample image must contain one face only


eo_image = face_recognition.load_image_file('face-recognition-library/sample/eo.jpg')
eo_face_encodings = face_recognition.face_encodings(eo_image)[0] # sample image must contain one face only

# create an array to keep the encodings
known_face_encodings = [rdj_face_encodings, eo_face_encodings]

# create another array to hold the labels
known_face_names = ['Iron Man', 'Scarlet Witch']


# load an unknown image to identify
image_to_recognize = cv2.imread('face-recognition-library/test-images/mix4.jpg')

# find all faces and face encodings in the unknown image
all_face_locations = face_recognition.face_locations(image_to_recognize, model='hog') # by default model will be hog
all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)
# print the no of faces in the array
print("There are {} face(s) in this image".format(len(all_face_locations)))

# loop through face locations and encodings
for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
    # print location of each faces inside the loop
    # split the tuple
    top_pos, right_pos, bottom_pos, left_pos = current_face_location # goes in clockwise direction

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
    cv2.rectangle(image_to_recognize, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
    # write name below face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_recognize, name_of_person, (left_pos,bottom_pos), font, 0.5, (255, 255, 255), 1)

    # show the image with rectangle and text
    cv2.imshow('Faces Identified', image_to_recognize)
    cv2.waitKey(10000)
