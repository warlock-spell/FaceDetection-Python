import face_recognition
import cv2

image_to_detect = cv2.imread('image-library/face1.jpg')



"""
find all face locations using face_locations() function
model can be 'cnn' or 'hog'
number_of_times_to_upsample =1 ; higher and detect more faces
"""
all_face_locations = face_recognition.face_locations(image_to_detect, model='cnn') # by default model will be hog

# print the no of faces in the array
print("There are {} face(s) in this image".format(len(all_face_locations)))

# cv2.imshow("test_title", image_to_detect)
# cv2.waitKey(0)