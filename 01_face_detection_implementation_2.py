import face_recognition
import cv2


# load image to detect
image_to_detect = cv2.imread('image-library/baby-mom.jpg')

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
    # show each sliced face inside the loop
    cv2.imshow("Face No: " +str(index+1), current_face_image) # for n images we will need n unique titles
    cv2.waitKey(3000)



