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
    AGE_GENDER_MODEL_MEAN_VALUE = (78.4263377603, 87.7689143744, 114.895847746)

    # create blob of current face slice
    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUE,
                                                    swapRB=False)

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
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
    # display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, gender + " " + age + " " + "years", (left_pos, bottom_pos), font, 0.5, (0, 255, 0), 1)

    # showing the current face with rectangular drawn
cv2.imshow("image", image_to_detect)
cv2.waitKey(5000)











