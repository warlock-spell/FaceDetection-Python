import face_recognition
import cv2


# load image to detect
webcam_video_stream = cv2.VideoCapture('videos-library/Man-angry.mp4') # 1 is for usb webcam, 0 for system webcam


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
        print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))

        # slicing the current face from main image to blur
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
        current_face_image = cv2.GaussianBlur(current_face_image, (99,99), 30)

        # paste the blurred face into the actual frame
        current_frame[top_pos:bottom_pos, left_pos:right_pos] = current_face_image

        # draw rectangle around each face location
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)

    # showing the current face with rectangular drawn
    cv2.imshow("Webcam Video", current_frame)

    # wait for a key press to break the while loop
    # press 'q' on keyboard to break the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# once the loop breaks, release the camera resources and close all open windows
webcam_video_stream.release()
cv2.destroyAllWindows()


