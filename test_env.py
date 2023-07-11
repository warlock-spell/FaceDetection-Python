import cv2
import dlib
import face_recognition

print(cv2.__version__)
print(dlib.__version__)
print(face_recognition.__version__)


# pip install opencv-python
# pip install cmake
# download wheel file from face-recognition pakage
# pip install wheelfile



image_test = cv2.imread('image-library/face1.jpg')
cv2.imshow("Images", image_test)