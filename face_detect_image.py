import cv2

# Load the cascade
cascading_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

# Reading input image
image = cv2.imread('Group_girls.jpg')

# converting to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detecting faces
faces = cascading_face.detectMultiScale(gray, 1.1, 4)

# draw rectangle around a face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

# display the output
cv2.imshow('Faces_found', image)
cv2.waitKey()
