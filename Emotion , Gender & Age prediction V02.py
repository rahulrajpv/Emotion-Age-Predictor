import cv2
from deepface import DeepFace

# Load the face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread(r"C:\Users\rahul\OneDrive\Pictures\sample.jpg")

# Resize the image to 50% of its original size
img = cv2.resize(img, (0,0), fx=0.75, fy=0.75)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image using OpenCV's face detector
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Loop through each detected face
for (x, y, w, h) in faces:
    # Extract the face from the image
    face = img[y:y+h, x:x+w]

    # Analyze the face using DeepFace to obtain facial attributes
    predictions = DeepFace.analyze(face, actions=['age','gender','emotion'])

    # Extract the age, race, and emotion from the predictions dictionary
    age = predictions[0]['age']
    gender = predictions[0]['dominant_gender']
    emotion = predictions[0]['dominant_emotion']

    # Draw a rectangle around the detected face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add text above the rectangle showing the predicted age
    cv2.putText(img, f"Age: {age}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Add text above the rectangle showing the predicted gender
    cv2.putText(img, f"Gender: {gender}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Add text above the rectangle showing the predicted emotion
    cv2.putText(img, f"Emotion: {emotion}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
   
# Display the image
cv2.imshow('output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
