import cv2
import time
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

output_folder = "images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

face_count = 0

last_saved_time = 0
delay = 5


while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    current_time = time.time()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        if (current_time - last_saved_time > delay):
            face = frame[y:y + h, x:x + w]

            face_filename = os.path.join(output_folder, f"face{face_count}.jpg")
            cv2.imwrite(face_filename, face)
            face_count += 1
            last_saved_time = current_time
    
    if len(faces) != 0:
        cv2.putText(frame, 'Face Detected' , (10,60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0) , 2)

    cv2.imshow('Face Detect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
