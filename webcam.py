import cv2
import torch
import numpy as np
from model import GazeNet

# Load trained model
model = GazeNet()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# Load eye detector
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_eye(eye_img):
    eye_resized = cv2.resize(eye_img, (60, 36))
    tensor = torch.tensor(eye_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor

def gaze_to_arrow(cx, cy, gaze_vec, length=60):
    # Project 3D gaze to 2D arrow
    dx = int(gaze_vec[0] * length)
    dy = int(gaze_vec[1] * length)
    return (cx, cy), (cx + dx, cy + dy)

cap = cv2.VideoCapture(0)
print("🎥 Webcam started — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        face_gray = gray[fy:fy+fh, fx:fx+fw]
        face_col  = frame[fy:fy+fh, fx:fx+fw]

        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)

        for (ex, ey, ew, eh) in eyes:
            eye_img = face_gray[ey:ey+eh, ex:ex+ew]
            tensor  = preprocess_eye(eye_img)

            with torch.no_grad():
                gaze = model(tensor).squeeze().numpy()

            # Draw eye box
            cv2.rectangle(face_col, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Draw gaze arrow
            cx, cy   = ex + ew//2, ey + eh//2
            start, end = gaze_to_arrow(cx, cy, gaze)
            cv2.arrowedLine(face_col, start, end, (0, 0, 255), 2)

            # Show gaze values
            cv2.putText(frame, f"Gaze: ({gaze[0]:.2f}, {gaze[1]:.2f}, {gaze[2]:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Gaze Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()