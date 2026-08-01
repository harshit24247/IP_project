import cv2
import os
import time
import numpy as np

save_dir = r'C:\Users\ASUS\eye_movement_detection\video_dataset'
os.makedirs(save_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 9 gaze directions with labels (x, y, z)
directions = {
    'center':       ( 0.00,  0.00, -1.00),
    'left':         (-0.50,  0.00, -0.87),
    'right':        ( 0.50,  0.00, -0.87),
    'up':           ( 0.00,  0.50, -0.87),
    'down':         ( 0.00, -0.50, -0.87),
    'top_left':     (-0.40,  0.40, -0.82),
    'top_right':    ( 0.40,  0.40, -0.82),
    'bottom_left':  (-0.40, -0.40, -0.82),
    'bottom_right': ( 0.40, -0.40, -0.82),
}

cap = cv2.VideoCapture(0)
all_frames, all_labels = [], []

for direction, gaze_label in directions.items():
    print(f"\n👀 Look {direction.upper()}")
    print("Recording in 3 seconds...")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("🔴 Recording!")
    frames_collected = 0

    start_time = time.time()
    while time.time() - start_time < 15:  # 5 seconds per direction
        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (fx, fy, fw, fh) in faces:
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)

            for (ex, ey, ew, eh) in eyes[:1]:  # only first eye
                eye_img = face_gray[ey:ey+eh, ex:ex+ew]
                eye_resized = cv2.resize(eye_img, (60, 36))
                all_frames.append(eye_resized)
                all_labels.append(gaze_label)
                frames_collected += 1

        # Show preview
        cv2.putText(frame, f'Look {direction.upper()}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Frames: {frames_collected}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Recording', frame)
        cv2.waitKey(1)

    print(f"✅ {direction}: {frames_collected} frames collected")

cap.release()
cv2.destroyAllWindows()

# Save dataset
all_frames = np.array(all_frames)
all_labels = np.array(all_labels)
np.save(f'{save_dir}/frames.npy', all_frames)
np.save(f'{save_dir}/labels.npy', all_labels)
print(f"\n✅ Dataset saved!")
print(f"Total frames: {len(all_frames)}")
print(f"Shape: {all_frames.shape}")