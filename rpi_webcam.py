import cv2
import torch
import numpy as np
from collections import deque
from model import GazeNet
import mediapipe as mp
import os
import time

# ── Load CNN only model ───────────────────────────────────────
model = GazeNet()
model.load_state_dict(torch.load('best_model.pth',
                                  map_location='cpu',
                                  weights_only=False))
model.eval()
print("✅ CNN model loaded: best_model.pth")

# ── MediaPipe ─────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ── Smoothing buffer (replaces LSTM) ─────────────────────────
gaze_buffer = deque(maxlen=8)

# ── Calibration values ────────────────────────────────────────
CENTER_X     = -0.2802
CENTER_Y     = +0.1560
DEAD_ZONE    =  0.07
LEFT_THRESH  = +0.13
RIGHT_THRESH =  0.05
UP_THRESH    =  0.05
DOWN_THRESH  = -0.16

# ── Camera — lower resolution for RPi speed ──────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)   # lower for RPi
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS,          15)
print("🎥 Started — press Q to quit")
print("📏 Optimal distance: 40-70cm\n")

# ── FPS counter ───────────────────────────────────────────────
fps_buffer  = deque(maxlen=30)
prev_time   = time.time()

# ── Helpers ───────────────────────────────────────────────────
def get_eye_box(landmarks, indices, w, h, padding=15):
    pts = np.array([
        (int(landmarks[i].x * w),
         int(landmarks[i].y * h))
        for i in indices
    ])
    x1 = max(0, pts[:, 0].min() - padding)
    y1 = max(0, pts[:, 1].min() - padding)
    x2 = min(w, pts[:, 0].max() + padding)
    y2 = min(h, pts[:, 1].max() + padding)
    return x1, y1, x2, y2

def get_iris_center(landmarks, indices, w, h):
    pts = np.array([
        (int(landmarks[i].x * w),
         int(landmarks[i].y * h))
        for i in indices
    ])
    return pts.mean(axis=0).astype(int)

# ── Main loop ─────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps_buffer.append(1.0 / max(curr_time - prev_time, 0.001))
    prev_time = curr_time
    fps       = np.mean(fps_buffer)

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results         = face_mesh.process(rgb)
    gaze_text       = "No face detected"
    direction       = ""
    direction_color = (0, 255, 0)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # Distance
        face_width_px = abs(
            int(lm[454].x * w) - int(lm[234].x * w)
        )
        distance_cm = int((14 * 600) / max(face_width_px, 1))

        if distance_cm < 30 or distance_cm > 80:
            dist_color = (0, 0, 255)
            dist_text  = f"Dist: {distance_cm}cm {'CLOSE' if distance_cm < 30 else 'FAR'}"
        else:
            dist_color = (0, 255, 0)
            dist_text  = f"Dist: {distance_cm}cm OK"

        # Eye boxes
        l_x1, l_y1, l_x2, l_y2 = get_eye_box(lm,
            [33, 7, 163, 144, 145, 153, 154, 155, 133], w, h)
        r_x1, r_y1, r_x2, r_y2 = get_eye_box(lm,
            [362, 382, 381, 380, 374, 373, 390, 249, 263], w, h)

        cv2.rectangle(frame, (l_x1, l_y1), (l_x2, l_y2), (0, 255, 0), 1)
        cv2.rectangle(frame, (r_x1, r_y1), (r_x2, r_y2), (0, 255, 0), 1)

        # Iris dots
        try:
            l_iris = get_iris_center(lm, LEFT_IRIS,  w, h)
            r_iris = get_iris_center(lm, RIGHT_IRIS, w, h)
            cv2.circle(frame, tuple(l_iris), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(r_iris), 3, (0, 0, 255), -1)
        except:
            pass

        # ✅ CNN only — single frame inference
        eye_img = gray[l_y1:l_y2, l_x1:l_x2]
        if eye_img.size > 0:
            eye_resized = cv2.resize(eye_img, (60, 36))
            tensor = torch.tensor(
                eye_resized, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 36, 60)

            with torch.no_grad():
                gaze = model(tensor).squeeze().numpy()

            # Smoothing buffer instead of LSTM
            gaze_buffer.append(gaze)
            smooth_gaze = np.mean(gaze_buffer, axis=0)

            # Relative gaze
            rel_x        = smooth_gaze[0] - CENTER_X
            rel_y        = smooth_gaze[1] - CENTER_Y
            displacement = np.sqrt(rel_x**2 + rel_y**2)

            # Arrow
            cx = int(lm[468].x * w)
            cy = int(lm[468].y * h)
            dx = int(rel_x * 100)
            dy = int(-rel_y * 100)
            cv2.arrowedLine(frame, (cx, cy),
                           (cx+dx, cy+dy),
                           (0, 0, 255), 2, tipLength=0.35)

            # Direction
            if displacement < DEAD_ZONE:
                direction       = "CENTER"
                direction_color = (0, 255, 0)
            elif abs(rel_x) >= abs(rel_y):
                if rel_x > LEFT_THRESH:
                    direction       = "<-- LEFT"
                    direction_color = (255, 100, 0)
                elif rel_x < -RIGHT_THRESH:
                    direction       = "RIGHT -->"
                    direction_color = (255, 100, 0)
                else:
                    direction       = "CENTER"
                    direction_color = (0, 255, 0)
            else:
                if rel_y < DOWN_THRESH:
                    direction       = "v DOWN v"
                    direction_color = (0, 100, 255)
                elif rel_y > UP_THRESH:
                    direction       = "^ UP ^"
                    direction_color = (0, 100, 255)
                else:
                    direction       = "CENTER"
                    direction_color = (0, 255, 0)

            gaze_text = (f"X:{smooth_gaze[0]:+.2f} "
                        f"Y:{smooth_gaze[1]:+.2f} "
                        f"relX:{rel_x:+.2f} "
                        f"relY:{rel_y:+.2f}")

        cv2.putText(frame, dist_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dist_color, 1)

    # Display
    cv2.putText(frame, gaze_text, (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, direction, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, direction_color, 2)
    cv2.putText(frame, f"Buffer: {len(gaze_buffer)}/8", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(frame, "CNN only | RPi4 Mode", (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    cv2.imshow('Gaze Estimation - RPi4', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()