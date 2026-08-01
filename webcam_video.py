import cv2
import torch
import numpy as np
from collections import deque
from video_model import GazeNetVideo
import mediapipe as mp
import os

# ── Load model ────────────────────────────────────────────────
model = GazeNetVideo()

# ✅ Auto select best available model
if os.path.exists('best_finetuned_model.pth'):
    model_path = 'best_finetuned_model.pth'
    print("✅ Using finetuned model")
else:
    model_path = 'best_video_model.pth'
    print("⚠️ Using base model (run finetune.py for better accuracy)")

model.load_state_dict(torch.load(model_path,
                                  map_location='cpu',
                                  weights_only=False))
model.eval()
print(f"📦 Model loaded: {model_path}")

# ── MediaPipe ─────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

SEQ_LEN      = 8
frame_buffer = deque(maxlen=SEQ_LEN)
gaze_buffer  = deque(maxlen=5)

# ── Calibration values ────────────────────────────────────────
CENTER_X     = -0.2802
CENTER_Y     = +0.1560
DEAD_ZONE    =  0.07
LEFT_THRESH  = +0.13
RIGHT_THRESH =  0.05
UP_THRESH    =  0.05
DOWN_THRESH  = -0.16

# ── Camera ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("🎥 Started — press Q to quit")
print("📏 Optimal distance: 40-70cm\n")

# ── Helpers ───────────────────────────────────────────────────
def get_eye_box(landmarks, indices, w, h, padding=20):
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

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results         = face_mesh.process(rgb)
    gaze_text       = "No face detected"
    direction       = ""
    direction_color = (0, 255, 0)
    dist_text       = ""
    dist_color      = (0, 255, 0)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # ── Distance ──────────────────────────────────────────
        face_width_px = abs(
            int(lm[454].x * w) - int(lm[234].x * w)
        )
        distance_cm = int((14 * 600) / max(face_width_px, 1))

        if distance_cm < 30:
            dist_color = (0, 0, 255)
            dist_text  = f"Distance: {distance_cm}cm TOO CLOSE"
        elif distance_cm > 80:
            dist_color = (0, 0, 255)
            dist_text  = f"Distance: {distance_cm}cm TOO FAR"
        else:
            dist_color = (0, 255, 0)
            dist_text  = f"Distance: {distance_cm}cm GOOD ✓"

        # ── Eye boxes ─────────────────────────────────────────
        l_x1, l_y1, l_x2, l_y2 = get_eye_box(lm,
            [33, 7, 163, 144, 145, 153, 154, 155, 133], w, h, padding=20)
        r_x1, r_y1, r_x2, r_y2 = get_eye_box(lm,
            [362, 382, 381, 380, 374, 373, 390, 249, 263], w, h, padding=20)

        cv2.rectangle(frame, (l_x1, l_y1), (l_x2, l_y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (r_x1, r_y1), (r_x2, r_y2), (0, 255, 0), 2)

        # ── Iris dots ─────────────────────────────────────────
        try:
            l_iris = get_iris_center(lm, LEFT_IRIS,  w, h)
            r_iris = get_iris_center(lm, RIGHT_IRIS, w, h)
            cv2.circle(frame, tuple(l_iris), 4, (0, 0, 255), -1)
            cv2.circle(frame, tuple(r_iris), 4, (0, 0, 255), -1)
        except:
            pass

        # ── Eye crop → model ──────────────────────────────────
        eye_img = gray[l_y1:l_y2, l_x1:l_x2]
        if eye_img.size > 0:
            eye_resized = cv2.resize(eye_img, (60, 36))
            frame_buffer.append(eye_resized)

        if len(frame_buffer) == SEQ_LEN:
            seq    = np.array(frame_buffer)
            tensor = torch.tensor(
                seq, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(2) / 255.0

            with torch.no_grad():
                gaze = model(tensor).squeeze().numpy()

            gaze_buffer.append(gaze)
            smooth_gaze = np.mean(gaze_buffer, axis=0)

            # ── Relative gaze ──────────────────────────────────
            rel_x = smooth_gaze[0] - CENTER_X
            rel_y = smooth_gaze[1] - CENTER_Y
            displacement = np.sqrt(rel_x**2 + rel_y**2)

            # ── Gaze arrow ────────────────────────────────────
            cx = int(lm[468].x * w)
            cy = int(lm[468].y * h)
            dx = int(rel_x * 150)
            dy = int(-rel_y * 150)
            cv2.arrowedLine(frame, (cx, cy),
                           (cx+dx, cy+dy),
                           (0, 0, 255), 3, tipLength=0.35)

            # ── Direction detection ───────────────────────────
            if displacement < DEAD_ZONE:
                direction       = "Looking CENTER"
                direction_color = (0, 255, 0)

            elif abs(rel_x) >= abs(rel_y):
                if rel_x > LEFT_THRESH:
                    direction       = "<-- Looking LEFT"
                    direction_color = (255, 100, 0)
                elif rel_x < -RIGHT_THRESH:
                    direction       = "Looking DOWN -->"
                    direction_color = (255, 100, 0)
                else:
                    direction       = "Looking CENTER"
                    direction_color = (0, 255, 0)

            else:
                if rel_y < DOWN_THRESH:
                    direction       = "v Looking DOWN v"
                    direction_color = (0, 100, 255)
                elif rel_y > UP_THRESH:
                    direction       = "^ Looking UP ^"
                    direction_color = (0, 100, 255)
                else:
                    direction       = "Looking CENTER"
                    direction_color = (0, 255, 0)

            gaze_text = (f"X:{smooth_gaze[0]:+.2f} "
                        f"Y:{smooth_gaze[1]:+.2f} "
                        f"relX:{rel_x:+.2f} "
                        f"relY:{rel_y:+.2f} "
                        f"disp:{displacement:.2f}")

        # ── Distance text ─────────────────────────────────────
        cv2.putText(frame, dist_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)

    # ── Display ───────────────────────────────────────────────
    cv2.putText(frame, gaze_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, direction, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, direction_color, 2)
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{SEQ_LEN}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame,
                f"Model: {os.path.basename(model_path)} | 1.22deg",
                (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow('Smart Glasses Gaze Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()