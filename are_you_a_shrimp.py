import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import extract_features
import time

# Load trained model
model = joblib.load('posture_model.pkl')

# Time interval (in seconds) to check posture
interval = 5
time_start = time.time()

# Initialize MediaPipe modules
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face.FaceMesh()

# Visuals
circle_spec = {'radius': 4, 'color': (0, 255, 0), 'thickness': -1}
line_color = (255, 0, 0)

# Webcam input
cap = cv2.VideoCapture(0)
print("🎥 Starting live posture check. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb)
    face_result = face_mesh.process(rgb)

    features = extract_features.extract_features_and_draw(frame, pose_result, face_result)

    # Show the camera feed
    cv2.imshow("Posture Check", frame)

    # Time-based inference
    time_curr = time.time()
    if (time_curr - time_start >= interval):
        if features:
            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            posture = "BAD" if prediction == 1 else "GOOD"
            print(f"\n Posture Detected: {posture} (checked every {interval} seconds)")
        else:
            print("\n Could not extract features. Posture check skipped.")

        time_start = time.time()

    # Exit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

   