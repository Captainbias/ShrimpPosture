import cv2
import mediapipe as mp
import numpy as np
import os
import extract_features
import time

interval = 5 # time interval to check posture in seconds
time_start = time.time()
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face.FaceMesh()

# Drawing settings
circle_spec = {'radius': 4, 'color': (0, 255, 0), 'thickness': -1}
line_color = (255, 0, 0)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb)
    face_result = face_mesh.process(rgb)

    features = extract_features.extract_features_and_draw(frame, pose_result, face_result)
    cv2.imshow("Posture Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    time_curr = time.time()
    if (time_curr - time_start >= interval):
        # run the posture detection algorithm
        print("it's been", interval, "seconds")
        time_start = time.time()
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

   