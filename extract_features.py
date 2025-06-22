from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import time
from Roast import random_roast, alert_bad_posture, load_roasts

app = Flask(__name__)

# Load the trained model
model_path = 'posture_model.pkl'
model = joblib.load(model_path)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face.FaceMesh()

# Drawing styles
circle_spec = {'radius': 4, 'color': (0, 255, 0), 'thickness': -1}
line_color = (255, 0, 0)

def get_point(landmarks, idx):
    return np.array([landmarks[idx].x, landmarks[idx].y])

def calculate_angle(a, b, c):
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_features_and_draw(frame, pose_result, face_result):
    if not pose_result.pose_landmarks or not face_result.multi_face_landmarks:
        return None

    pose_lm = pose_result.pose_landmarks.landmark
    face_lm = face_result.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    try:
        # Key points
        left_eye = get_point(face_lm, 33)
        right_eye = get_point(face_lm, 263)
        eye_mid = (left_eye + right_eye) / 2

        left_shoulder = get_point(pose_lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_point(pose_lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        shoulder_mid = (left_shoulder + right_shoulder) / 2

        nose = get_point(pose_lm, mp_pose.PoseLandmark.NOSE.value)

        # Measurements
        dist_eye = np.linalg.norm(left_eye - right_eye)
        dist_shoulders = np.linalg.norm(left_shoulder - right_shoulder)
        dist_nose_midshoulder = np.linalg.norm(nose - shoulder_mid)

        angle1 = calculate_angle(eye_mid, shoulder_mid, nose)
        angle2 = calculate_angle(left_shoulder, shoulder_mid, right_shoulder)
        angle3 = calculate_angle(eye_mid, nose, shoulder_mid)

        for pt in [left_eye, right_eye, nose, left_shoulder, right_shoulder]:
            cv2.circle(frame, (int(pt[0]*w), int(pt[1]*h)), **circle_spec)
        cv2.line(frame,
                 (int(left_shoulder[0]*w), int(left_shoulder[1]*h)),
                 (int(right_shoulder[0]*w), int(right_shoulder[1]*h)),
                 line_color, 2)

        # Normalize distances
        dist_eye_norm = dist_eye / dist_shoulders if dist_shoulders > 0 else 0
        dist_nose_midshoulder_norm = dist_nose_midshoulder / dist_shoulders if dist_shoulders > 0 else 0

        features = [dist_eye_norm, dist_shoulders, dist_nose_midshoulder_norm,
                    angle1, angle2, angle3]

        return features

    except Exception:
        return None

def gen_frames():
    cap = cv2.VideoCapture(0)
    previous_posture = None
    last_alert_time = time.time()  # Track the last alert time
    alert_interval = 5  # Interval in seconds

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        face_result = face_mesh.process(rgb)
        features = extract_features_and_draw(frame, pose_result, face_result)

        # Annotate the frame with posture status
        if features:
            # Predict posture using the model
            features_df = np.array(features).reshape(1, -1)
            prediction = model.predict(features_df)[0]
            current_posture = "BAD" if prediction == 1 else "GOOD"

            # Annotate the frame with posture status
            cv2.putText(frame, f"Posture: {current_posture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if current_posture == "GOOD" else (0, 0, 255), 2)

            # Detect posture transition and trigger alert at intervals
            if current_posture == "BAD" and time.time() - last_alert_time >= alert_interval:
                alert_bad_posture(random_roast(load_roasts("PostureRoast.txt")))
                last_alert_time = time.time()  # Update the last alert time

            previous_posture = current_posture

        # Encode the frame as JPEG and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
