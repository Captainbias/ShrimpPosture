# are_you_a_shrimp.py

# 📦 Imports: standard libraries + MediaPipe + your custom feature extractor
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import extract_features  # This should contain extract_features_and_draw()

#  ShrimpMonitor: tracks posture using a webcam feed and a trained ML model
class ShrimpMonitor:
    def __init__(self, model_path='posture_model.pkl', interval=5):
        """
        Initialize the posture monitor:
        - Loads the pre-trained model from disk
        - Sets the time interval between posture checks (default = 5 seconds)
        - Sets up MediaPipe pose + face landmarks
        - Starts the webcam
        """
        self.model = joblib.load(model_path)
        self.interval = interval
        self.previous_posture = None
        self.last_check = time.time()

        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.pose = self.mp_pose.Pose()
        self.face_mesh = self.mp_face.FaceMesh()

        self.cap = cv2.VideoCapture(0)  # Use default webcam (device 0)

    def get_status(self):
        """
        Run a single frame capture:
        - Extracts pose and face landmarks
        - Every `interval` seconds, runs the ML model to detect posture
        - Returns a transition flag:
            - True  = posture improved (bad → good)
            - False = posture declined (good → bad)
            - None  = no change or not enough time passed
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None  # If camera fails, return empty result

        # Process image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(rgb)
        face_result = self.face_mesh.process(rgb)

        # Extract custom features (your posture logic)
        features = extract_features.extract_features_and_draw(frame, pose_result, face_result)

        now = time.time()
        transition = None

        # Time-based inference: every X seconds
        if now - self.last_check >= self.interval and features:
            # Model prediction (posture = GOOD or BAD)
            features_df = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features_df)[0]
            current = "BAD" if prediction == 1 else "GOOD"

            # Detect change
            if self.previous_posture == "BAD" and current == "GOOD":
                transition = True  # improved posture
            elif self.previous_posture == "GOOD" and current == "BAD":
                transition = False  # declined posture

            self.previous_posture = current
            self.last_check = now

        # Return transition flag and the annotated frame
        return transition, frame

    def release(self):
        """
        Gracefully shut down the webcam and close all OpenCV windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()