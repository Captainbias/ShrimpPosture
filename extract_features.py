import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face.FaceMesh()

# Drawing settings
circle_spec = {'radius': 4, 'color': (0, 255, 0), 'thickness': -1}
line_color = (255, 0, 0)

# CSV setup
csv_path = 'posture_data.csv'
if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        headers = ['dist_eye', 'dist_shoulders', 'dist_nose_midshoulder',
                   'angle_eye_shoulder_nose', 'angle_shoulder_mid_nose', 'angle_eye_nose_mid',
                   'label']
        f.write(','.join(headers) + '\n')


def get_point(landmarks, idx):
    return np.array([landmarks[idx].x, landmarks[idx].y])


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def normalize_features(features, ref_dist):
    if ref_dist == 0:
        return features
    return [f / ref_dist for f in features]


def extract_features_and_draw(frame, pose_result, face_result):
    if not pose_result.pose_landmarks or not face_result.multi_face_landmarks:
        return None

    pose_lm = pose_result.pose_landmarks.landmark
    face_lm = face_result.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    try:
        left_eye = get_point(face_lm, 33)
        right_eye = get_point(face_lm, 263)
        eye_mid = (left_eye + right_eye) / 2
        left_shoulder = get_point(pose_lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_point(pose_lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        nose = get_point(pose_lm, mp_pose.PoseLandmark.NOSE.value)

        dist_eye = np.linalg.norm(left_eye - right_eye)
        dist_shoulders = np.linalg.norm(left_shoulder - right_shoulder)
        dist_nose_midshoulder = np.linalg.norm(nose - shoulder_mid)

        angle1 = calculate_angle(eye_mid, shoulder_mid, nose)
        angle2 = calculate_angle(left_shoulder, shoulder_mid, right_shoulder)
        angle3 = calculate_angle(eye_mid, nose, shoulder_mid)

        def draw_point(p): cv2.circle(frame, (int(p[0]*w), int(p[1]*h)), **circle_spec)
        for p in [left_eye, right_eye, nose, left_shoulder, right_shoulder]:
            draw_point(p)
        cv2.line(frame, (int(left_shoulder[0]*w), int(left_shoulder[1]*h)),
                 (int(right_shoulder[0]*w), int(right_shoulder[1]*h)), line_color, 2)

        features = [dist_eye, dist_shoulders, dist_nose_midshoulder, angle1, angle2, angle3]
        norm_features = normalize_features(features, dist_shoulders)
        return norm_features

    except Exception:
        return None


cap = cv2.VideoCapture(0)
print("Press 'g' = good, 'b' = bad, 'q' = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb)
    face_result = face_mesh.process(rgb)

    features = extract_features_and_draw(frame, pose_result, face_result)
    cv2.imshow("Posture Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in [ord('g'), ord('b')]:
        label = 0 if key == ord('g') else 1
        if features and all(np.isfinite(f) and f > 0 for f in features):
            print("Press 'y' to confirm saving this sample, or any other key to skip.")
            confirm = cv2.waitKey(0) & 0xFF
            if confirm == ord('y'):
                features.append(label)
                with open(csv_path, 'a') as f:
                    f.write(','.join(map(str, features)) + '\n')
                print(f"{'GOOD' if label == 0 else 'BAD'} posture recorded.")
            else:
                print("⏩ Sample skipped.")
        else:
            print("⚠ Noisy data. Skipped.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

