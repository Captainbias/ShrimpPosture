from flask import Flask, Response, render_template
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialize camera and MediaPipe
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose 

def gen_frames():
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Mirror it if you want
            frame = cv2.flip(frame, 1)

            # Convert color for Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)

            # Draw landmarks back on BGR frame
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # JPEG‐encode and yield a multipart response
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/')
def index():
    # A simple page that shows the video stream
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
