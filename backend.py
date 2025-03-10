from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import collections
import json
import time

app = Flask(__name__)
CORS(app)

# ✅ Load trained model & label encoder
model = tf.keras.models.load_model("asl_model.h5")
label_encoder = np.load("label_encoder.npy", allow_pickle=True)

# ✅ Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

history = collections.deque(maxlen=10)
message = ""

def get_stable_prediction(prediction):
    history.append(prediction)
    return max(set(history), key=history.count)

def find_camera():
    """Try different camera indexes until one works."""
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Using camera index {i}")
            cap.release()
            return i
    print("❌ ERROR: No available webcam found!")
    return None

camera_index = find_camera()
if camera_index is None:
    exit()  # Stop execution if no camera is found

@app.route('/events')
def events():
    """Server-Sent Events (SSE) to send real-time ASL predictions to frontend."""
    def generate():
        global message
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("❌ ERROR: Could not open webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ ERROR: Could not read frame!")
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        wrist = hand_landmarks.landmark[0]

                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z])

                        if len(landmarks) == 63:
                            prediction = model.predict(np.array([landmarks]))
                            class_index = np.argmax(prediction)
                            sign = label_encoder[class_index]

                            stable_sign = get_stable_prediction(sign)

                            if len(message) == 0 or message[-1] != stable_sign:
                                message += stable_sign

                            # ✅ Send real-time data to frontend
                            data = json.dumps({"prediction": stable_sign, "message": message})
                            yield f"data: {data}\n\n"

                time.sleep(0.1)  # ✅ Prevent excessive updates

        except GeneratorExit:
            print("✅ SSE Connection closed")

        cap.release()

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
