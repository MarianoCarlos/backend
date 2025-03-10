import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import collections

# ✅ Load trained model & label encoder
model = tf.keras.models.load_model("asl_model.h5")
label_encoder = np.load("label_encoder.npy", allow_pickle=True)

# ✅ Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ✅ Webcam setup
cap = cv2.VideoCapture(0)

# ✅ Store last 10 predictions for smoothing
history = collections.deque(maxlen=10)
message = ""

def get_stable_prediction(prediction):
    history.append(prediction)
    most_common = max(set(history), key=history.count)
    return most_common

with hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                wrist = hand_landmarks.landmark[0]

                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z])

                # ✅ Ensure input has 63 features
                if len(landmarks) == 63:
                    # Predict ASL sign
                    prediction = model.predict(np.array([landmarks]))
                    class_index = np.argmax(prediction)
                    sign = label_encoder[class_index]

                    # ✅ Get stable prediction
                    stable_sign = get_stable_prediction(sign)

                    # ✅ Update message if a new letter is detected
                    if len(message) == 0 or message[-1] != stable_sign:
                        message += stable_sign

                    # ✅ Display prediction
                    cv2.putText(frame, f"Prediction: {stable_sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2, cv2.LINE_AA)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
        cv2.imshow("ASL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\n📩 Final Message:", message)
