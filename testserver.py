from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import mediapipe as mp
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and label encoder
model = load_model("asl_model.h5")
label_map = np.load("label_encoder.npy", allow_pickle=True)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Global variable to track last prediction
last_prediction = {"letter": None}

# Function to extract normalized hand landmarks
def extract_landmarks(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        wrist = hand_landmarks.landmark[0]
        landmarks = [
            (lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z)
            for lm in hand_landmarks.landmark
        ]
        return np.array(landmarks).flatten()
    return None

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    landmarks = extract_landmarks(image_np)
    if landmarks is None:
        return {"prediction": "", "error": "No hand detected"}

    landmarks = landmarks.reshape(1, -1)
    prediction = model.predict(landmarks)[0]
    predicted_label = label_map[np.argmax(prediction)]

    # Avoid repeating the same prediction
    if predicted_label != last_prediction["letter"]:
        last_prediction["letter"] = predicted_label
        return {"prediction": predicted_label}
    else:
        return {"prediction": ""}  # Suppress duplicate

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, port=8000)
