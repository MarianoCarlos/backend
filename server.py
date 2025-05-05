import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
from io import BytesIO
from PIL import Image

# Initialize FastAPI
app = FastAPI()

# ✅ CORS Fix for Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model & label encoder
MODEL_PATH = "asl_model.h5"
ENCODER_PATH = "label_encoder.npy"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("❌ Model or label encoder file not found! Train the model first.")

model = tf.keras.models.load_model(MODEL_PATH)
label_encoder = np.load(ENCODER_PATH, allow_pickle=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(image: np.array):
    """Extracts 21 hand landmarks from an image using MediaPipe"""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                print("✅ Landmarks extracted:", landmarks)  # Debugging
                return landmarks  # Return first detected hand

        print("❌ No hand detected!")
        return None  # No hand detected
    except Exception as e:
        print(f"❌ Error in extracting landmarks: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handles image input, extracts hand landmarks, and predicts ASL letters."""
    try:
        # Read image file
        image_data = await file.read()
        image = np.array(Image.open(BytesIO(image_data)))

        # Extract landmarks
        landmarks = extract_landmarks(image)
        if landmarks is None:
            return {"error": "No hand detected"}

        # Predict ASL letter
        prediction = model.predict([landmarks])
        predicted_letter = label_encoder[np.argmax(prediction)]
        
        print(f"✅ Predicted Letter: {predicted_letter}")  # Debugging
        return {"prediction": predicted_letter}
    except Exception as e:
        print(f"❌ Server Error: {e}")
        return {"error": str(e)}

# Root route
@app.get("/")
def home():
    return {"message": "ASL Recognition API is running!"}
