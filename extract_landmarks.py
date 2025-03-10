import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import albumentations as A  # Install with: pip install albumentations

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define augmentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
])

# Dataset Paths
DATASET_PATH = "dataset"
CSV_FILE = "asl_landmarks1.csv"

def augment_image(image):
    """Applies augmentation to an image."""
    augmented = augmentations(image=image)
    return augmented["image"]

def extract_landmarks(image):
    """Extracts 21 hand landmarks (x, y, z) from an image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            wrist = hand_landmarks.landmark[0]  # Normalize based on wrist position
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z])
            return landmarks  # Return first detected hand
    return None  # No hand detected

def process_dataset():
    """Processes the dataset, extracts landmarks, and saves to CSV."""
    data = []
    labels = []

    print("\n📌 Processing dataset...\n")

    for label in sorted(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue

        print(f"🔤 Processing label: {label}")

        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"❌ Skipping (Cannot read): {image_path}")
                continue

            # Apply augmentation before extracting landmarks
            augmented_image = augment_image(image)
            landmarks = extract_landmarks(augmented_image)

            if landmarks:
                data.append(landmarks)
                labels.append(label)
                print(f"  ✔ Processed: {image_file}")
            else:
                print(f"  ⚠️ No hand detected: {image_file}")

    # Fix: Correct column names for 63 landmark features
    column_names = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
    df = pd.DataFrame(data, columns=column_names)
    df["label"] = labels
    df.to_csv(CSV_FILE, index=False)

    print(f"\n✅ Landmarks saved to {CSV_FILE}")

if __name__ == "__main__":
    process_dataset()
